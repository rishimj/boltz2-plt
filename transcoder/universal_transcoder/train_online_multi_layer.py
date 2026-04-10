"""Train multi-layer transcoders directly from streamed Boltz activations.

This script trains PLT (Per-Layer Transcoder) models on activations collected
from Boltz2 pairformer layers. Supports deterministic training with fixed seeds.

Usage:
    python train_online_multi_layer.py \
        --fasta /path/to/proteins \
        --layers 0 8 16 24 32 40 \
        --num_steps 1000 \
        --checkpoint_dir deterministic_checkpoints
"""

import argparse
import inspect
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def setup_determinism(seed: int = 42) -> None:
    """
    Set ALL random seeds for fully deterministic behavior.

    This ensures reproducible training and activation collection.

    Args:
        seed: Random seed to use for all RNGs
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch >= 1.8, enable deterministic algorithms
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Set environment variable for additional determinism
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Add boltz and collector modules to the path.
script_dir = Path(__file__).resolve().parent
boltz_root = script_dir.parent
sys.path.insert(0, str(boltz_root / "src"))
sys.path.insert(0, str(boltz_root / "collection_scripts"))

from boltz.model.models.boltz2 import Boltz2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.fasta import parse_fasta
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.types import Input
from boltz.data.mol import load_canonicals

from collect_multi_layer import MultiLayerActivationCollector
from universal_model import UniversalTranscoder


def _expand_pair_predictions(predictions, batch_size, num_tokens, num_pairs):
    """Broadcast token-level predictions across pair positions."""
    return predictions.unsqueeze(1).expand(batch_size * num_tokens, num_tokens, -1).reshape(batch_size * num_pairs, -1)


def _nmse_loss(prediction, target, eps=1e-8):
    """Normalized MSE: mean squared error divided by target variance."""
    mse = F.mse_loss(prediction, target)
    variance = target.var(unbiased=False).detach().clamp_min(eps)
    return mse / variance


def load_boltz_model(checkpoint_path, device, disable_msa_subsample=True):
    """Load the Boltz2 model used for streaming activation collection.

    Args:
        checkpoint_path: Path to Boltz2 checkpoint
        device: Device to load model on
        disable_msa_subsample: If True, disable MSA subsampling for determinism

    Returns:
        Loaded and configured Boltz2 model
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    all_hparams = checkpoint['hyper_parameters']
    valid_params = set(inspect.signature(Boltz2.__init__).parameters.keys()) - {'self'}
    hparams = {key: value for key, value in all_hparams.items() if key in valid_params}

    print(f"Creating model with {len(hparams)} valid hyperparameters...")
    model = Boltz2(**hparams)

    print("Loading weights from checkpoint...")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    # Disable MSA subsampling for deterministic behavior
    # MSA subsampling uses torch.randperm() which causes non-determinism
    if disable_msa_subsample and hasattr(model, 'msa_module'):
        original_value = getattr(model.msa_module, 'subsample_msa', None)
        model.msa_module.subsample_msa = False
        print(f"  MSA subsampling disabled (was: {original_value})")

    return model


def load_fasta_files(fasta_path, max_proteins):
    """Resolve FASTA files from either a single file or a directory."""
    fasta_path = Path(fasta_path).resolve()
    if fasta_path.is_file():
        fasta_files = [fasta_path]
    elif fasta_path.is_dir():
        fasta_files = sorted(fasta_path.glob("*.fasta")) + sorted(fasta_path.glob("*.fa"))
    else:
        raise ValueError(f"Invalid FASTA path: {fasta_path}")

    if not fasta_files:
        raise ValueError(f"No FASTA files found at {fasta_path}")

    if max_proteins > 0:
        fasta_files = fasta_files[:max_proteins]

    return fasta_files


def build_features(fasta_file, molecules, moldir, tokenizer, featurizer, device, recycling_steps):
    """Parse and featurize one FASTA entry into a model input batch."""
    target = parse_fasta(fasta_file, molecules, moldir, boltz2=True)

    msa_dict = {}
    for chain in target.record.chains:
        if chain.msa_id and chain.msa_id != -1:
            msa_path = Path(chain.msa_id)
            if not msa_path.is_absolute():
                msa_path = (fasta_file.parent / msa_path).resolve()

            if msa_path.exists():
                msa = parse_a3m(msa_path, taxonomy=None)
                msa_dict[chain.chain_name] = msa

    input_data = Input(
        structure=target.structure,
        msa=msa_dict,
        record=target.record,
        residue_constraints=target.residue_constraints,
        templates=target.templates,
        extra_mols=target.extra_mols,
    )

    tokens = tokenizer.tokenize(input_data)
    random_generator = np.random.default_rng(42)
    features = featurizer.process(
        data=tokens,
        molecules=molecules,
        random=random_generator,
        training=False,
        max_seqs=128,
    )

    feats = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            feats[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, np.ndarray):
            feats[key] = torch.from_numpy(value).unsqueeze(0).to(device)
        else:
            feats[key] = value

    return feats


@dataclass
class LayerTrainer:
    """Stateful trainer for one transcoder layer."""

    layer_idx: int
    d_model: int
    d_hidden: int
    d_pair: int
    k: int
    auxk: int
    batch_size: int
    dead_steps_threshold: int
    lr: float
    log_every: int
    device: torch.device
    model: UniversalTranscoder = field(init=False)
    optimizer: torch.optim.Optimizer = field(init=False)
    step_count: int = field(default=0, init=False)
    metrics_history: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.model = UniversalTranscoder(
            d_model=self.d_model,
            d_hidden=self.d_hidden,
            d_pair=self.d_pair,
            k=self.k,
            auxk=self.auxk,
            batch_size=self.batch_size,
            dead_steps_threshold=self.dead_steps_threshold,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
        )

    @property
    def done(self):
        return self.step_count >= self.max_steps

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps

    def train_on_batch(self, batch):
        if batch is None or self.step_count >= self.max_steps:
            return None

        self.model.train()

        s1 = batch['input_s'].to(self.device)
        s2 = batch['output_s'].to(self.device)
        y1_true = batch['input_z'].to(self.device)
        y2_true = batch['output_z'].to(self.device)

        batch_size, num_tokens, _ = s1.shape
        num_pairs = y1_true.shape[1]

        s1_flat = s1.reshape(batch_size * num_tokens, -1)
        s2_flat = s2.reshape(batch_size * num_tokens, -1)
        y1_true_flat = y1_true.reshape(batch_size * num_pairs, -1)
        y2_true_flat = y2_true.reshape(batch_size * num_pairs, -1)

        y1_pred1, y2_pred1, aux_y1_1, aux_y2_1, dead_mask = self.model(s1_flat)
        y1_pred2, y2_pred2, aux_y1_2, aux_y2_2, _ = self.model(s2_flat)

        y1_pred1_expanded = _expand_pair_predictions(y1_pred1, batch_size, num_tokens, num_pairs)
        y2_pred1_expanded = _expand_pair_predictions(y2_pred1, batch_size, num_tokens, num_pairs)
        y1_pred2_expanded = _expand_pair_predictions(y1_pred2, batch_size, num_tokens, num_pairs)
        y2_pred2_expanded = _expand_pair_predictions(y2_pred2, batch_size, num_tokens, num_pairs)

        loss_recon_y1_from_s1 = _nmse_loss(y1_pred1_expanded, y1_true_flat)
        loss_recon_y2_from_s1 = _nmse_loss(y2_pred1_expanded, y2_true_flat)
        loss_recon_y1_from_s2 = _nmse_loss(y1_pred2_expanded, y1_true_flat)
        loss_recon_y2_from_s2 = _nmse_loss(y2_pred2_expanded, y2_true_flat)

        loss_reconstruction = (
            loss_recon_y1_from_s1 +
            loss_recon_y2_from_s1 +
            loss_recon_y1_from_s2 +
            loss_recon_y2_from_s2
        )

        var_y1 = y1_true_flat.var(unbiased=False).detach().clamp_min(1e-8)
        var_y2 = y2_true_flat.var(unbiased=False).detach().clamp_min(1e-8)

        loss_consistency_y1 = F.mse_loss(y1_pred1, y1_pred2) / var_y1
        loss_consistency_y2 = F.mse_loss(y2_pred1, y2_pred2) / var_y2
        loss_consistency = loss_consistency_y1 + loss_consistency_y2

        loss_auxk = torch.tensor(0.0, device=self.device)
        auxk_coef = 1.0 / 32.0

        if aux_y1_1 is not None and dead_mask.sum() > 0:
            residual_y1_1 = (y1_true_flat - y1_pred1_expanded).detach()
            residual_y2_1 = (y2_true_flat - y2_pred1_expanded).detach()

            aux_y1_1_expanded = _expand_pair_predictions(aux_y1_1, batch_size, num_tokens, num_pairs)
            aux_y2_1_expanded = _expand_pair_predictions(aux_y2_1, batch_size, num_tokens, num_pairs)

            loss_auxk += _nmse_loss(aux_y1_1_expanded, residual_y1_1) * auxk_coef
            loss_auxk += _nmse_loss(aux_y2_1_expanded, residual_y2_1) * auxk_coef

            residual_y1_2 = (y1_true_flat - y1_pred2_expanded).detach()
            residual_y2_2 = (y2_true_flat - y2_pred2_expanded).detach()

            aux_y1_2_expanded = _expand_pair_predictions(aux_y1_2, batch_size, num_tokens, num_pairs)
            aux_y2_2_expanded = _expand_pair_predictions(aux_y2_2, batch_size, num_tokens, num_pairs)

            loss_auxk += _nmse_loss(aux_y1_2_expanded, residual_y1_2) * auxk_coef
            loss_auxk += _nmse_loss(aux_y2_2_expanded, residual_y2_2) * auxk_coef

        loss_total = loss_reconstruction + loss_consistency + loss_auxk

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        self.model.norm_weights()

        self.step_count += 1

        metrics = {
            'step': self.step_count,
            'loss_total': float(loss_total.item()),
            'loss_reconstruction': float(loss_reconstruction.item()),
            'loss_consistency': float(loss_consistency.item()),
            'loss_auxk': float(loss_auxk.item()),
            'dead_neurons': int(dead_mask.sum().item()),
        }
        self.metrics_history.append(metrics)

        return metrics

    def save_checkpoint(self, checkpoint_dir, extra_metadata=None):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / 'universal_transcoder_final.pt'
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step_count,
            'training_time': extra_metadata.get('training_time_seconds') if extra_metadata else None,
            'final_metrics': self.metrics_history[-1] if self.metrics_history else None,
            'hyperparameters': {
                'd_model': self.d_model,
                'd_hidden': self.d_hidden,
                'd_pair': self.d_pair,
                'k': self.k,
                'auxk': self.auxk,
                'batch_size': self.batch_size,
                'dead_steps_threshold': self.dead_steps_threshold,
                'lr': self.lr,
                'layer_idx': self.layer_idx,
                'max_steps': self.max_steps,
            },
            'metrics_history': self.metrics_history,
        }
        if extra_metadata:
            checkpoint['metadata'] = extra_metadata

        torch.save(checkpoint, checkpoint_path)

        metrics_path = checkpoint_dir / 'training_metrics.json'
        with open(metrics_path, 'w') as handle:
            json.dump({
                'hyperparameters': checkpoint['hyperparameters'],
                'metadata': extra_metadata or {},
                'final_metrics': self.metrics_history[-1] if self.metrics_history else None,
                'all_metrics': self.metrics_history,
            }, handle, indent=2)

        return checkpoint_path, metrics_path


def train_online_multi_layer(
    checkpoint_path,
    fasta_path,
    checkpoint_base_dir,
    layer_indices,
    num_steps=100,
    batch_size=10,
    lr=1e-3,
    log_every=10,
    d_model=384,
    d_hidden=2048,
    d_pair=128,
    k=16,
    auxk=32,
    dead_steps_threshold=10000,
    max_proteins=0,
    device='cuda',
    recycling_steps=0,
    seed=42,
    deterministic=True,
):
    """Train transcoders from streamed activations without saving collections.

    Args:
        checkpoint_path: Path to Boltz2 checkpoint
        fasta_path: Path to FASTA file or directory
        checkpoint_base_dir: Directory to save transcoder checkpoints
        layer_indices: List of pairformer layer indices to train
        num_steps: Number of training steps per layer
        batch_size: Batch size for training
        lr: Learning rate
        log_every: Log metrics every N steps
        d_model: Input dimension (single representation)
        d_hidden: Latent dimension (number of features)
        d_pair: Pair representation dimension
        k: Top-K activation
        auxk: Auxiliary K for dead neurons
        dead_steps_threshold: Steps before neuron considered dead
        max_proteins: Limit proteins per epoch (0 = use all)
        device: Device to use
        recycling_steps: Number of recycling steps
        seed: Random seed for determinism
        deterministic: If True, enable deterministic mode
    """
    # Set up determinism FIRST before any other operations
    if deterministic:
        print("\nSetting up deterministic mode...")
        setup_determinism(seed)
        print(f"  Seed: {seed}")
        print("  cuDNN deterministic: enabled")
        print("  MSA subsampling: will be disabled")

    print()
    print("=" * 80)
    print("ONLINE MULTI-LAYER PLT TRAINING")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"FASTA source: {fasta_path}")
    print(f"Checkpoint directory: {checkpoint_base_dir}")
    print(f"Layers: {layer_indices}")
    print(f"Steps per layer: {num_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Model: d_model={d_model}, d_hidden={d_hidden}, k={k}, auxk={auxk}")
    print(f"Deterministic: {deterministic}, Seed: {seed}")
    print("=" * 80)

    device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    print(f"Using device: {device}")
    print()

    boltz_model = load_boltz_model(checkpoint_path, device, disable_msa_subsample=deterministic)
    tokenizer = Boltz2Tokenizer()
    featurizer = Boltz2Featurizer()

    moldir = Path("/usr/scratch/rmanimaran8/boltz/.boltz_cache/mols")
    if not moldir.exists():
        moldir = Path.home() / ".boltz_cache" / "mols"
    if not moldir.exists():
        raise ValueError(f"Molecules directory not found. Expected at {moldir}")
    molecules = load_canonicals(str(moldir))
    print(f"Loaded {len(molecules)} canonical molecules")
    print()

    fasta_files = load_fasta_files(fasta_path, max_proteins)
    print(f"Found {len(fasta_files)} FASTA file(s) for streaming")
    print()

    model_layer_count = len(boltz_model.pairformer_module.layers)
    invalid_layers = [idx for idx in layer_indices if idx < 0 or idx >= model_layer_count]
    if invalid_layers:
        raise ValueError(f"Invalid layer indices for model with {model_layer_count} layers: {invalid_layers}")

    trainers = {}
    for layer_idx in layer_indices:
        trainer = LayerTrainer(
            layer_idx=layer_idx,
            d_model=d_model,
            d_hidden=d_hidden,
            d_pair=d_pair,
            k=k,
            auxk=auxk,
            batch_size=batch_size,
            dead_steps_threshold=dead_steps_threshold,
            lr=lr,
            log_every=log_every,
            device=device,
        )
        trainer.set_max_steps(num_steps)
        trainers[layer_idx] = trainer

    collector = MultiLayerActivationCollector(boltz_model, layer_indices, device=device)

    def all_done():
        return all(trainer.step_count >= num_steps for trainer in trainers.values())

    start_time = time.time()
    protein_count = 0
    epoch = 0

    try:
        while not all_done():
            epoch += 1
            epoch_files = list(fasta_files)
            rng = np.random.default_rng(42 + epoch)
            rng.shuffle(epoch_files)

            epoch_updates = 0

            for fasta_file in epoch_files:
                if all_done():
                    break

                try:
                    print(f"\nProcessing {fasta_file.name} (epoch {epoch})")
                    feats = build_features(
                        fasta_file=fasta_file,
                        molecules=molecules,
                        moldir=moldir,
                        tokenizer=tokenizer,
                        featurizer=featurizer,
                        device=device,
                        recycling_steps=recycling_steps,
                    )

                    forward_error = None
                    try:
                        with torch.no_grad():
                            boltz_model(feats, recycling_steps=recycling_steps)
                    except Exception as error:
                        # Some Boltz forward failures occur after transition hooks fire.
                        # We still consume collected activations if available.
                        forward_error = error
                        print(f"  Note: model forward encountered {error}; attempting to use collected activations.")

                    layer_batches = collector.pop_batches()
                    has_any_batch = any(batch is not None for batch in layer_batches.values())

                    if not has_any_batch:
                        if forward_error is not None:
                            print("  No activations were collected; skipping this protein.")
                        collector.clear_activations()
                        continue

                    protein_count += 1
                    epoch_updates += 1

                    for layer_idx, batch in layer_batches.items():
                        trainer = trainers[layer_idx]
                        if trainer.step_count >= num_steps or batch is None:
                            continue

                        metrics = trainer.train_on_batch(batch)
                        if metrics is None:
                            continue

                        if trainer.step_count % log_every == 0:
                            print()
                            print(f"Step {trainer.step_count}/{num_steps} for layer {layer_idx} (epoch {epoch})")
                            print(f"  Total Loss: {metrics['loss_total']:.6f}")
                            print(f"  Reconstruction Loss: {metrics['loss_reconstruction']:.6f}")
                            print(f"  Consistency Loss: {metrics['loss_consistency']:.6f}")
                            print(f"  AuxK Loss: {metrics['loss_auxk']:.6f}")
                            print(f"  Dead Neurons: {metrics['dead_neurons']} / {d_hidden}")

                except Exception as error:
                    print(f"Error processing {fasta_file.name}: {error}")
                    import traceback
                    traceback.print_exc()
                    collector.clear_activations()
                    continue

            if epoch_updates == 0:
                raise RuntimeError("No proteins could be processed in this epoch; check the FASTA inputs and Boltz checkpoint.")

    finally:
        collector.remove_hooks()

    training_time = time.time() - start_time

    print()
    print("=" * 80)
    print("ONLINE TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total training time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")
    print(f"Proteins processed: {protein_count}")
    print()

    os.makedirs(checkpoint_base_dir, exist_ok=True)
    all_results = []

    for layer_idx, trainer in trainers.items():
        layer_dir = Path(checkpoint_base_dir) / f"layer_{layer_idx:02d}"
        ckpt_path, metrics_path = trainer.save_checkpoint(
            layer_dir,
            extra_metadata={
                'timestamp': datetime.now().isoformat(),
                'training_time_seconds': training_time,
                'proteins_processed': protein_count,
                'epoch_count': epoch,
                'streaming_mode': True,
                'deterministic': deterministic,
                'seed': seed,
                'fasta_path': str(fasta_path),
                'checkpoint_dir': str(layer_dir),
            },
        )

        final_metrics = trainer.metrics_history[-1] if trainer.metrics_history else {}
        all_results.append({
            'layer_idx': layer_idx,
            'status': 'success' if trainer.metrics_history else 'skipped',
            'training_time_seconds': training_time,
            'final_loss_total': final_metrics.get('loss_total'),
            'final_loss_reconstruction': final_metrics.get('loss_reconstruction'),
            'final_loss_consistency': final_metrics.get('loss_consistency'),
            'final_loss_auxk': final_metrics.get('loss_auxk'),
            'final_dead_neurons': final_metrics.get('dead_neurons'),
            'num_steps': trainer.step_count,
            'checkpoint_dir': str(layer_dir),
            'checkpoint_path': str(ckpt_path),
            'metrics_path': str(metrics_path),
        })

        print(f"✓ Saved layer {layer_idx} checkpoint to {ckpt_path}")

    summary_path = Path(checkpoint_base_dir) / 'online_multi_layer_training_summary.json'
    with open(summary_path, 'w') as handle:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'proteins_processed': protein_count,
            'epoch_count': epoch,
            'streaming_mode': True,
            'deterministic': deterministic,
            'seed': seed,
            'fasta_path': str(fasta_path),
            'checkpoint_dir': str(checkpoint_base_dir),
            'layers': layer_indices,
            'results': all_results,
            'hyperparameters': {
                'num_steps': num_steps,
                'batch_size': batch_size,
                'lr': lr,
                'd_model': d_model,
                'd_hidden': d_hidden,
                'd_pair': d_pair,
                'k': k,
                'auxk': auxk,
                'dead_steps_threshold': dead_steps_threshold,
                'max_proteins': max_proteins,
                'recycling_steps': recycling_steps,
                'seed': seed,
                'deterministic': deterministic,
            },
        }, handle, indent=2)

    print(f"✓ Training summary saved to: {summary_path}")
    print()

    return {
        'training_time_seconds': training_time,
        'proteins_processed': protein_count,
        'epoch_count': epoch,
        'layers': all_results,
        'summary_path': str(summary_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train multi-layer transcoders from streamed Boltz activations")
    parser.add_argument('--checkpoint', type=str,
                        default='/usr/scratch/rmanimaran8/boltz/.boltz_cache/boltz2_conf.ckpt',
                        help='Path to Boltz2 checkpoint')
    parser.add_argument('--fasta', type=str, required=True,
                        help='Path to FASTA file or directory with FASTA files')
    parser.add_argument('--checkpoint_dir', type=str, default='online_multi_layer_checkpoints',
                        help='Base directory to save transcoder checkpoints')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 8, 16, 24, 32, 40],
                        help='Layer indices to train (default: 0 8 16 24 32 40)')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of training steps per layer')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--d_model', type=int, default=384,
                        help='Input dimension (single representation)')
    parser.add_argument('--d_hidden', type=int, default=2048,
                        help='Latent dimension (number of features)')
    parser.add_argument('--d_pair', type=int, default=128,
                        help='Pair representation dimension')
    parser.add_argument('--k', type=int, default=16,
                        help='Top-K activation')
    parser.add_argument('--auxk', type=int, default=32,
                        help='Auxiliary K for dead neurons')
    parser.add_argument('--dead_steps_threshold', type=int, default=10000,
                        help='Steps before neuron considered dead')
    parser.add_argument('--max-proteins', type=int, default=0,
                        help='Limit proteins per epoch (0 means use all files)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--recycling-steps', type=int, default=0,
                        help='Number of recycling steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for determinism')
    parser.add_argument('--no-deterministic', action='store_true',
                        help='Disable deterministic mode (faster but non-reproducible)')

    args = parser.parse_args()

    train_online_multi_layer(
        checkpoint_path=args.checkpoint,
        fasta_path=args.fasta,
        checkpoint_base_dir=args.checkpoint_dir,
        layer_indices=args.layers,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=args.log_every,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        d_pair=args.d_pair,
        k=args.k,
        auxk=args.auxk,
        dead_steps_threshold=args.dead_steps_threshold,
        max_proteins=args.max_proteins,
        device=args.device,
        recycling_steps=args.recycling_steps,
        seed=args.seed,
        deterministic=not args.no_deterministic,
    )


if __name__ == '__main__':
    main()