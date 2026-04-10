"""
Deterministic Baseline for Boltz2

This script provides utilities for running Boltz2 with full determinism,
enabling reproducible activations and structure outputs for PLT training
and verification.

Sources of randomness in Boltz that need to be controlled:
1. MSA subsampling (trunkv2.py:632) - torch.randperm()
2. Dropout (dropout.py:32) - disabled in eval mode
3. Diffusion init noise (diffusionv2.py:347) - torch.randn()
4. Diffusion step noise (diffusionv2.py:379) - torch.randn()
5. Random augmentation (utils.py:51) - torch.randn()
6. Random rotations (utils.py:281) - torch.randn()
7. Sigma sampling (diffusionv2.py:542) - torch.randn()

Usage:
    python deterministic_baseline.py --fasta input.fasta --output results/
"""

import argparse
import inspect
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

# Add paths
script_dir = Path(__file__).resolve().parent
boltz_root = script_dir.parent.parent
sys.path.insert(0, str(boltz_root / "src"))
sys.path.insert(0, str(boltz_root))

from boltz.model.models.boltz2 import Boltz2
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.a3m import parse_a3m
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_canonicals
from boltz.data.types import Input


def setup_determinism(seed: int = 42) -> None:
    """
    Set ALL random seeds for fully deterministic behavior.

    This must be called before any random operations to ensure reproducibility.

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


def prepare_deterministic_model(model: Boltz2, disable_msa_subsample: bool = True) -> Boltz2:
    """
    Configure model for deterministic inference.

    Args:
        model: Boltz2 model instance
        disable_msa_subsample: If True, disable MSA subsampling which uses torch.randperm

    Returns:
        Configured model
    """
    model.eval()

    # Disable MSA subsampling (uses torch.randperm without seed)
    if disable_msa_subsample:
        if hasattr(model, 'msa_module'):
            original_subsample = getattr(model.msa_module, 'subsample_msa', None)
            model.msa_module.subsample_msa = False
            print(f"  MSA subsampling disabled (was: {original_subsample})")

    return model


def load_boltz_model(checkpoint_path: str, device: str = 'cuda') -> Boltz2:
    """
    Load Boltz2 model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded Boltz2 model
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

    num_layers = len(model.pairformer_module.layers)
    print(f"Model loaded (pairformer has {num_layers} layers)")

    return model


def featurize_input(
    fasta_path: Path,
    molecules: dict,
    moldir: Path,
    tokenizer: Boltz2Tokenizer,
    featurizer: Boltz2Featurizer,
    device: str,
    seed: int = 42
) -> Tuple[Dict[str, torch.Tensor], Any]:
    """
    Parse and featurize a FASTA file with deterministic processing.

    Args:
        fasta_path: Path to FASTA file
        molecules: Canonical molecules dictionary
        moldir: Path to molecules directory
        tokenizer: Boltz2 tokenizer
        featurizer: Boltz2 featurizer
        device: Device to put tensors on
        seed: Random seed for featurizer

    Returns:
        Tuple of (feature dict, target object)
    """
    # Parse FASTA
    target = parse_fasta(fasta_path, molecules, moldir, boltz2=True)

    # Load MSAs if specified
    msa_dict = {}
    for chain in target.record.chains:
        if chain.msa_id and chain.msa_id != -1:
            msa_path = Path(chain.msa_id)
            if not msa_path.is_absolute():
                msa_path = (fasta_path.parent / msa_path).resolve()

            if msa_path.exists():
                msa = parse_a3m(msa_path, taxonomy=None)
                msa_dict[chain.chain_name] = msa
                print(f"  Loaded MSA for chain {chain.chain_name}: {len(msa.sequences)} sequences")

    # Create Input object
    input_data = Input(
        structure=target.structure,
        msa=msa_dict,
        record=target.record,
        residue_constraints=target.residue_constraints,
        templates=target.templates,
        extra_mols=target.extra_mols,
    )

    # Tokenize
    tokens = tokenizer.tokenize(input_data)

    # Featurize with fixed seed
    random_generator = np.random.default_rng(seed)
    features = featurizer.process(
        data=tokens,
        molecules=molecules,
        random=random_generator,
        training=False,
        max_seqs=128,
    )

    # Convert to batch format
    feats = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            feats[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, np.ndarray):
            feats[key] = torch.from_numpy(value).unsqueeze(0).to(device)
        else:
            feats[key] = value

    return feats, target


class DeterministicActivationCapture:
    """
    Capture activations from specified layers with deterministic execution.

    Registers hooks on transition_s and transition_z modules to capture
    both input and output activations.
    """

    def __init__(
        self,
        model: Boltz2,
        layer_indices: List[int],
        device: str = 'cuda'
    ):
        """
        Initialize activation capture.

        Args:
            model: Boltz2 model
            layer_indices: List of pairformer layer indices to capture
            device: Device for captured tensors
        """
        self.model = model
        self.layer_indices = layer_indices
        self.device = device

        self.activations: Dict[int, Dict[str, torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on target layers."""
        for layer_idx in self.layer_indices:
            if layer_idx >= len(self.model.pairformer_module.layers):
                print(f"  Warning: Layer {layer_idx} out of range, skipping")
                continue

            layer = self.model.pairformer_module.layers[layer_idx]
            self.activations[layer_idx] = {}

            # Hooks for transition_s (single representation)
            def make_hook_s_input(idx):
                def hook(module, input, output):
                    self.activations[idx]['input_s'] = input[0].detach().cpu().clone()
                return hook

            def make_hook_s_output(idx):
                def hook(module, input, output):
                    self.activations[idx]['output_s'] = output.detach().cpu().clone()
                return hook

            # Hooks for transition_z (pair representation)
            def make_hook_z_input(idx):
                def hook(module, input, output):
                    self.activations[idx]['input_z'] = input[0].detach().cpu().clone()
                return hook

            def make_hook_z_output(idx):
                def hook(module, input, output):
                    self.activations[idx]['output_z'] = output.detach().cpu().clone()
                return hook

            # Register hooks
            if hasattr(layer, 'transition_s'):
                self.hooks.append(
                    layer.transition_s.register_forward_hook(make_hook_s_input(layer_idx))
                )
                self.hooks.append(
                    layer.transition_s.register_forward_hook(make_hook_s_output(layer_idx))
                )

            if hasattr(layer, 'transition_z'):
                self.hooks.append(
                    layer.transition_z.register_forward_hook(make_hook_z_input(layer_idx))
                )
                self.hooks.append(
                    layer.transition_z.register_forward_hook(make_hook_z_output(layer_idx))
                )

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self) -> None:
        """Clear captured activations."""
        for layer_idx in self.activations:
            self.activations[layer_idx] = {}

    def get_activations(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Return captured activations."""
        return self.activations


def run_deterministic_forward(
    model: Boltz2,
    feats: Dict[str, torch.Tensor],
    seed: int = 42,
    recycling_steps: int = 0
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Run forward pass with seed reset for full reproducibility.

    CRITICAL: Seeds are reset immediately before forward pass to ensure
    all torch.randn() calls in diffusion produce identical results.

    Args:
        model: Boltz2 model
        feats: Input features
        seed: Random seed to reset before forward
        recycling_steps: Number of recycling steps

    Returns:
        Model output dict or None if forward fails
    """
    # Reset seed immediately before forward pass
    setup_determinism(seed)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            try:
                output = model(feats=feats, recycling_steps=recycling_steps)
                return output
            except Exception as e:
                print(f"  Forward pass error: {e}")
                return None


def compare_activations(
    act1: Dict[int, Dict[str, torch.Tensor]],
    act2: Dict[int, Dict[str, torch.Tensor]],
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Compare activations from two runs.

    Args:
        act1: Activations from run 1
        act2: Activations from run 2
        tolerance: Maximum allowed difference for identical comparison

    Returns:
        Comparison results dict
    """
    results = {
        'identical': True,
        'layers': {},
        'max_diff': 0.0
    }

    for layer_idx in sorted(set(act1.keys()) | set(act2.keys())):
        layer_results = {}

        for key in ['input_s', 'output_s', 'input_z', 'output_z']:
            if key in act1.get(layer_idx, {}) and key in act2.get(layer_idx, {}):
                t1 = act1[layer_idx][key]
                t2 = act2[layer_idx][key]

                max_diff = torch.abs(t1 - t2).max().item()
                mean_diff = torch.abs(t1 - t2).mean().item()
                identical = max_diff < tolerance

                layer_results[key] = {
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'identical': identical,
                    'shape': list(t1.shape)
                }

                if not identical:
                    results['identical'] = False

                results['max_diff'] = max(results['max_diff'], max_diff)

        results['layers'][layer_idx] = layer_results

    return results


def print_comparison_results(results: Dict[str, Any]) -> None:
    """Pretty print comparison results."""
    print("\n" + "=" * 70)
    print("ACTIVATION COMPARISON RESULTS")
    print("=" * 70)

    if results['identical']:
        print("\n  PERFECT REPRODUCIBILITY - All activations identical!")
    else:
        print(f"\n  Non-identical activations detected (max diff: {results['max_diff']:.2e})")

    print("\nPer-layer breakdown:")
    for layer_idx, layer_data in sorted(results['layers'].items()):
        print(f"\n  Layer {layer_idx}:")
        for key, stats in layer_data.items():
            status = "PASS" if stats['identical'] else "FAIL"
            print(f"    {key}: max={stats['max_diff']:.2e}, mean={stats['mean_diff']:.2e} [{status}]")


def run_baseline_test(
    checkpoint_path: str,
    fasta_path: str,
    output_dir: str,
    layer_indices: List[int] = [0, 8, 16, 24, 32, 40],
    device: str = 'cuda',
    seed: int = 42,
    recycling_steps: int = 0,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Run deterministic baseline test.

    Runs Boltz2 twice with identical configuration and verifies
    that activations are perfectly reproducible.

    Args:
        checkpoint_path: Path to Boltz2 checkpoint
        fasta_path: Path to FASTA file
        output_dir: Directory to save results
        layer_indices: Layer indices to capture
        device: Device to use
        seed: Random seed
        recycling_steps: Number of recycling steps
        tolerance: Tolerance for identical comparison

    Returns:
        Test results dict
    """
    print("\n" + "=" * 70)
    print("DETERMINISTIC BASELINE TEST")
    print("=" * 70)
    print(f"Input: {fasta_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Layers: {layer_indices}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer and featurizer
    tokenizer = Boltz2Tokenizer()
    featurizer = Boltz2Featurizer()

    # Load molecules
    moldir = Path("/usr/scratch/rmanimaran8/boltz/.boltz_cache/mols")
    if not moldir.exists():
        moldir = Path.home() / ".boltz_cache" / "mols"
    if not moldir.exists():
        raise ValueError(f"Molecules directory not found. Expected at {moldir}")
    molecules = load_canonicals(str(moldir))
    print(f"Loaded {len(molecules)} canonical molecules")

    fasta_path = Path(fasta_path)

    # ===== RUN 1 =====
    print("\n" + "-" * 70)
    print("RUN 1: First deterministic forward pass")
    print("-" * 70)

    setup_determinism(seed)

    model = load_boltz_model(checkpoint_path, device)
    model = prepare_deterministic_model(model, disable_msa_subsample=True)

    # Create activation capture
    capture1 = DeterministicActivationCapture(model, layer_indices, device)

    # Featurize with fixed seed
    print("\nFeaturizing input...")
    feats, target = featurize_input(
        fasta_path, molecules, moldir, tokenizer, featurizer, device, seed
    )

    # Run forward pass
    print("\nRunning forward pass...")
    output1 = run_deterministic_forward(model, feats, seed, recycling_steps)

    # Get activations
    act1 = {k: {k2: v2.clone() for k2, v2 in v.items()}
            for k, v in capture1.get_activations().items()}

    capture1.remove_hooks()

    # Report shapes
    for layer_idx in sorted(act1.keys()):
        layer_act = act1[layer_idx]
        print(f"  Layer {layer_idx}:")
        for key, tensor in layer_act.items():
            print(f"    {key}: {list(tensor.shape)}")

    # ===== RUN 2 =====
    print("\n" + "-" * 70)
    print("RUN 2: Second deterministic forward pass (verification)")
    print("-" * 70)

    # Full reset
    setup_determinism(seed)

    # Reload model fresh
    model = load_boltz_model(checkpoint_path, device)
    model = prepare_deterministic_model(model, disable_msa_subsample=True)

    # Create new capture
    capture2 = DeterministicActivationCapture(model, layer_indices, device)

    # Re-featurize with same seed
    print("\nFeaturizing input (should be identical)...")
    feats2, _ = featurize_input(
        fasta_path, molecules, moldir, tokenizer, featurizer, device, seed
    )

    # Run forward pass
    print("\nRunning forward pass...")
    output2 = run_deterministic_forward(model, feats2, seed, recycling_steps)

    # Get activations
    act2 = capture2.get_activations()
    capture2.remove_hooks()

    # ===== COMPARE =====
    print("\nComparing activations between runs...")
    comparison = compare_activations(act1, act2, tolerance)
    print_comparison_results(comparison)

    # ===== SAVE RESULTS =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save comparison results
    results = {
        'timestamp': timestamp,
        'fasta_path': str(fasta_path),
        'checkpoint_path': checkpoint_path,
        'layer_indices': layer_indices,
        'seed': seed,
        'recycling_steps': recycling_steps,
        'tolerance': tolerance,
        'device': device,
        'identical': comparison['identical'],
        'max_diff': comparison['max_diff'],
        'comparison': {
            str(k): {k2: {k3: float(v3) if isinstance(v3, (int, float)) else v3
                         for k3, v3 in v2.items()}
                    for k2, v2 in v.items()}
            for k, v in comparison['layers'].items()
        }
    }

    results_path = output_dir / f"baseline_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save baseline activations
    if comparison['identical']:
        activations_path = output_dir / f"baseline_activations_{timestamp}.pt"
        torch.save(act1, activations_path)
        print(f"Baseline activations saved to: {activations_path}")

    # Summary
    print("\n" + "=" * 70)
    print("BASELINE TEST SUMMARY")
    print("=" * 70)

    if comparison['identical']:
        print("\n  SUCCESS: Boltz2 produces deterministic outputs!")
        print("  The saved activations can be used as ground truth for PLT training.")
    else:
        print("\n  WARNING: Non-determinism detected!")
        print("  Check if MSA subsampling is disabled and seeds are properly set.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run deterministic baseline test for Boltz2'
    )
    parser.add_argument('--fasta', type=str, required=True,
                       help='Input FASTA file')
    parser.add_argument('--checkpoint', type=str,
                       default='/usr/scratch/rmanimaran8/boltz/.boltz_cache/boltz2_conf.ckpt',
                       help='Path to Boltz2 checkpoint')
    parser.add_argument('--output', type=str, default='baseline_output',
                       help='Output directory')
    parser.add_argument('--layers', type=int, nargs='+',
                       default=[0, 8, 16, 24, 32, 40],
                       help='Layer indices to capture')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--recycling-steps', type=int, default=0,
                       help='Number of recycling steps')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Tolerance for identical comparison')

    args = parser.parse_args()

    results = run_baseline_test(
        checkpoint_path=args.checkpoint,
        fasta_path=args.fasta,
        output_dir=args.output,
        layer_indices=args.layers,
        device=args.device,
        seed=args.seed,
        recycling_steps=args.recycling_steps,
        tolerance=args.tolerance
    )

    return 0 if results['identical'] else 1


if __name__ == '__main__':
    sys.exit(main())
