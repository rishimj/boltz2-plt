"""
Collect real activations from Boltz layer 47 (pairformer layer 48).
Simple approach: Load model, run inference with hooks, save activations.
"""
import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from datetime import datetime

# Add boltz to path
boltz_root = Path(__file__).parent.parent
sys.path.insert(0, str(boltz_root / "src"))


class ActivationCollector:
    """Captures activations from layer 47 transition MLPs."""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self, model, layer_idx=47):
        """Register forward hooks on transition_s and transition_z."""
        target_layer = model.pairformer_module.layers[layer_idx]
        
        def hook_s(module, input, output):
            self.activations['input_s'] = input[0].detach().cpu()
            self.activations['output_s'] = output.detach().cpu()
        
        def hook_z(module, input, output):
            self.activations['input_z'] = input[0].detach().cpu()
            self.activations['output_z'] = output.detach().cpu()
        
        self.hooks.append(target_layer.transition_s.register_forward_hook(hook_s))
        self.hooks.append(target_layer.transition_z.register_forward_hook(hook_z))
        
        print(f"✓ Registered hooks on layer {layer_idx}")
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def save(self, output_path):
        """Save captured activations."""
        if len(self.activations) != 4:
            print(f"✗ Only captured {len(self.activations)}/4 tensors: {list(self.activations.keys())}")
            return False
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy and save
        data = {}
        for key, tensor in self.activations.items():
            arr = tensor.numpy()
            data[key] = arr
            print(f"  {key}: {arr.shape}")
        
        np.savez_compressed(output_path, **data)
        print(f"✓ Saved to {output_path}")
        return True


def run_inference_and_collect(
    fasta_path,
    checkpoint_path,
    output_path,
    layer_idx=47,
    device='cuda'
):
    """
    Run Boltz inference on a FASTA file and collect layer 47 activations.
    """
    print(f"=== Collecting Layer {layer_idx} Activations ===")
    print(f"FASTA: {fasta_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}\n")
    
    # Load FASTA
    from boltz.data.parse.fasta import parse_fasta
    sequences = parse_fasta(fasta_path)
    print(f"Loaded {len(sequences)} sequence(s)")
    
    # Create Input
    from boltz.data.types import Input, Record
    records = []
    msa_dict = {}
    
    for seq_id, seq in sequences.items():
        record = Record(
            id=seq_id,
            mol_type='protein',
            entity_type='ligand',
            source='input',
        )
        records.append(record)
    
    input_data = Input(
        records=records,
        sequences=sequences,
        msa=msa_dict,
    )
    
    # Tokenize
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    tokenizer = Boltz2Tokenizer()
    tokens = tokenizer.tokenize(input_data)
    print(f"✓ Tokenized input")
    
    # Load canonical molecules
    from boltz.data.mol import load_canonicals
    mol_dir = boltz_root / ".boltz_cache" / "mols"
    if not mol_dir.exists():
        print(f"✗ Molecules directory not found: {mol_dir}")
        print("  Run: boltz download --cache {boltz_root}/.boltz_cache")
        return False
    
    molecules = load_canonicals(str(mol_dir))
    print(f"✓ Loaded {len(molecules)} canonical molecules")
    
    # Featurize
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    featurizer = Boltz2Featurizer()
    
    feats = featurizer.process(
        data=tokens,
        random=np.random.default_rng(42),
        molecules=molecules,
        training=False,
        max_seqs=128,
    )
    print(f"✓ Featurized input")
    
    # Convert to batch
    feats_batch = {}
    for k, v in feats.items():
        if isinstance(v, np.ndarray):
            feats_batch[k] = torch.from_numpy(v).unsqueeze(0)
        elif isinstance(v, torch.Tensor):
            feats_batch[k] = v.unsqueeze(0)
        else:
            feats_batch[k] = v
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    print("(This may take 1-3 minutes for weight initialization...)")
    
    from boltz.model.models.boltz2 import Boltz2
    
    try:
        model = Boltz2.load_from_checkpoint(
            checkpoint_path,
            map_location='cpu',
            strict=False
        )
        print("✓ Loaded model (load_from_checkpoint)")
    except Exception as e:
        print(f"load_from_checkpoint failed: {e}")
        print("Trying manual loading...")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        import inspect
        valid_params = set(inspect.signature(Boltz2.__init__).parameters.keys()) - {'self'}
        hparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in valid_params}
        
        model = Boltz2(**hparams)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("✓ Loaded model (manual)")
    
    model = model.to(device)
    model.eval()
    
    # Register hooks
    collector = ActivationCollector()
    collector.register_hooks(model, layer_idx=layer_idx)
    
    # Move features to device
    for k, v in feats_batch.items():
        if isinstance(v, torch.Tensor):
            feats_batch[k] = v.to(device)
    
    # Run forward pass (trunk only, no structure module)
    print("\nRunning forward pass...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            try:
                # Run trunk forward pass
                output = model(
                    feats=feats_batch,
                    recycling_steps=0,
                    run_trunk_and_structure=True,
                )
                print("✓ Forward pass complete")
            except Exception as e:
                print(f"✗ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    # Save activations
    print("\nSaving activations:")
    success = collector.save(output_path)
    
    # Cleanup
    collector.remove_hooks()
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Collect Boltz layer 47 activations')
    parser.add_argument('--fasta', default='../examples/prot.fasta', help='Input FASTA file')
    parser.add_argument('--checkpoint', default='../.boltz_cache/boltz2_conf.ckpt', help='Boltz2 checkpoint')
    parser.add_argument('--output', default='real_activations/protein_001.npz', help='Output .npz file')
    parser.add_argument('--layer', type=int, default=47, help='Layer index (0-indexed)')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set cache to avoid permission issues
    os.environ['BOLTZ_CACHE'] = str(boltz_root / ".boltz_cache")
    
    success = run_inference_and_collect(
        fasta_path=args.fasta,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        layer_idx=args.layer,
        device=args.device,
    )
    
    if success:
        print(f"\n{'='*60}")
        print("SUCCESS! Activations collected.")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("FAILED to collect activations.")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == '__main__':
    main()
