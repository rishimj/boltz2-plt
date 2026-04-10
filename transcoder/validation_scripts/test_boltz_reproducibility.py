"""
Test Boltz2 Reproducibility (No Transcoder)

Simple test: Run Boltz2 twice on the same input and verify 
layer 47 activations are identical.

This is a sanity check that Boltz2 behaves deterministically before 
testing transcoder interventions.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add Boltz to path
boltz_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(boltz_root))

from boltz.model.models.boltz2 import Boltz2
from boltz.data.parse.fasta import parse_fasta
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_canonicals
from boltz.data.parse.a3m import parse_a3m
from boltz.data.types import Input


# Global storage for activations
captured_activations = {}


def register_hooks(model, layer_idx=47):
    """Register hooks to capture layer 47 outputs"""
    global captured_activations
    
    def hook_s(module, input, output):
        captured_activations['s'] = output.detach().cpu().clone()
    
    def hook_z(module, input, output):
        captured_activations['z'] = output.detach().cpu().clone()
    
    layer = model.pairformer_module.layers[layer_idx]
    layer.transition_s.register_forward_hook(hook_s)
    layer.transition_z.register_forward_hook(hook_z)


def run_boltz_prediction(checkpoint_path, fasta_path, device='cuda'):
    """
    Run Boltz2 prediction on a FASTA file.
    
    Returns:
        dict with layer 47 activations
    """
    global captured_activations
    captured_activations = {}
    
    # 1. Load model
    print(f"Loading model from {checkpoint_path}...")
    model = Boltz2.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu',
        strict=False
    )
    model = model.to(device)
    model.eval()
    
    # 2. Register hooks
    register_hooks(model, layer_idx=47)
    
    # 3. Load canonical molecules
    mol_dir = boltz_root / ".boltz_cache" / "mols"
    molecules = load_canonicals(str(mol_dir))
    
    # 4. Parse FASTA
    fasta_path = Path(fasta_path)
    target = parse_fasta(fasta_path, molecules, mol_dir, boltz2=True)
    
    # 5. Load MSAs
    msa_dict = {}
    for chain in target.record.chains:
        if chain.msa_id and chain.msa_id != -1:
            msa_path = Path(chain.msa_id)
            if not msa_path.is_absolute():
                msa_path = (fasta_path.parent / msa_path).resolve()
            
            if msa_path.exists():
                msa = parse_a3m(msa_path, taxonomy=None)
                msa_dict[chain.chain_name] = msa
    
    # Create Input object
    input_data = Input(
        structure=target.structure,
        msa=msa_dict,
        record=target.record,
        residue_constraints=target.residue_constraints,
        templates=target.templates,
        extra_mols=target.extra_mols,
    )
    
    # 6. Tokenize
    tokenizer = Boltz2Tokenizer()
    tokens = tokenizer.tokenize(input_data)
    
    # 7. Featurize
    featurizer = Boltz2Featurizer()
    feats = featurizer.process(
        data=tokens,
        random=np.random.default_rng(42),  # Fixed seed for reproducibility!
        molecules=molecules,
        training=False,
        max_seqs=128,
    )
    
    # 8. Convert to batch
    feats_batch = {}
    for k, v in feats.items():
        if isinstance(v, np.ndarray):
            feats_batch[k] = torch.from_numpy(v).unsqueeze(0).to(device)
        elif isinstance(v, torch.Tensor):
            feats_batch[k] = v.unsqueeze(0).to(device)
        else:
            feats_batch[k] = v
    
    # 9. Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            try:
                output = model(feats=feats_batch, recycling_steps=0)
            except Exception as e:
                # Structure module may fail, but activations already captured
                if len(captured_activations) == 2:
                    print(f"  (Structure module failed, but activations captured)")
                else:
                    raise
    
    return {
        's': captured_activations.get('s'),
        'z': captured_activations.get('z'),
    }


def compare_outputs(output1, output2, tolerance=1e-6):
    """
    Compare two sets of layer 47 activations.
    
    Returns:
        dict with comparison results
    """
    results = {}
    
    # Compare single representation (s)
    if output1['s'] is not None and output2['s'] is not None:
        s1 = output1['s']
        s2 = output2['s']
        
        max_diff = torch.abs(s1 - s2).max().item()
        mean_diff = torch.abs(s1 - s2).mean().item()
        
        results['s_max_diff'] = max_diff
        results['s_mean_diff'] = mean_diff
        results['s_identical'] = max_diff < tolerance
        results['s_shape'] = tuple(s1.shape)
    
    # Compare pair representation (z)
    if output1['z'] is not None and output2['z'] is not None:
        z1 = output1['z']
        z2 = output2['z']
        
        max_diff = torch.abs(z1 - z2).max().item()
        mean_diff = torch.abs(z1 - z2).mean().item()
        
        results['z_max_diff'] = max_diff
        results['z_mean_diff'] = mean_diff
        results['z_identical'] = max_diff < tolerance
        results['z_shape'] = tuple(z1.shape)
    
    return results


def main():
    """Run reproducibility test"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test Boltz2 reproducibility (no transcoder)'
    )
    parser.add_argument('--fasta', type=str, required=True,
                       help='Input FASTA file')
    parser.add_argument('--boltz-checkpoint', type=str,
                       default='/usr/scratch/rmanimaran8/boltz/.boltz_cache/boltz2_conf.ckpt',
                       help='Path to Boltz2 checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Tolerance for considering outputs identical')
    
    args = parser.parse_args()
    
    print("="*70)
    print("BOLTZ2 REPRODUCIBILITY TEST")
    print("="*70)
    print(f"Input: {args.fasta}")
    print(f"Device: {args.device}")
    print(f"Tolerance: {args.tolerance}")
    print()
    
    # Run 1
    print("="*70)
    print("RUN 1: First prediction")
    print("="*70)
    output1 = run_boltz_prediction(args.boltz_checkpoint, args.fasta, args.device)
    print(f"✓ Run 1 complete")
    if output1['s'] is not None:
        print(f"  Single rep (s) shape: {output1['s'].shape}")
    if output1['z'] is not None:
        print(f"  Pair rep (z) shape: {output1['z'].shape}")
    print()
    
    # Run 2
    print("="*70)
    print("RUN 2: Second prediction (should be identical)")
    print("="*70)
    output2 = run_boltz_prediction(args.boltz_checkpoint, args.fasta, args.device)
    print(f"✓ Run 2 complete")
    if output2['s'] is not None:
        print(f"  Single rep (s) shape: {output2['s'].shape}")
    if output2['z'] is not None:
        print(f"  Pair rep (z) shape: {output2['z'].shape}")
    print()
    
    # Compare
    print("="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    results = compare_outputs(output1, output2, args.tolerance)
    
    # Print results
    if 's_max_diff' in results:
        print(f"\nSingle representation (s): {results['s_shape']}")
        print(f"  Max difference:  {results['s_max_diff']:.2e}")
        print(f"  Mean difference: {results['s_mean_diff']:.2e}")
        
        if results['s_identical']:
            print(f"  ✅ IDENTICAL (within tolerance {args.tolerance:.0e})")
        else:
            print(f"  ❌ DIFFERENT (exceeds tolerance {args.tolerance:.0e})")
    
    if 'z_max_diff' in results:
        print(f"\nPair representation (z): {results['z_shape']}")
        print(f"  Max difference:  {results['z_max_diff']:.2e}")
        print(f"  Mean difference: {results['z_mean_diff']:.2e}")
        
        if results['z_identical']:
            print(f"  ✅ IDENTICAL (within tolerance {args.tolerance:.0e})")
        else:
            print(f"  ❌ DIFFERENT (exceeds tolerance {args.tolerance:.0e})")
    
    # Overall verdict
    print()
    print("="*70)
    all_identical = all(
        results.get(k, True) 
        for k in ['s_identical', 'z_identical']
    )
    
    if all_identical:
        print("✅ BOLTZ2 IS REPRODUCIBLE - Outputs are identical!")
        print()
        print("This means:")
        print("  - Boltz2 behaves deterministically")
        print("  - Safe to proceed with transcoder intervention tests")
        print("  - Any differences in intervention will be due to transcoder")
        return 0
    else:
        print("⚠️ BOLTZ2 HAS NON-DETERMINISTIC BEHAVIOR")
        print()
        print("Possible causes:")
        print("  - Non-deterministic CUDA operations")
        print("  - Dropout layers (should be disabled in eval mode)")
        print("  - Random sampling in model")
        print()
        print("Recommendation:")
        print("  - Try running on CPU (--device cpu)")
        print("  - Check if model is in eval() mode")
        print("  - Increase tolerance if differences are very small")
        return 1
    
    print("="*70)


if __name__ == '__main__':
    sys.exit(main())
