"""
Diagnose the source of non-determinism in Boltz2

This script:
1. Tests multiple layers to see where non-determinism starts
2. Tests with MSA subsampling disabled
3. Checks if it's CUDA operations or MSA sampling
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


def setup_determinism():
    """Set all random seeds for determinism"""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def register_multi_layer_hooks(model, layers_to_check=[0, 10, 20, 30, 40, 47, 63]):
    """Register hooks to capture outputs at multiple pairformer layers"""
    captured = {}
    hooks = []
    
    for layer_idx in layers_to_check:
        def make_hook_s(idx):
            def hook(module, input, output):
                captured[f'layer{idx}_s'] = output.detach().cpu().clone()
            return hook
        
        def make_hook_z(idx):
            def hook(module, input, output):
                captured[f'layer{idx}_z'] = output.detach().cpu().clone()
            return hook
        
        if layer_idx < len(model.pairformer_module.layers):
            layer = model.pairformer_module.layers[layer_idx]
            hooks.append(layer.transition_s.register_forward_hook(make_hook_s(layer_idx)))
            hooks.append(layer.transition_z.register_forward_hook(make_hook_z(layer_idx)))
    
    # Also capture MSA module output
    def hook_msa_z(module, input, output):
        captured['msa_output_z'] = output.detach().cpu().clone()
    
    if hasattr(model, 'msa_module'):
        hooks.append(model.msa_module.register_forward_hook(hook_msa_z))
    
    return captured, hooks


def run_test(checkpoint_path, fasta_path, device='cuda', test_name="Test", 
             disable_msa_subsample=False):
    """
    Run a single test with specified configuration
    
    Args:
        disable_msa_subsample: If True, patches the model to disable MSA subsampling
    """
    setup_determinism()
    
    print(f"\n{'='*70}")
    print(f"{test_name}")
    print(f"{'='*70}")
    
    # 1. Load model
    print(f"Loading model...")
    model = Boltz2.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu',
        strict=False
    )
    
    # Patch MSA subsampling if requested
    if disable_msa_subsample:
        print("⚠️  Disabling MSA subsampling...")
        if hasattr(model, 'msa_module'):
            model.msa_module.subsample_msa = False
    else:
        if hasattr(model, 'msa_module'):
            print(f"MSA subsampling enabled: {model.msa_module.subsample_msa}")
    
    model = model.to(device)
    model.eval()
    
    # 2. Register hooks for multiple layers
    captured, hooks = register_multi_layer_hooks(model)
    
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
    
    # 6. Tokenize
    tokenizer = Boltz2Tokenizer()
    tokens = tokenizer.tokenize(input_data)
    
    # 7. Featurize - with FIXED seed
    featurizer = Boltz2Featurizer()
    feats = featurizer.process(
        data=tokens,
        random=np.random.default_rng(42),  # Fixed seed
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
                if len(captured) > 0:
                    print(f"  (Structure module failed, but activations captured)")
                else:
                    raise
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    return captured


def compare_multi_layer(output1, output2, tolerance=1e-6):
    """Compare outputs from multiple layers"""
    
    print(f"\n{'='*70}")
    print("LAYER-BY-LAYER COMPARISON")
    print(f"{'='*70}")
    
    all_layers = sorted(set(
        [k.split('_')[0] for k in output1.keys()] +
        [k.split('_')[0] for k in output2.keys()]
    ))
    
    results = {}
    
    for layer_key in all_layers:
        s_key = f'{layer_key}_s'
        z_key = f'{layer_key}_z'
        
        # Check single representation
        if s_key in output1 and s_key in output2:
            s1 = output1[s_key]
            s2 = output2[s_key]
            
            max_diff = torch.abs(s1 - s2).max().item()
            mean_diff = torch.abs(s1 - s2).mean().item()
            identical = max_diff < tolerance
            
            results[s_key] = {
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'identical': identical,
                'shape': tuple(s1.shape)
            }
            
            status = "✅" if identical else "❌"
            print(f"\n{layer_key} single (s): {s1.shape}")
            print(f"  Max diff:  {max_diff:.2e}  {status}")
            print(f"  Mean diff: {mean_diff:.2e}")
        
        # Check pair representation
        if z_key in output1 and z_key in output2:
            z1 = output1[z_key]
            z2 = output2[z_key]
            
            max_diff = torch.abs(z1 - z2).max().item()
            mean_diff = torch.abs(z1 - z2).mean().item()
            identical = max_diff < tolerance
            
            results[z_key] = {
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'identical': identical,
                'shape': tuple(z1.shape)
            }
            
            status = "✅" if identical else "❌"
            print(f"{layer_key} pair (z): {z1.shape}")
            print(f"  Max diff:  {max_diff:.2e}  {status}")
            print(f"  Mean diff: {mean_diff:.2e}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Diagnose non-determinism in Boltz2'
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
    print("BOLTZ2 NON-DETERMINISM DIAGNOSIS")
    print("="*70)
    print(f"Input: {args.fasta}")
    print(f"Device: {args.device}")
    
    # Test 1: Default behavior (with MSA subsampling)
    output1_default = run_test(
        args.boltz_checkpoint, 
        args.fasta, 
        args.device,
        test_name="RUN 1: With MSA subsampling (default)",
        disable_msa_subsample=False
    )
    
    output2_default = run_test(
        args.boltz_checkpoint, 
        args.fasta, 
        args.device,
        test_name="RUN 2: With MSA subsampling (default)",
        disable_msa_subsample=False
    )
    
    results_default = compare_multi_layer(output1_default, output2_default, args.tolerance)
    
    # Test 2: Disable MSA subsampling
    print(f"\n\n{'='*70}")
    print("TESTING WITH MSA SUBSAMPLING DISABLED")
    print(f"{'='*70}")
    
    output1_no_subsample = run_test(
        args.boltz_checkpoint, 
        args.fasta, 
        args.device,
        test_name="RUN 3: WITHOUT MSA subsampling",
        disable_msa_subsample=True
    )
    
    output2_no_subsample = run_test(
        args.boltz_checkpoint, 
        args.fasta, 
        args.device,
        test_name="RUN 4: WITHOUT MSA subsampling",
        disable_msa_subsample=True
    )
    
    results_no_subsample = compare_multi_layer(output1_no_subsample, output2_no_subsample, args.tolerance)
    
    # Summary
    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*70}")
    
    # Check if disabling MSA subsampling fixes the issue
    default_identical = all(v['identical'] for v in results_default.values())
    no_subsample_identical = all(v['identical'] for v in results_no_subsample.values())
    
    if not default_identical and no_subsample_identical:
        print("\n✅ ROOT CAUSE IDENTIFIED: MSA Subsampling")
        print("\nThe non-determinism is caused by torch.randperm() in MSA subsampling.")
        print("This happens in the MSA module before the pairformer layers.")
        print("\nLocation: boltz/model/modules/trunkv2.py, line ~632")
        print("Code: msa_indices = torch.randperm(msa.shape[1])[: self.num_subsampled_msa]")
        print("\nThis random permutation is applied even during inference (eval mode),")
        print("causing different MSA sequences to be selected on each run.")
        print("\nSolution:")
        print("  - Disable MSA subsampling during inference")
        print("  - OR use a fixed seed for torch.randperm")
        print("  - OR sort indices to make selection deterministic")
    elif not default_identical:
        print("\n❌ MSA subsampling is not the only source of non-determinism")
        print("There may be additional sources such as:")
        print("  - Non-deterministic CUDA operations")
        print("  - Other random operations in the model")
    else:
        print("\n✅ BOLTZ2 is deterministic with current configuration")
    
    return 0 if no_subsample_identical else 1


if __name__ == '__main__':
    sys.exit(main())
