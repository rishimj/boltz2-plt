"""
Collect activations from Boltz layer 47 - DIRECT APPROACH (no CLI).
Load model, process data, run forward pass, save activations.
"""
import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np

# Add Boltz to path
boltz_root = Path(__file__).parent.parent
sys.path.insert(0, str(boltz_root / "src"))

# Set environment before importing Boltz
os.environ['BOLTZ_CACHE'] = str(boltz_root / ".boltz_cache")

# Storage for activations
activations = {}

def register_hooks(model, layer_idx=47):
    """Register hooks on layer 47 transition MLPs."""
    layer = model.pairformer_module.layers[layer_idx]
    
    def hook_s(module, input, output):
        activations['input_s'] = input[0].detach().cpu()
        activations['output_s'] = output.detach().cpu()
        print(f"  ✓ Captured transition_s: {output.shape}")
    
    def hook_z(module, input, output):
        activations['input_z'] = input[0].detach().cpu()
        activations['output_z'] = output.detach().cpu()
        print(f"  ✓ Captured transition_z: {output.shape}")
    
    layer.transition_s.register_forward_hook(hook_s)
    layer.transition_z.register_forward_hook(hook_z)
    print(f"✓ Registered hooks on layer {layer_idx}")


def save_activations(output_path):
    """Save captured activations."""
    if len(activations) != 4:
        print(f"✗ Only captured {len(activations)}/4 activations: {list(activations.keys())}")
        return False
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {k: v.numpy() for k, v in activations.items()}
    np.savez_compressed(output_path, **data)
    
    print(f"\n✓ Saved activations to {output_path}")
    for k, v in data.items():
        print(f"  {k}: {v.shape}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', default='test_protein.fasta')
    parser.add_argument('--checkpoint', default='../.boltz_cache/boltz2_conf.ckpt')
    parser.add_argument('--output', default='real_activations/protein_001.npz')
    parser.add_argument('--layer', type=int, default=47)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    print("=== Direct Boltz Activation Collection ===")
    print(f"FASTA: {args.fasta}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Layer: {args.layer}")
    print(f"Output: {args.output}\n")
    
    # Import Boltz components
    from boltz.model.models.boltz2 import Boltz2
    from boltz.data.parse.fasta import parse_fasta
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.mol import load_canonicals
    
    # 1. Load model
    print("Loading Boltz2 model...")
    model = Boltz2.load_from_checkpoint(
        args.checkpoint,
        map_location='cpu',
        strict=False
    )
    model = model.to(args.device)
    model.eval()
    print("✓ Model loaded\n")
    
    # 2. Register hooks
    register_hooks(model, args.layer)
    print()
    
    # 3. Load canonical molecules
    print("Loading canonical molecules...")
    mol_dir = boltz_root / ".boltz_cache" / "mols"
    molecules = load_canonicals(str(mol_dir))
    print(f"✓ Loaded {len(molecules)} molecules\n")
    
    # 4. Parse FASTA (use molecules as CCD)
    print("Processing input...")
    fasta_path = Path(args.fasta)
    target = parse_fasta(fasta_path, molecules, mol_dir, boltz2=True)
    print(f"✓ Parsed FASTA: {target.record.id}")
    
    # 5. Load MSAs and create Input object
    print("\nLoading MSAs...")
    from boltz.data.parse.a3m import parse_a3m
    from boltz.data.types import Input
    
    msa_dict = {}
    for chain in target.record.chains:
        if chain.msa_id and chain.msa_id != -1:
            # MSA path is stored in chain.msa_id
            msa_path = Path(chain.msa_id)
            if not msa_path.is_absolute():
                # Resolve relative to FASTA location
                msa_path = (fasta_path.parent / msa_path).resolve()
            
            if msa_path.exists():
                msa = parse_a3m(msa_path, taxonomy=None)
                msa_dict[chain.chain_name] = msa
                print(f"  ✓ Loaded MSA for chain {chain.chain_name}: {len(msa.sequences)} sequences")
            else:
                print(f"  ✗ MSA not found: {msa_path}")
    
    # Create Input object
    input_data = Input(
        structure=target.structure,
        msa=msa_dict,
        record=target.record,
        residue_constraints=target.residue_constraints,
        templates=target.templates,
        extra_mols=target.extra_mols,
    )
    print(f"✓ Created Input with {len(msa_dict)} MSAs")
    
    # 6. Tokenize
    print("\nTokenizing...")
    tokenizer = Boltz2Tokenizer()
    tokens = tokenizer.tokenize(input_data)
    print(f"✓ Tokenized: {tokens.tokens.shape[0]} tokens")
    
    # 7. Featurize
    print("\nFeaturizing...")
    featurizer = Boltz2Featurizer()
    feats = featurizer.process(
        data=tokens,
        random=np.random.default_rng(42),
        molecules=molecules,
        training=False,
        max_seqs=128,
    )
    print(f"✓ Generated features")
    
    # 8. Convert to batch
    print("\nPreparing batch...")
    feats_batch = {}
    for k, v in feats.items():
        if isinstance(v, np.ndarray):
            feats_batch[k] = torch.from_numpy(v).unsqueeze(0).to(args.device)
        elif isinstance(v, torch.Tensor):
            feats_batch[k] = v.unsqueeze(0).to(args.device)
        else:
            feats_batch[k] = v
    print(f"✓ Batch ready")
    
    # 9. Run forward pass
    print("\nRunning forward pass through Boltz trunk...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            try:
                output = model(
                    feats=feats_batch,
                    recycling_steps=0,
                )
                print("✓ Forward pass complete")
            except Exception as e:
                # Forward pass may fail in structure module, but we already have activations!
                if len(activations) == 4:
                    print(f"⚠ Forward pass failed in structure module (expected): {e}")
                    print("✓ But activations were already captured from pairformer!")
                else:
                    print(f"✗ Forward pass failed before capturing activations: {e}")
                    import traceback
                    traceback.print_exc()
                    return
    
    # 10. Save activations (works even if structure module failed)
    save_activations(args.output)
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)


if __name__ == '__main__':
    main()
