"""
Collect activations from multiple proteins in batch.
"""
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add Boltz to path
boltz_root = Path(__file__).parent.parent
sys.path.insert(0, str(boltz_root / "src"))

# Set environment
os.environ['BOLTZ_CACHE'] = str(boltz_root / ".boltz_cache")

# Import Boltz components
from boltz.model.models.boltz2 import Boltz2
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.a3m import parse_a3m
from boltz.data.types import Input
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_canonicals


# Storage for activations
activations = {}


def register_hooks(model, layer_idx=47):
    """Register hooks on layer 47."""
    layer = model.pairformer_module.layers[layer_idx]
    
    def hook_s(module, input, output):
        activations['input_s'] = input[0].detach().cpu()
        activations['output_s'] = output.detach().cpu()
    
    def hook_z(module, input, output):
        activations['input_z'] = input[0].detach().cpu()
        activations['output_z'] = output.detach().cpu()
    
    layer.transition_s.register_forward_hook(hook_s)
    layer.transition_z.register_forward_hook(hook_z)


def collect_activation(fasta_path, model, tokenizer, featurizer, molecules, device='cuda'):
    """Collect activation from a single FASTA file."""
    global activations
    activations = {}  # Reset
    
    print(f"\n{'='*60}")
    print(f"Processing: {fasta_path}")
    print(f"{'='*60}")
    
    try:
        # Parse FASTA
        fasta_path = Path(fasta_path)
        target = parse_fasta(fasta_path, molecules, boltz_root / ".boltz_cache" / "mols", boltz2=True)
        print(f"✓ Parsed: {target.record.id}")
        
        # Load MSAs
        msa_dict = {}
        for chain in target.record.chains:
            if chain.msa_id and chain.msa_id != -1:
                msa_path = Path(chain.msa_id)
                if not msa_path.is_absolute():
                    msa_path = (fasta_path.parent / msa_path).resolve()
                
                if msa_path.exists():
                    msa = parse_a3m(msa_path, taxonomy=None)
                    msa_dict[chain.chain_name] = msa
                    print(f"  ✓ MSA chain {chain.chain_name}: {len(msa.sequences)} sequences")
        
        # Create Input
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
        print(f"✓ Tokenized: {tokens.tokens.shape[0]} tokens")
        
        # Featurize
        feats = featurizer.process(
            data=tokens,
            random=np.random.default_rng(42),
            molecules=molecules,
            training=False,
            max_seqs=128,
        )
        
        # Convert to batch
        feats_batch = {}
        for k, v in feats.items():
            if isinstance(v, np.ndarray):
                feats_batch[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            elif isinstance(v, torch.Tensor):
                feats_batch[k] = v.unsqueeze(0).to(device)
            else:
                feats_batch[k] = v
        
        # Run forward pass
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                try:
                    model(feats=feats_batch, recycling_steps=0)
                except:
                    pass  # Ignore structure module errors
        
        # Check if we captured activations
        if len(activations) == 4:
            data = {k: v.numpy() for k, v in activations.items()}
            print(f"✓ Captured activations: s{data['input_s'].shape}, z{data['input_z'].shape}")
            return data
        else:
            print(f"✗ Failed to capture activations")
            return None
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*80)
    print("BATCH ACTIVATION COLLECTION")
    print("="*80)
    
    device = 'cuda:0'  # Use GPU 0
    
    # FASTA files to process
    fasta_files = [
        boltz_root / "examples/prot.fasta",
        boltz_root / "transcoder/test_protein.fasta",
    ]
    
    print(f"\nFiles to process: {len(fasta_files)}")
    for f in fasta_files:
        print(f"  - {f.name}")
    print()
    
    # Load model
    print("Loading Boltz2 model...")
    checkpoint = boltz_root / ".boltz_cache/boltz2_conf.ckpt"
    model = Boltz2.load_from_checkpoint(checkpoint, map_location='cpu', strict=False)
    model = model.to(device)
    model.eval()
    register_hooks(model, layer_idx=47)
    print("✓ Model loaded\n")
    
    # Load molecules
    mol_dir = boltz_root / ".boltz_cache/mols"
    molecules = load_canonicals(str(mol_dir))
    print(f"✓ Loaded {len(molecules)} molecules\n")
    
    # Initialize tokenizer and featurizer
    tokenizer = Boltz2Tokenizer()
    featurizer = Boltz2Featurizer()
    
    # Collect from all files
    output_dir = Path("real_activations")
    output_dir.mkdir(exist_ok=True)
    
    all_activations = []
    
    for i, fasta_file in enumerate(fasta_files, 1):
        if not fasta_file.exists():
            print(f"✗ File not found: {fasta_file}")
            continue
        
        data = collect_activation(
            fasta_file,
            model,
            tokenizer,
            featurizer,
            molecules,
            device=device
        )
        
        if data is not None:
            all_activations.append(data)
    
    # Save as single batch file
    if all_activations:
        print(f"\n{'='*60}")
        print(f"Combining {len(all_activations)} proteins into single batch")
        print(f"{'='*60}")
        
        # Combine into batch (need to pad to same size)
        # For now, save separately
        for i, data in enumerate(all_activations, 1):
            output_file = output_dir / f"protein_{i:03d}.npz"
            np.savez_compressed(output_file, **data)
            print(f"✓ Saved: {output_file.name}")
        
        print(f"\n✓ Total proteins collected: {len(all_activations)}")
    else:
        print("\n✗ No activations collected")


if __name__ == '__main__':
    main()
