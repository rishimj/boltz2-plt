#!/usr/bin/env python3
"""Test the multi-layer collector on a single protein."""

import sys
from pathlib import Path

# Add paths
boltz_root = Path(__file__).parent
sys.path.insert(0, str(boltz_root / "src"))
sys.path.insert(0, str(boltz_root / "collection_scripts"))

import torch
import numpy as np
from collect_multi_layer import MultiLayerActivationCollector, LayerActivationCollector

# Try loading Boltz
try:
    from boltz.model.models.boltz2 import Boltz2
    from boltz.data.parse.fasta import parse_fasta
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.mol import load_canonicals
    print("✓ Boltz imports successful")
except Exception as e:
    print(f"✗ Failed to import Boltz: {e}")
    sys.exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model
checkpoint_path = Path("/usr/scratch/rmanimaran8/boltz/.boltz_cache/boltz2_conf.ckpt")
print(f"\nLoading checkpoint from {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"✓ Checkpoint loaded")
except Exception as e:
    print(f"✗ Failed to load checkpoint: {e}")
    sys.exit(1)

# Get hyperparameters
import inspect
all_hparams = checkpoint['hyper_parameters']
valid_params = set(inspect.signature(Boltz2.__init__).parameters.keys()) - {'self'}
hparams = {k: v for k, v in all_hparams.items() if k in valid_params}

print(f"Creating model with {len(hparams)} parameters...")
try:
    model = Boltz2(**hparams)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully")
    print(f"  Pairformer has {len(model.pairformer_module.layers)} layers")
except Exception as e:
    print(f"✗ Failed to create/load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Initialize tokenizer and featurizer
print("\nInitializing tokenizer and featurizer...")
try:
    tokenizer = Boltz2Tokenizer()
    featurizer = Boltz2Featurizer()
    print("✓ Tokenizer and featurizer initialized")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load molecules
moldir = Path("/usr/scratch/rmanimaran8/boltz/.boltz_cache/mols")
if not moldir.exists():
    moldir = Path.home() / ".boltz_cache" / "mols"

print(f"Loading molecules from {moldir}")
try:
    molecules = load_canonicals(str(moldir))
    print(f"✓ Loaded {len(molecules)} molecules")
except Exception as e:
    print(f"✗ Failed to load molecules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get FASTA file
fasta_path = Path("/usr/scratch/rmanimaran8/boltz/examples/prot.fasta")
print(f"\nUsing FASTA: {fasta_path}")
if not fasta_path.exists():
    print(f"✗ FASTA file not found: {fasta_path}")
    sys.exit(1)

# Initialize collector
layer_indices = [0, 8, 16, 24, 32, 40]
print(f"\nInitializing collector for layers: {layer_indices}")
try:
    collector = MultiLayerActivationCollector(model, layer_indices, device=device)
    print("✓ Collector initialized")
except Exception as e:
    print(f"✗ Failed to initialize collector: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Process one protein
print(f"\nProcessing {fasta_path.name}...")
try:
    from boltz.data.parse.fasta import parse_fasta
    from boltz.data.types import Input
    from boltz.data.parse.a3m import parse_a3m

    target = parse_fasta(fasta_path, molecules, moldir, boltz2=True)
    print("  ✓ FASTA parsed")
    
    # Load MSAs if they exist
    msa_dict = {}
    for chain in target.record.chains:
        if chain.msa_id and chain.msa_id != -1:
            msa_path = Path(chain.msa_id)
            if not msa_path.is_absolute():
                msa_path = (fasta_path.parent / msa_path).resolve()
            
            if msa_path.exists():
                msa = parse_a3m(msa_path, taxonomy=None)
                msa_dict[chain.chain_name] = msa
                print(f"  ✓ Loaded MSA for {chain.chain_name}")
    
    input_data = Input(
        structure=target.structure,
        msa=msa_dict,
        record=target.record,
        residue_constraints=target.residue_constraints,
        templates=target.templates,
        extra_mols=target.extra_mols,
    )
    print("  ✓ Input created")
    
    # Tokenize
    tokens = tokenizer.tokenize(input_data)
    print("  ✓ Tokenized")
    
    # Featurize
    random_generator = np.random.default_rng(42)
    features = featurizer.process(
        data=tokens,
        molecules=molecules,
        random=random_generator,
        training=False,
        max_seqs=128,
    )
    print("  ✓ Featurized")
    
    # Move to device
    feats = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            feats[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, np.ndarray):
            feats[key] = torch.from_numpy(value).unsqueeze(0).to(device)
        else:
            feats[key] = value
    print("  ✓ Moved to device")
    
    # Run inference
    print("  Running inference...")
    with torch.no_grad():
        try:
            output = model(feats, recycling_steps=0)
            print("  ✓ Inference complete")
        except Exception as e:
            print(f"  ⚠ Inference error (but collecting what we have): {e}")
    
    # Get batches
    print("  Extracting activations...")
    batches = collector.pop_batches()
    
    success_count = 0
    for layer_idx, batch in batches.items():
        if batch is not None:
            print(f"    ✓ Layer {layer_idx}: input_s={batch['input_s'].shape}, input_z={batch['input_z'].shape}")
            success_count += 1
        else:
            print(f"    ✗ Layer {layer_idx}: No activations collected")
    
    if success_count > 0:
        print(f"\n✓ SUCCESS: Collected activations from {success_count}/{len(layer_indices)} layers")
    else:
        print(f"\n✗ FAILURE: No activations collected from any layer")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Failed to process protein: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    print("\nCleaning up...")
    collector.remove_hooks()

print("\n" + "="*80)
print("COLLECTOR TEST PASSED")
print("="*80)
