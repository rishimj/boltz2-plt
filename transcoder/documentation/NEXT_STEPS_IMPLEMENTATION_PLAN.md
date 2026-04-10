# Transcoder Project: Next Steps Implementation Plan

**Date:** March 8, 2026  
**Goal:** Extend Universal Transcoder to multi-layer analysis and intervention experiments

---

## ⚠️ Important Clarification

**Boltz2 Architecture:**
- **Pairformer layers:** 48 total (indices 0-47)
- **Last pairformer layer:** Layer 47 (index 47)
- **No layer 64 exists** in the pairformer module

Your current setup hooks **layer 47** (the final pairformer layer before structure prediction).

**Possible interpretation of "layer 64":**
- If you meant the **structure prediction module** layers, that's a different component
- If you meant **layer 40 or another layer**, specify which

---

## Phase 1: Intervention Experiments (Reconstruction Validation)

### Goal
Verify that transcoder reconstructions are faithful by feeding them back into Boltz2 and checking if outputs match.

### Implementation Steps

#### Step 1.1: Collect Baseline Activations

**File to modify:** `collection_scripts/collect_intervention_baseline.py` (NEW)

```python
"""
Collect activations AND final structure predictions for baseline comparison.
"""
import torch
import numpy as np
from pathlib import Path
from boltz.model.models.boltz2 import Boltz2
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.a3m import parse_a3m
from boltz.data.types import Input
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer

# Storage
activations = {}
final_outputs = {}

def register_hooks(model, layer_idx=47):
    """Hook layer 47 AND capture final model output"""
    layer = model.pairformer_module.layers[layer_idx]
    
    def hook_s(module, input, output):
        activations['layer47_output_s'] = output.detach().cpu().clone()
    
    def hook_z(module, input, output):
        activations['layer47_output_z'] = output.detach().cpu().clone()
    
    layer.transition_s.register_forward_hook(hook_s)
    layer.transition_z.register_forward_hook(hook_z)

def collect_baseline(fasta_path, output_path):
    """
    Run Boltz2 normally, capture:
    1. Layer 47 activations (s, z)
    2. Final structure prediction
    3. All intermediate outputs for comparison
    """
    # Load model
    model = Boltz2.load_from_checkpoint(
        '/usr/scratch/rmanimaran8/boltz/.boltz_cache/boltz2_conf.ckpt',
        map_location='cpu'
    ).to('cuda')
    model.eval()
    
    # Register hooks
    register_hooks(model, layer_idx=47)
    
    # Load data (same as collect_batch.py)
    # ... tokenize, featurize ...
    
    # Run forward pass
    with torch.no_grad():
        output = model(features)
    
    # Save everything
    np.savez_compressed(output_path,
        # Activations
        layer47_s=activations['layer47_output_s'].numpy(),
        layer47_z=activations['layer47_output_z'].numpy(),
        
        # Final outputs (for comparison)
        final_coordinates=output['coordinates'].cpu().numpy(),
        final_confidence=output.get('plddt', None),
        # ... other outputs
    )
    
    return output

if __name__ == '__main__':
    baseline = collect_baseline(
        '../examples/prot.fasta',
        'intervention_data/baseline_protein1.npz'
    )
```

#### Step 1.2: Intervention Reconstruction

**File to create:** `collection_scripts/run_intervention.py` (NEW)

```python
"""
Intervention experiment: Replace layer 47 activations with transcoder reconstructions.
"""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '../universal_transcoder')
from universal_model import UniversalTranscoder

# Load baseline data
baseline = np.load('intervention_data/baseline_protein1.npz')
layer47_s = torch.from_numpy(baseline['layer47_s']).cuda()
layer47_z = torch.from_numpy(baseline['layer47_z']).cuda()

# Load trained transcoder
checkpoint = torch.load('../universal_transcoder/checkpoints/universal_transcoder_final.pt')
transcoder = UniversalTranscoder(
    d_model=384,
    d_hidden=2048,
    d_pair=128,
    k=16
).cuda()
transcoder.load_state_dict(checkpoint['model_state_dict'])
transcoder.eval()

# Reconstruct pair representations from single representation
with torch.no_grad():
    y1_recon, y2_recon, _, _, _ = transcoder(layer47_s)

# Now we need to INJECT these back into Boltz2
# This requires modifying Boltz2's forward pass...
```

**Challenge:** Boltz2 doesn't easily allow mid-forward injection.

**Solution:** Use activation hooks to **replace** activations during forward pass:

```python
def create_replacement_hook(replacement_tensor):
    """Create hook that replaces output with our reconstruction"""
    def hook(module, input, output):
        return replacement_tensor
    return hook

# Register replacement hooks
model.pairformer_module.layers[47].transition_z.register_forward_hook(
    create_replacement_hook(y1_recon)  # Replace with reconstruction
)

# Now run forward pass - it will use our reconstructed activations
modified_output = model(features)

# Compare with baseline
print(f"Coordinate RMSD: {rmsd(modified_output.coords, baseline.coords)}")
print(f"pLDDT difference: {(modified_output.plddt - baseline.plddt).abs().mean()}")
```

#### Step 1.3: Quantify Reconstruction Fidelity

**Metrics to compute:**

```python
def evaluate_intervention(baseline_output, intervention_output):
    """
    Compare baseline vs intervention-modified predictions.
    
    If transcoder is perfect (R²=1.0), outputs should be identical.
    In practice, outputs will differ proportionally to transcoder error.
    """
    metrics = {}
    
    # 1. Coordinate RMSD
    coords_baseline = baseline_output['coordinates']
    coords_intervention = intervention_output['coordinates']
    rmsd = torch.sqrt(((coords_baseline - coords_intervention)**2).sum(-1).mean())
    metrics['coordinate_rmsd'] = rmsd.item()
    
    # 2. pLDDT correlation
    plddt_baseline = baseline_output['plddt']
    plddt_intervention = intervention_output['plddt']
    correlation = torch.corrcoef(torch.stack([plddt_baseline, plddt_intervention]))[0, 1]
    metrics['plddt_correlation'] = correlation.item()
    
    # 3. TM-score (structural similarity)
    # Requires external library like TMalign
    
    # 4. Per-residue deviation
    per_res_rmsd = torch.sqrt(((coords_baseline - coords_intervention)**2).sum(-1))
    metrics['per_residue_rmsd'] = per_res_rmsd.cpu().numpy()
    
    return metrics
```

**Expected results:**
- If R² = 0.58: Expect ~40% information loss → moderate structural differences
- If R² = 0.95: Expect ~5% information loss → minor structural differences
- If R² = 1.0: Expect identical structures

---

## Phase 2: Multi-Layer PLT (Every 8th Layer)

### Goal
Train transcoders for layers [0, 8, 16, 24, 32, 40, 47] to track feature evolution.

### Implementation Steps

#### Step 2.1: Multi-Layer Activation Collection

**File to modify:** `collection_scripts/collect_multilayer.py` (NEW)

```python
"""
Collect activations from multiple pairformer layers.
"""

activations = {layer_idx: {} for layer_idx in [0, 8, 16, 24, 32, 40, 47]}

def register_multilayer_hooks(model, layer_indices=[0, 8, 16, 24, 32, 40, 47]):
    """Register hooks on multiple layers"""
    for layer_idx in layer_indices:
        layer = model.pairformer_module.layers[layer_idx]
        
        # Need to use closure to capture layer_idx
        def make_hook_s(idx):
            def hook(module, input, output):
                activations[idx]['input_s'] = input[0].detach().cpu()
                activations[idx]['output_s'] = output.detach().cpu()
                print(f"  ✓ Layer {idx} transition_s captured")
            return hook
        
        def make_hook_z(idx):
            def hook(module, input, output):
                activations[idx]['input_z'] = input[0].detach().cpu()
                activations[idx]['output_z'] = output.detach().cpu()
                print(f"  ✓ Layer {idx} transition_z captured")
            return hook
        
        layer.transition_s.register_forward_hook(make_hook_s(layer_idx))
        layer.transition_z.register_forward_hook(make_hook_z(layer_idx))

def save_multilayer_activations(protein_id, output_dir='multilayer_activations'):
    """
    Save activations in format:
    multilayer_activations/
        protein_001/
            layer_00.npz
            layer_08.npz
            layer_16.npz
            ...
    """
    output_dir = Path(output_dir)
    protein_dir = output_dir / f'protein_{protein_id:03d}'
    protein_dir.mkdir(parents=True, exist_ok=True)
    
    for layer_idx, acts in activations.items():
        if len(acts) == 4:  # All activations captured
            np.savez_compressed(
                protein_dir / f'layer_{layer_idx:02d}.npz',
                input_s=acts['input_s'].numpy(),
                output_s=acts['output_s'].numpy(),
                input_z=acts['input_z'].numpy(),
                output_z=acts['output_z'].numpy()
            )
            print(f"  ✓ Saved layer {layer_idx} → {protein_dir / f'layer_{layer_idx:02d}.npz'}")
```

**Run collection:**
```bash
cd /usr/scratch/rmanimaran8/boltz/transcoder
source ../boltz_env/bin/activate

CUDA_VISIBLE_DEVICES=0 python collection_scripts/collect_multilayer.py \
    --fasta ../examples/prot.fasta \
    --layers 0 8 16 24 32 40 47 \
    --output multilayer_activations/
```

#### Step 2.2: Train Layer-Specific Transcoders

**File to create:** `universal_transcoder/train_multilayer.py` (NEW)

```python
"""
Train separate transcoder for each layer.
"""
import torch
from universal_model import UniversalTranscoder
from pathlib import Path

layer_indices = [0, 8, 16, 24, 32, 40, 47]

# Train one transcoder per layer
transcoders = {}
for layer_idx in layer_indices:
    print(f"\n{'='*60}")
    print(f"Training transcoder for layer {layer_idx}")
    print(f"{'='*60}")
    
    # Initialize model
    model = UniversalTranscoder(
        d_model=384,
        d_hidden=2048,
        d_pair=128,
        k=16
    ).cuda()
    
    # Load data for this layer
    dataloader = create_dataloader(
        data_dir=f'../multilayer_activations',
        layer_idx=layer_idx,
        batch_size=1
    )
    
    # Train (same as train_universal.py)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(500):
        # ... training loop ...
        pass
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'layer_idx': layer_idx,
        'step': 500
    }, f'checkpoints/transcoder_layer{layer_idx:02d}.pt')
    
    transcoders[layer_idx] = model

print("\n✓ Trained all layer-specific transcoders")
```

#### Step 2.3: Analyze Feature Evolution

**File to create:** `training_scripts/analyze_multilayer_features.py` (NEW)

```python
"""
Compare features learned across different layers.
"""
import torch
import numpy as np
from pathlib import Path

# Load all transcoders
transcoders = {}
for layer_idx in [0, 8, 16, 24, 32, 40, 47]:
    checkpoint = torch.load(f'../universal_transcoder/checkpoints/transcoder_layer{layer_idx:02d}.pt')
    model = UniversalTranscoder(d_model=384, d_hidden=2048, d_pair=128, k=16).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    transcoders[layer_idx] = model

# Analyze same protein across all layers
protein_data = np.load('multilayer_activations/protein_001/layer_00.npz')

# For each layer, get sparse representations
sparse_features = {}
for layer_idx, transcoder in transcoders.items():
    layer_data = np.load(f'multilayer_activations/protein_001/layer_{layer_idx:02d}.npz')
    s = torch.from_numpy(layer_data['output_s']).cuda()
    
    with torch.no_grad():
        # Get sparse latent representation
        x_norm, mu, std = transcoder.LN(s.reshape(-1, 384))
        x_centered = x_norm - transcoder.b_pre
        pre_acts = transcoder.encoder(x_centered) + transcoder.b_enc
        sparse = transcoder.topK_activation(pre_acts, k=16)
    
    sparse_features[layer_idx] = sparse.cpu().numpy()

# Analysis 1: Which neurons are active at each layer?
print("\nActive neurons by layer:")
for layer_idx in [0, 8, 16, 24, 32, 40, 47]:
    active = (sparse_features[layer_idx] != 0).any(axis=0)
    num_active = active.sum()
    print(f"  Layer {layer_idx:2d}: {num_active:4d} / 2048 neurons ever active ({100*num_active/2048:.1f}%)")

# Analysis 2: Feature overlap across layers
def feature_overlap(features1, features2):
    """Compute Jaccard similarity of active neuron sets"""
    active1 = set(np.where((features1 != 0).any(axis=0))[0])
    active2 = set(np.where((features2 != 0).any(axis=0))[0])
    
    intersection = len(active1 & active2)
    union = len(active1 | active2)
    
    return intersection / union if union > 0 else 0

print("\nFeature overlap (Jaccard similarity):")
print("      ", end="")
for l2 in [0, 8, 16, 24, 32, 40, 47]:
    print(f"L{l2:2d}  ", end="")
print()

for l1 in [0, 8, 16, 24, 32, 40, 47]:
    print(f"L{l1:2d}:  ", end="")
    for l2 in [0, 8, 16, 24, 32, 40, 47]:
        overlap = feature_overlap(sparse_features[l1], sparse_features[l2])
        print(f"{overlap:.2f} ", end="")
    print()

# Analysis 3: Feature specialization over layers
# Early layers (0, 8) → local features
# Mid layers (16, 24, 32) → motif features
# Late layers (40, 47) → global structural features
```

---

## Phase 3: Get Final Pairformer Output

### Goal
Capture the absolute final output of the pairformer module (after layer 47).

### Current Status
You already capture layer 47 output. The pairformer module outputs are:
- `s`: Single representation `[B, N, 384]`
- `z`: Pair representation `[B, N, N, 128]`

These go directly to the structure prediction module.

### If You Want Structure Module Input

**Clarification needed:** The structure prediction module takes the pairformer outputs and processes them through:
1. Score model (diffusion conditioning)
2. Atom decoder
3. Structure prediction

**To capture structure module inputs:**

```python
def register_structure_module_hook(model):
    """Capture inputs to structure prediction module"""
    
    # Hook the structure module's forward method
    original_forward = model.structure_module.forward
    
    def hooked_forward(*args, **kwargs):
        # Save inputs
        structure_inputs['args'] = args
        structure_inputs['kwargs'] = kwargs
        
        # Call original
        return original_forward(*args, **kwargs)
    
    model.structure_module.forward = hooked_forward
```

---

## Implementation Timeline

### Week 1: Intervention Experiments
- [ ] Day 1-2: Implement baseline collection with final outputs
- [ ] Day 3-4: Implement reconstruction intervention
- [ ] Day 5: Run experiments, compute metrics
- [ ] Day 6-7: Analyze results, write up findings

### Week 2: Multi-Layer Data Collection
- [ ] Day 1-2: Implement multilayer collection script
- [ ] Day 3-5: Collect activations from 50-100 proteins across 7 layers
- [ ] Day 6-7: Organize data, verify correctness

### Week 3: Multi-Layer Training
- [ ] Day 1-3: Train 7 layer-specific transcoders (parallel if possible)
- [ ] Day 4-5: Validate all models, check reconstruction quality
- [ ] Day 6-7: Compare hyperparameters, dead neurons across layers

### Week 4: Analysis & Interpretation
- [ ] Day 1-3: Feature evolution analysis
- [ ] Day 4-5: Feature overlap, specialization patterns
- [ ] Day 6-7: Visualizations, write comprehensive report

---

## Quick Start Commands

### Phase 1: Intervention
```bash
cd /usr/scratch/rmanimaran8/boltz/transcoder
source ../boltz_env/bin/activate

# 1. Collect baseline
python collection_scripts/collect_intervention_baseline.py \
    --fasta ../examples/prot.fasta \
    --output intervention_data/baseline.npz

# 2. Run intervention
python collection_scripts/run_intervention.py \
    --baseline intervention_data/baseline.npz \
    --transcoder universal_transcoder/checkpoints/universal_transcoder_final.pt \
    --output intervention_data/intervention_result.npz

# 3. Analyze
python training_scripts/analyze_intervention.py \
    --baseline intervention_data/baseline.npz \
    --intervention intervention_data/intervention_result.npz
```

### Phase 2: Multi-Layer
```bash
# 1. Collect multi-layer activations
python collection_scripts/collect_multilayer.py \
    --fasta_dir ../examples/ \
    --layers 0 8 16 24 32 40 47 \
    --output multilayer_activations/

# 2. Train all transcoders
cd universal_transcoder
python train_multilayer.py \
    --data_dir ../multilayer_activations/ \
    --layers 0 8 16 24 32 40 47 \
    --num_steps 500

# 3. Analyze features
cd ../training_scripts
python analyze_multilayer_features.py \
    --checkpoints ../universal_transcoder/checkpoints/ \
    --data ../multilayer_activations/
```

---

## Expected Outcomes

### Intervention Experiments

**Success criteria:**
- Coordinate RMSD < 2Å: Transcoder preserves structural information
- pLDDT correlation > 0.9: Confidence predictions remain accurate
- Per-residue deviations correlate with transcoder R² scores

**Interpretation:**
- Low RMSD → transcoder is faithful, can be used for analysis
- High RMSD → transcoder loses critical information, need to improve

### Multi-Layer Analysis

**Expected patterns:**

**Layer 0 (early):**
- Universal features (100% activation): Basic amino acid properties
- Local features: Adjacent residue interactions
- Low feature specialization

**Layer 24 (middle):**
- Emerging selectivity: ~50% activation patterns
- Motif detection: Beta-turns, loops
- Increased feature specialization

**Layer 47 (late):**
- High selectivity: 20-60% activation (as you observed)
- Global features: Tertiary structure, domain architecture
- Maximum feature specialization

**Feature overlap:**
```
Expected Jaccard similarity matrix:
       L0   L8   L16  L24  L32  L40  L47
L0:   1.00 0.75 0.50 0.30 0.20 0.10 0.05
L8:   0.75 1.00 0.70 0.45 0.30 0.15 0.10
L16:  0.50 0.70 1.00 0.75 0.50 0.30 0.15
...
```
Decreasing overlap as layers get further apart → hierarchical feature learning.

---

## File Organization

```
transcoder/
├── collection_scripts/
│   ├── collect_multilayer.py          # NEW: Multi-layer collection
│   ├── collect_intervention_baseline.py  # NEW: Baseline for interventions
│   └── run_intervention.py            # NEW: Intervention experiments
│
├── multilayer_activations/            # NEW: Multi-layer data
│   ├── protein_001/
│   │   ├── layer_00.npz
│   │   ├── layer_08.npz
│   │   ├── ...
│   │   └── layer_47.npz
│   ├── protein_002/
│   └── ...
│
├── intervention_data/                 # NEW: Intervention results
│   ├── baseline_*.npz
│   └── intervention_*.npz
│
├── universal_transcoder/
│   ├── train_multilayer.py           # NEW: Train multiple transcoders
│   └── checkpoints/
│       ├── transcoder_layer00.pt     # NEW
│       ├── transcoder_layer08.pt     # NEW
│       ├── ...
│       └── transcoder_layer47.pt     # Already exists
│
├── training_scripts/
│   ├── analyze_intervention.py       # NEW: Intervention analysis
│   └── analyze_multilayer_features.py  # NEW: Multi-layer feature analysis
│
└── documentation/
    └── MULTILAYER_PLT_GUIDE.md       # NEW: Document your findings
```

---

## Key Considerations

### Computational Resources

**Multi-layer collection:**
- 7 layers × 100 proteins × ~10 MB/protein = ~7 GB
- Collection time: ~10 min/protein × 100 = ~17 hours

**Multi-layer training:**
- 7 transcoders × 500 steps × 30 sec = ~3 hours
- Can parallelize on multiple GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3`

### Memory Requirements

**Collection:** 4-8 GB VRAM (same as current)

**Training:** 
- Single transcoder: <1 GB VRAM
- 7 transcoders in parallel: 7 GB VRAM total

### Validation

**Before proceeding:**
1. Verify layer 47 intervention works correctly
2. Test multilayer collection on 1-2 proteins
3. Train 2-3 layer transcoders as proof-of-concept
4. Check that results make sense before scaling up

---

## Questions to Answer Before Starting

1. **Intervention target:** What metric proves transcoder fidelity? (RMSD? TM-score? pLDDT correlation?)

2. **Layer selection:** Confirm layers [0, 8, 16, 24, 32, 40, 47] or different spacing?

3. **Data scale:** How many proteins for multi-layer? (10, 50, 100, 500?)

4. **Shared vs separate:** Share encoder across layers or fully independent transcoders?

5. **Success criteria:** What R² or sparsity metrics indicate success for each layer?

---

## Next Immediate Action

**Recommended starting point:**

```bash
# Test intervention on 1 protein first
cd /usr/scratch/rmanimaran8/boltz/transcoder

# Create intervention_data directory
mkdir -p intervention_data

# Start with simplest intervention test:
# 1. Run Boltz normally, capture outputs
# 2. Run Boltz with transcoder reconstruction injected
# 3. Compare outputs
```

Want me to create the implementation files for any of these phases?
