# PLT Integration Scripts

Scripts for training, verifying, and inserting Per-Layer Transcoders (PLTs) into Boltz2.

## Overview

This directory contains three main scripts that implement the PLT integration plan:

1. **Phase 1**: `deterministic_baseline.py` - Establish deterministic baseline
2. **Phase 2**: Modified `train_online_multi_layer.py` - Train PLT with determinism
3. **Phase 3**: `plt_insertion.py` + `verify_plt_structure.py` - Insert and verify PLT

## Scripts

### deterministic_baseline.py

Establishes a deterministic baseline for Boltz2 by:
- Setting all random seeds (Python, NumPy, PyTorch, CUDA)
- Disabling MSA subsampling (main source of non-determinism)
- Running two forward passes and verifying identical activations

**Usage:**
```bash
python scripts/deterministic_baseline.py \
    --fasta test.fasta \
    --output baseline_results/ \
    --layers 0 8 16 24 32 40
```

**Success criteria:**
- Activation max_diff < 1e-6
- Structure RMSD < 1e-5 Å

### plt_insertion.py

Module for inserting trained PLTs into Boltz2 forward pass.

**Key features:**
- Three modes: `capture`, `replace`, `compare`
- Outer sum reconstruction: `z[i,j] = y[i] + y[j]`
- Per-layer z comparison metrics (NMSE, R²)

**Usage (as module):**
```python
from plt_insertion import PLTInsertion

plt_inserter = PLTInsertion(
    plt_checkpoints_dir='checkpoints/',
    layer_indices=[0, 8, 16]
)

plt_inserter.register_hooks(model, mode='replace')
output = model(feats)
results = plt_inserter.get_comparison_results()
plt_inserter.remove_hooks()
```

### verify_plt_structure.py

End-to-end verification script that:
1. Runs baseline Boltz2 prediction
2. Runs PLT-inserted prediction
3. Compares structure outputs (RMSD, pLDDT)
4. Reports per-layer z reconstruction quality

**Usage:**
```bash
python scripts/verify_plt_structure.py \
    --fasta test.fasta \
    --plt-checkpoints checkpoints/ \
    --layers 0 \
    --output verification_results/
```

**Success criteria:**
- Structure RMSD < 2.0 Å
- pLDDT correlation > 0.95
- z reconstruction NMSE < 0.1
- z reconstruction R² > 0.9

## Training (Modified Script)

The `universal_transcoder/train_online_multi_layer.py` script has been modified to support deterministic training:

```bash
python universal_transcoder/train_online_multi_layer.py \
    --fasta /path/to/proteins \
    --layers 0 8 16 24 32 40 \
    --num_steps 1000 \
    --checkpoint_dir deterministic_checkpoints \
    --seed 42
```

**New arguments:**
- `--seed`: Random seed (default: 42)
- `--no-deterministic`: Disable deterministic mode for faster training

## Execution Order

### Step 1: Verify Determinism
```bash
python scripts/deterministic_baseline.py \
    --fasta test.fasta \
    --output baseline_results/
```

### Step 2: Train PLT for Layer 0
```bash
python universal_transcoder/train_online_multi_layer.py \
    --fasta /path/to/proteins \
    --layers 0 \
    --num_steps 1000 \
    --checkpoint_dir plt_checkpoints/
```

### Step 3: Verify PLT Integration
```bash
python scripts/verify_plt_structure.py \
    --fasta test.fasta \
    --plt-checkpoints plt_checkpoints/ \
    --layers 0 \
    --output verification_results/
```

### Step 4: Scale to All Layers (if Step 3 succeeds)
```bash
python universal_transcoder/train_online_multi_layer.py \
    --fasta /path/to/proteins \
    --layers 0 8 16 24 32 40 \
    --num_steps 1000 \
    --checkpoint_dir plt_checkpoints/
```

## Key Implementation Details

### Determinism Sources Controlled

| Source | Location | Solution |
|--------|----------|----------|
| MSA subsampling | trunkv2.py | Disabled via `model.msa_module.subsample_msa = False` |
| Dropout | dropout.py | Disabled in eval mode |
| Diffusion noise | diffusionv2.py | Seed reset before forward |
| Random rotations | utils.py | Seed reset before forward |

### Pair Reconstruction

The PLT predicts token-level values `y2` with shape `[B, N, 128]`.
To reconstruct the pair representation `z` with shape `[B, N, N, 128]`, we use **outer sum**:

```python
z[i, j] = y[i] + y[j]  # Shape: [B, N, N, 128]
```

This creates a symmetric pair matrix that captures pairwise relationships from token-level predictions.

## Files Created/Modified

**Created:**
- `scripts/deterministic_baseline.py`
- `scripts/plt_insertion.py`
- `scripts/verify_plt_structure.py`
- `scripts/README.md`

**Modified:**
- `universal_transcoder/train_online_multi_layer.py`
  - Added `setup_determinism()` function
  - Added `--seed` and `--no-deterministic` arguments
  - Modified `load_boltz_model()` to disable MSA subsampling
  - Added determinism info to checkpoint metadata
