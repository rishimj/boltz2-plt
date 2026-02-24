# End-to-End Transcoder Training Flow

## Overview

Train a sparse autoencoder (transcoder) to learn interpretable features from Boltz2's pairformer layer 48 MLP.

---

## Step-by-Step Process

### STEP 1: Collect Real Activations ⏳ IN PROGRESS

**What it does:**
- Loads Boltz2 model from checkpoint (1-3 hours for initialization)
- Registers hooks on layer 48's `transition_s` and `transition_z` MLPs
- Processes a protein through Boltz2's inference pipeline
- Captures MLP inputs/outputs when layer 48 executes
- Saves activations to `.npz` files

**Key Fix Applied:**
```python
# Load amino acid molecules (was empty, causing KeyError)
from boltz.data.mol import load_canonicals
molecules = load_canonicals("/usr/scratch/rmanimaran8/boltz/.boltz_cache/mols")

# Now featurizer can process protein structure
feats = featurizer.process(
    data=tokens,
    molecules=molecules,  # ✓ Fixed!
    ...
)
```

**Current Status:**
- Process ID: 3087932
- Started: ~30 seconds ago
- Stage: Model initialization (scipy.stats.truncnorm weight init)
- Expected duration: 1-3 hours total
- Log: `collection_fixed.log`
- Output: `real_activations/*.npz`

**Monitor with:**
```bash
./check_status.sh                    # Quick status
tail -f collection_fixed.log         # Watch log
ps -p $(cat collection.pid)          # Check process
```

---

### STEP 2: Train Transcoder (NOT STARTED - waiting for Step 1)

**What it does:**
- Loads activation files from `real_activations/`
- Initializes JointTranscoder (2.1M parameters)
  - Encoder_s: 384 → 2048 (sparse latent)
  - Decoder_s: 2048 → 384 (reconstruct)
  - Encoder_z: 128 → 2048 (sparse latent)
  - Decoder_z: 2048 → 384 (reconstruct)
- Trains for 100 epochs (~10-15 minutes)
- Loss = reconstruction_error + L1_sparsity_penalty
- Saves checkpoints and final model

**Command (will run automatically):**
```bash
python train.py \
  --activations real_activations \
  --checkpoints real_model_checkpoints \
  --log training_log_real.txt \
  --epochs 100 \
  --batch-size 1 \
  --lr 1e-3 \
  --device cuda
```

**Output:**
- `real_model_checkpoints/step_*.pt` - Training checkpoints
- `training_log_real.txt` - Metrics (loss, sparsity)
- `transcoder_real_final.pt` - Final trained model

---

### STEP 3: Analyze Results (NOT STARTED)

**What to check:**
1. **Reconstruction quality**: How well does it recreate the MLP transformation?
2. **Sparsity**: What % of latent neurons activate? (target: ~50%)
3. **Compare to synthetic**: Is real data better than random?

**Synthetic baseline (already have):**
- Loss: 2.89 → 0.92 (-68% reduction)
- Sparsity: ~49% L0
- Model: `transcoder_final.pt`

---

## Timeline

| Step | Status | Duration | Details |
|------|--------|----------|---------|
| 1. Collect activations | ⏳ Running | 1-3 hours | Model loading bottleneck |
| 2. Train transcoder | ⏱️ Pending | ~10 mins | 100 epochs on 1 protein |
| 3. Analyze | ⏱️ Pending | Manual | Compare synthetic vs real |

**Total time:** ~1-3 hours (mostly Step 1)

---

## What You Have Now

### ✅ Complete & Working
- Transcoder architecture (`model.py`)
- Training pipeline (`train.py`)
- Synthetic data (1.4 GB, 100 samples)
- Synthetic model (`transcoder_final.pt`, loss 2.89→0.92)

### ⏳ In Progress  
- **Real activation collection** (PID: 3087932)
  - Fix applied: Load molecules from `.boltz_cache/mols`
  - Currently: Model initialization phase
  - Check: `./check_status.sh`

### ⏱️ Pending
- Train on real activations (auto-runs after collection)
- Compare synthetic vs real results

---

## Quick Commands

```bash
# Check collection status
./check_status.sh

# Watch log in real-time
tail -f collection_fixed.log

# Run full pipeline (after collection completes)
./run_pipeline.sh

# Or run training manually
python train.py --activations real_activations \
  --checkpoints real_model_checkpoints \
  --epochs 100 --batch-size 1 --device cuda
```

---

## Key Files

```
transcoder/
├── collect_activations_fixed.py    # Activation collector (RUNNING)
├── train.py                        # Training script
├── model.py                        # JointTranscoder architecture
├── check_status.sh                 # Status checker
├── run_pipeline.sh                 # Full pipeline script
│
├── collection_fixed.log            # Current collection log
├── collection.pid                  # Process ID
│
├── real_activations/               # Will contain collected data
│   └── protein_*.npz              # [Pending]
│
├── pilot_activations_synthetic/   # Synthetic data (complete)
│   ├── batch_00000.npz
│   └── ...
│
├── transcoder_final.pt            # Synthetic model (complete)
└── training_log.txt               # Synthetic training log
```

---

## Next Steps

1. **Wait ~1-3 hours** for model loading
2. **Collection completes** → saves to `real_activations/*.npz`
3. **Training starts** → 100 epochs (~10 mins)
4. **Compare results** → real vs synthetic performance

Check status: `./check_status.sh`
