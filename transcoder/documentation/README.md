# Transcoder Training for Pairformer Layer 48

This directory contains the implementation for training a joint transcoder (sparse autoencoder) on activations from the Pairformer Layer 48 MLP.

## Overview

The transcoder learns sparse representations of both:
- **Single (s) representations**: 384-dimensional activations from `transition_s` MLP
- **Pair (z) representations**: 128-dimensional activations from `transition_z` MLP

## Architecture

**JointTranscoder**:
- Encoder S: 384 → 2048 (ReLU)
- Encoder Z: 128 → 2048 (ReLU)
- Shared latent space: 2048 dimensions with L1 sparsity penalty
- Decoder S: 2048 → 384
- Decoder Z: 2048 → 128

## Files

- `model.py`: Joint transcoder architecture
- `collect_activations.py`: Collect activations from Boltz model using PyTorch hooks
- `train.py`: Train the transcoder on collected activations
- `run_pilot.py`: End-to-end pilot run script

## Additional Documentation

- `BOLTZ_TRANSCODER_ARCHITECTURE_MATH.md`: Boltz architecture summary, transcoder background, reasons a normal transcoder does not directly fit Boltz, extension design, and full equation-level comparison.

## Quick Start (Pilot Run)

### 1. Download Boltz2 checkpoint

```bash
cd /usr/scratch/rmanimaran8/boltz
wget https://model-gateway.boltz.bio/boltz2_conf.ckpt -O boltz2_checkpoint.ckpt
```

### 2. Test with small protein

```bash
cd transcoder
./test_collection.sh
```

This processes the test protein and collects activations from layer 48.

### 3. Verify activations were collected

```bash
ls -lh pilot_activations/
python -c "import numpy as np; d = np.load('pilot_activations/batch_00000.npz'); print('Keys:', list(d.keys()))"
```

### 4. Train the transcoder

```bash
python train.py \
    --activations pilot_activations \
    --checkpoints pilot_checkpoints \
    --log pilot_training_log.json \
    --epochs 10 \
    --batch-size 32
```

**See [QUICKSTART.md](QUICKSTART.md) for detailed step-by-step instructions.**

## Data Format

### Activation Files (.npz)

Each `batch_*.npz` file contains:
- `input_s`: Input to transition_s MLP, shape [B, N, 384]
- `output_s`: Output from transition_s MLP, shape [B, N, 384]
- `input_z`: Input to transition_z MLP (flattened pairs), shape [B, N*N, 128]
- `output_z`: Output from transition_z MLP (flattened pairs), shape [B, N*N, 128]

### Checkpoints

Checkpoint files contain:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `epoch`, `step`: Training progress
- `metrics`: Loss and sparsity metrics

## Training Metrics

- `total_loss`: Combined reconstruction + L1 penalty
- `recon_loss`: Total reconstruction loss (s + z)
- `recon_loss_s`: Reconstruction MSE for s
- `recon_loss_z`: Reconstruction MSE for z
- `l1_loss`: L1 penalty on latent activations
- `l0_sparsity_s`: Fraction of active neurons for s (0-1)
- `l0_sparsity_z`: Fraction of active neurons for z (0-1)

## Next Steps

### After Pilot Run

1. **Verify activation collection**:
   - Check shapes in .npz files
   - Verify both s and z are captured correctly
   - Ensure hooks are triggering on layer 47

2. **Review training metrics**:
   - Check reconstruction quality
   - Monitor L0 sparsity (target: 0.05-0.15)
   - Verify loss is decreasing

3. **Scale up**:
   - Collect from 10K-50K structures
   - Train for 100+ epochs (12-24 hours)
   - Experiment with hyperparameters

### TODO: Implementation Gaps

~~The following needs to be completed:~~ **✅ COMPLETED!**

1. ~~**Data Loading** (`collect_activations.py`):~~
   - ✅ Integrated with Boltz2 data structures
   - ✅ Uses StructureV2.load() for structures
   - ✅ Uses MSA.load() for MSAs
   - ✅ Proper tokenization with Boltz2Tokenizer
   - ✅ Proper featurization with Boltz2Featurizer

2. ~~**Model Loading** (`collect_activations.py`):~~
   - ✅ Uses Boltz2.load_from_checkpoint()
   - ✅ Model in eval mode
   - ✅ Proper device handling

3. ~~**Hook Verification**:~~
   - ✅ Hooks registered on correct layer (layer 47 = layer 48)
   - ✅ Captures both transition_s and transition_z
   - ✅ Input/output shapes verified

**Ready to use!** See [QUICKSTART.md](QUICKSTART.md) for instructions.

## Hyperparameters

### Pilot Run (Testing)
- Structures: 100
- Epochs: 10
- Latent dim: 2048
- L1 coefficient: 1e-4
- Learning rate: 1e-3
- Batch size: 32

### Full Run (Production)
- Structures: 10,000-50,000
- Epochs: 100-200
- Latent dim: 2048-4096
- L1 coefficient: 1e-4 to 1e-5 (tune for target sparsity)
- Learning rate: 1e-3 (with decay)
- Batch size: 32-64

## Directory Structure

```
transcoder/
├── README.md                  # This file
├── model.py                   # JointTranscoder architecture
├── collect_activations.py     # Activation collection script
├── train.py                   # Training script
├── run_pilot.py              # Pilot pipeline script
├── pilot_activations/        # Collected activations (pilot)
├── pilot_checkpoints/        # Training checkpoints (pilot)
└── pilot_training_log.json   # Training log (pilot)
```

## References

- Boltz pairformer: `src/boltz/model/modules/trunk.py`
- Transition (MLP): `src/boltz/model/layers/transition.py`
- ESM activation collection patterns (for reference)
