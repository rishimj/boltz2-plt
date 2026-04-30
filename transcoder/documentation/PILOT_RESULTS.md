# Transcoder Pilot Results

## Summary

Successfully implemented and tested a joint transcoder (sparse autoencoder) for Boltz2 pairformer layer 48 MLP activations.

## Architecture

**Target:** Boltz2 Pairformer Layer 48 (index 47)
- `transition_s`: Single representation MLP (384→1536→384)
- `transition_z`: Pair representation MLP (128→512→128)

**Transcoder:**
- Input dimensions: 384 (s) + 128 (z)
- Latent dimension: 2048 (shared latent space)
- Total parameters: 2,101,760
- Architecture: Joint encoder → shared latents → separate decoders

## Test Results (Synthetic Data)

Trained on 100 synthetic samples (5 batches × 20 samples each) from a 117-residue protein.

### Training Progress (2 epochs)

| Metric | Epoch 1 | Epoch 2 |
|--------|---------|---------|
| Total Loss | 2.888 | 0.923 |
| Recon Loss S | 1.698 | 0.596 |
| Recon Loss Z | 1.190 | 0.327 |
| L1 Loss | 1.098 | 1.064 |
| L0 Sparsity S | 49.8% | 49.4% |
| L0 Sparsity Z | 49.9% | 49.8% |

### Key Observations

1. **Loss Reduction**: Total loss decreased from 2.89 to 0.92 (-68%)
2. **Reconstruction**: Both s and z reconstruction improved significantly
   - S: 1.70 → 0.60 (-65%)
   - Z: 1.19 → 0.33 (-72%)
3. **Sparsity**: Achieved ~49% sparsity (half of latents active)
4. **L1 Regularization**: Stable at ~1.06-1.10

### Historical Pilot Artifacts

The original pilot produced:

- `transcoder_final.pt`: final trained model
- `pilot_model/step_000025.pt`: checkpoint with optimizer state
- `training_log.txt`: detailed training metrics

Those pilot artifacts are no longer kept in the repository root as active project outputs after the April 30, 2026 cleanup.

## Implementation Details

### Data Format

Activation batches stored as `.npz` files with:
- `input_s`: [batch, tokens, 384] - Input to transition_s
- `output_s`: [batch, tokens, 384] - Output from transition_s  
- `input_z`: [batch, tokens, tokens, 128] - Input to transition_z
- `output_z`: [batch, tokens, tokens, 128] - Output from transition_z

### Training Configuration

```python
latent_dim = 2048
l1_coeff = 1e-4
learning_rate = 1e-3
batch_size = 4
epochs = 2
device = "cuda"
```

## Known Issues & Next Steps

### Issue: Real Activation Collection Failed

**Problem:** Checkpoint version mismatch (Lightning v2.5.0.post0 vs v2.5.0)
- Boltz2 model forward pass fails with `AttentionPairBias` signature error
- Unable to collect real activations from layer 48

**Workaround:** Used synthetic data to validate transcoder architecture

### Next Steps

1. **Fix Checkpoint Compatibility:**
   - Upgrade PyTorch Lightning to v2.5.0.post0, OR
   - Download Boltz2 v2.5.0 checkpoint (exact version match), OR
   - Fix AttentionPairBias.forward() signature in code

2. **Collect Real Activations:**
   - Run `collect_activations.py` on real protein structures
   - Process multiple proteins for diversity

3. **Scale Up Training:**
   - Train on 1000+ structures
   - Increase latent dimensions (4096, 8192)
   - Tune sparsity (L1 coefficient)

4. **Analysis:**
   - Interpret learned features
   - Compare s vs z latent usage
   - Ablation studies

## Files Created Historically

```
pilot implementation:
├── model.py
├── collect_activations.py
├── train.py
├── create_synthetic_activations.py
├── transcoder_final.pt
├── pilot_activations_synthetic/
├── pilot_model/
└── training_log.txt
```

## Usage

### Train Transcoder (with real data once collected)

```bash
python collect_activations.py \
  --checkpoint ../boltz2_checkpoint.ckpt \
  --structures ../data/structures \
  --msa ../data/msa \
  --output activations \
  --max-structures 100 \
  --layer 47

python train.py \
  --activations activations \
  --checkpoints checkpoints \
  --epochs 10 \
  --device cuda
```

### Test with Synthetic Data

```bash
python create_synthetic_activations.py
python train.py \
  --activations pilot_activations_synthetic \
  --checkpoints pilot_model \
  --epochs 5 \
  --device cuda
```

## References

- Boltz2 Paper: https://arxiv.org/abs/2503.03686
- Sparse Autoencoders: Towards Monosemanticity (Anthropic)
- Architecture based on joint transcoder design from SAE literature
