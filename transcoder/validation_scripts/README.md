# Pipeline Validation Scripts

Tests to verify Boltz2 pipeline reproducibility and transcoder fidelity.

---

## Quick Start

### Test 1: Reproducibility Only (No Transcoder)

Verify that running Boltz2 twice produces identical outputs:

```bash
cd /usr/scratch/rmanimaran8/boltz/transcoder
source ../boltz_env/bin/activate

python validation_scripts/verify_pipeline_reproducibility.py \
    --fasta ../examples/prot.fasta \
    --no-transcoder \
    --output validation_output/
```

**Expected result:**
```
Reproducibility Check (Baseline vs Control)
   Max coordinate difference: 0.00e+00
   ✅ PERFECT reproducibility - outputs identical!
```

---

### Test 2: Transcoder Intervention

Test if transcoder preserves functional information:

```bash
python validation_scripts/verify_pipeline_reproducibility.py \
    --fasta ../examples/prot.fasta \
    --transcoder universal_transcoder/checkpoints/universal_transcoder_final.pt \
    --output validation_output/
```

**Expected result** (with R² = 0.58):
```
Intervention Effect (Baseline vs Transcoder)
   Coordinate RMSD: 1.8 Å
   Max atom deviation: 3.2 Å
   ✅ GOOD: Structures very similar
   → Transcoder preserves most important features
```

---

## What This Tests

### Baseline Run
- Normal Boltz2 prediction
- Captures layer 47 activations (s and z)
- Saves final structure and confidence

### Control Run
- Runs Boltz2 again on same input
- Should produce **identical** results
- Verifies deterministic behavior

### Intervention Run
- Captures layer 47 single rep (s)
- Runs s through transcoder → get z_reconstructed
- **REPLACES** layer 47's pair rep (z) with z_reconstructed
- Runs rest of Boltz2 with modified activations
- Compares final structure to baseline

---

## Interpretation Guide

### Reproducibility (Baseline vs Control)

| Max Difference | Interpretation |
|----------------|----------------|
| < 1e-6 | Perfect reproducibility ✅ |
| 1e-6 to 1e-4 | Numerical precision differences (OK) |
| > 1e-4 | Non-deterministic behavior ⚠️ |

### Intervention Effect (Baseline vs Transcoder)

| RMSD | Interpretation | Meaning |
|------|----------------|---------|
| < 0.5 Å | Nearly identical | Transcoder perfect ✅ |
| 0.5-2 Å | Very similar | Transcoder preserves fold ✅ |
| 2-5 Å | Moderately different | Some information lost ⚠️ |
| > 5 Å | Significantly different | Critical info lost ❌ |

### Your Transcoder (R² = 0.58)

**Predicted RMSD:** 1-3 Å

- **Optimistic:** R² measures per-residue reconstruction accuracy
  - Missing 42% variance might be noise/redundancy
  - Structure could be nearly identical (RMSD < 1 Å)

- **Realistic:** Some information loss
  - Core fold preserved, minor loop differences
  - RMSD ~ 2 Å

- **Pessimistic:** Significant information loss
  - Important structural features degraded
  - RMSD > 3 Å

---

## Understanding the Results

### Perfect Reconstruction (RMSD < 0.5 Å)

**Interpretation:**
```
Your transcoder with R² = 0.58 still preserves the fold perfectly!
→ The "missing" 42% was redundant/noise
→ The 16 sparse features capture everything important
```

**Conclusion:** Safe to use for interpretability analysis

---

### Good Reconstruction (RMSD 1-2 Å)

**Interpretation:**
```
Your transcoder preserves the main structural features
→ Core secondary structures intact
→ Minor deviations in loops/flexible regions
```

**Conclusion:** Mostly reliable, but be cautious about fine details

---

### Poor Reconstruction (RMSD > 3 Å)

**Interpretation:**
```
Your transcoder loses critical information
→ Need to improve reconstruction quality before analysis
```

**Action items:**
1. Increase k (16 → 32): More features active
2. Train longer (500 → 5000 steps)
3. Collect more data (2 → 100 proteins)
4. Increase d_hidden (2048 → 4096)

---

## Files Generated

```
validation_output/
├── validation_results_20260308_143022.json    # Numerical results
└── layer47_activations_20260308_143022.npz    # Baseline activations
```

### `validation_results_*.json`

```json
{
  "timestamp": "20260308_143022",
  "fasta_path": "../examples/prot.fasta",
  "transcoder_used": true,
  "results": {
    "reproducibility_max_diff": 0.0,
    "intervention_rmsd": 1.85,
    "intervention_max_diff": 3.21,
    "confidence_correlation": 0.92,
    "confidence_mae": 2.4
  }
}
```

### `layer47_activations_*.npz`

```python
data = np.load('layer47_activations_20260308_143022.npz')
print(data['layer47_s'].shape)  # (1, N, 384)  - single rep
print(data['layer47_z'].shape)  # (1, N, N, 128) - pair rep
```

---

## Advanced Usage

### Test Multiple Proteins

```bash
for fasta in ../examples/*.fasta; do
    python validation_scripts/verify_pipeline_reproducibility.py \
        --fasta "$fasta" \
        --output validation_output/
done
```

### Analyze Results

```python
import json
import glob

results = []
for file in glob.glob('validation_output/validation_results_*.json'):
    with open(file) as f:
        results.append(json.load(f))

# Compute statistics
rmsds = [r['results']['intervention_rmsd'] for r in results]
mean_rmsd = np.mean(rmsds)
std_rmsd = np.std(rmsds)

print(f"Mean RMSD: {mean_rmsd:.2f} ± {std_rmsd:.2f} Å")
```

---

## Technical Details

### How Intervention Works

```python
# Normal Boltz2:
layer47_input → [transition_s] → s_output
                [transition_z] → z_output → structure_module

# With intervention:
layer47_input → [transition_s] → s_output
                                      ↓
                                 [Transcoder]
                                      ↓
                                  z_recon
                                      ↓
                [✂️ HOOK REPLACES] → structure_module
```

The hook **intercepts** layer 47's pair representation output and **replaces** it with the transcoder's reconstruction before it reaches the structure prediction module.

### Why This Matters

**Question:** Does your transcoder preserve the information Boltz2 needs?

**Test:** If transcoder reconstruction → similar structure, then YES!

**Applications:**
1. Feature analysis: Safe to interpret sparse features
2. Interventions: Can manipulate features to steer predictions
3. Ablations: Can remove features to test necessity

---

## Troubleshooting

### Error: "No layer 64 found"

**Cause:** Boltz2 has only 48 pairformer layers (0-47)

**Solution:** The script uses layer 47 (final pairformer layer)

### Error: "Shape mismatch in hook"

**Cause:** Transcoder output shape doesn't match expected input

**Solution:** Check that:
- Transcoder trained on correct dimensions (384→128)
- Pair representation expanded correctly [N, 128] → [N, N, 128]

### Warning: "CUDA out of memory"

**Cause:** Protein too large

**Solution:** 
```bash
# Use CPU instead
CUDA_VISIBLE_DEVICES="" python validation_scripts/verify_pipeline_reproducibility.py ...
```

---

## Next Steps

After running validation:

1. **If RMSD < 2 Å:** Proceed with feature interpretation
2. **If RMSD 2-5 Å:** Improve transcoder, then re-test
3. **If RMSD > 5 Å:** Major transcoder improvements needed

See `NEXT_STEPS_IMPLEMENTATION_PLAN.md` for detailed guidance on:
- Multi-layer PLT
- Feature interpretation
- Targeted interventions
