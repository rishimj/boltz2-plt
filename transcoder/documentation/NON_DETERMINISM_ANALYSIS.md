# BOLTZ2 Non-Determinism Analysis

## Summary of Findings

The verify pipeline reproducibility test revealed that **BOLTZ2 produces non-deterministic outputs** across identical runs. Through code analysis, I've identified the root causes.

## Root Cause: MSA Subsampling with torch.randperm()

### Primary Source of Non-Determinism

**Location:** [src/boltz/model/modules/trunkv2.py](../src/boltz/model/modules/trunkv2.py#L632-L635)

```python
# Subsample the MSA
if self.subsample_msa:
    msa_indices = torch.randperm(msa.shape[1])[: self.num_subsampled_msa]
    m = m[:, msa_indices]
    msa_mask = msa_mask[:, msa_indices]
```

**The Problem:**
- `torch.randperm()` generates a random permutation of MSA sequence indices
- This occurs **even during inference** (eval mode), not just training
- Default configuration has `subsample_msa=True` with `num_subsampled_msa=1024`
- No seed is set for this random operation

**Impact:**
- Different MSA sequences are selected on each forward pass
- This randomness occurs BEFORE the pairformer layers
- All downstream activations (including layer 47) are affected
- The non-determinism propagates through all subsequent layers

## Test Results

From [validation_test.log](validation_test.log):

### Layer 47 Outputs (After 47 Pairformer Layers):

**Single representation (s):** shape (1, 117, 384)
- Max difference: **3.49e+00**
- Mean difference: **2.28e-01**
- Status: ❌ DIFFERENT

**Pair representation (z):** shape (1, 117, 117, 128)
- Max difference: **9.79e-01**
- Mean difference: **3.07e-02**
- Status: ❌ DIFFERENT

## Where Does Non-Determinism Start?

The non-determinism **does NOT start at the pairformer layer 47 output**. It starts much earlier:

1. **MSA Module** (before pairformer):
   - MSA sequences are randomly subsampled using `torch.randperm()`
   - This creates different MSA embeddings on each run

2. **Pairformer Input**:
   - The pair representation `z` receives different MSA-processed inputs
   
3. **All 48 Pairformer Layers**:
   - Layers 0-63 ALL receive different inputs
   - Layer 47 is NOT special - it just propagates the randomness

4. **Final Output**:
   - Structure predictions vary due to upstream randomness

## Secondary Sources (Less Likely During Inference)

### Dropout (Controlled by Training Mode)
**Location:** [src/boltz/model/layers/dropout.py](../src/boltz/model/layers/dropout.py)

```python
def get_dropout_mask(dropout: float, z: Tensor, training: bool, ...):
    dropout = dropout * training  # Disabled when training=False
    # ...
    d = torch.rand(v.shape, ...) >= dropout
```

- Dropout uses `torch.rand()` but is multiplied by `training` flag
- In eval mode (`model.eval()`), dropout should be 0
- **Not the primary cause during inference**

### MSA Random Subset Selection (in Featurizer)
**Location:** [src/boltz/data/feature/featurizerv2.py](../src/boltz/data/feature/featurizerv2.py#L414)

```python
if random_subset:
    indices = random.choice(
        np.arange(1, num_seqs), size=max_seqs - 1, replace=False
    )
```

- Uses the `random` generator passed to featurizer
- In test_boltz_reproducibility.py, a fixed seed (42) is used
- **Should be deterministic if seed is fixed**

## Solutions

### Option 1: Disable MSA Subsampling (RECOMMENDED)
```python
model = Boltz2.load_from_checkpoint(checkpoint_path)
model.msa_module.subsample_msa = False
model.eval()
```

### Option 2: Make MSA Subsampling Deterministic
Patch the MSA module to use a fixed seed or deterministic selection:

```python
# Instead of:
msa_indices = torch.randperm(msa.shape[1])[: self.num_subsampled_msa]

# Use:
torch.manual_seed(42)  # Set before each forward pass
msa_indices = torch.randperm(msa.shape[1])[: self.num_subsampled_msa]

# Or use deterministic selection (first N):
msa_indices = torch.arange(min(self.num_subsampled_msa, msa.shape[1]))
```

### Option 3: Set Global PyTorch Determinism
```python
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Then before each forward pass:
torch.manual_seed(42)  # Reset seed
output = model(feats)
```

## Implications for Transcoder Training

### Current Situation:
- Each time you run BOLTZ2 on the same protein, you get different activations
- The activations you're training your transcoder on are **non-reproducible**
- This explains variability in your training data collection

### Impact on Your Pipeline:
1. **Data Collection Phase:**
   - Running the same protein twice gives different activation values
   - Your "real_activations" dataset has inherent randomness

2. **Validation Phase:**
   - Testing transcoder reconstruction is complicated
   - Can't verify if intervention works because baseline changes

3. **Training Stability:**
   - If you re-collect activations, they'll be different
   - Can't reproduce exact training runs

### Recommendations:

**For Data Collection:**
```python
# Disable MSA subsampling when collecting activations
model = Boltz2.load_from_checkpoint(checkpoint)
if hasattr(model, 'msa_module'):
    model.msa_module.subsample_msa = False
model.eval()
```

**For Validation:**
- Use the same approach to ensure reproducible baseline
- This allows proper comparison of transcoder interventions

**For Transcoder Intervention:**
- Even if you perfectly reconstruct layer 47 activations (R² = 1.0)
- The final structure may still differ due to randomness in:
  - Diffusion sampling steps
  - Other potential random operations
- You need deterministic BOLTZ2 to properly test causality

## Next Steps

1. **Verify MSA Subsampling is the Cause:**
   - Run `diagnose_nondeterminism.py` to confirm
   - Compare runs with/without MSA subsampling

2. **Update Collection Scripts:**
   - Disable MSA subsampling in all activation collection scripts
   - Re-collect activations with deterministic BOLTZ2

3. **Update Validation Scripts:**
   - Ensure reproducibility before testing interventions
   - Use deterministic forward passes

4. **Check Diffusion Module:**
   - Investigate if diffusion sampling also has randomness
   - May need to fix seeds there too for full reproducibility
