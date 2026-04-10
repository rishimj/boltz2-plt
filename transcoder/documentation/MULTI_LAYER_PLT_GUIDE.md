# Multi-Layer PLT Training Pipeline Guide

**Version:** 1.0  
**Last Updated:** March 9, 2026  
**Status:** Production Ready ✅

**Important Terminology:**
- **PLT** = **Per-Layer Transcoder** (one sparse autoencoder per pairformer layer)
- This is NOT "Piecewise Linear Transcoder" - the name reflects training independent transcoders for different layers

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [End-to-End Flow](#end-to-end-flow)
4. [Component Documentation](#component-documentation)
5. [Code Deep Dive](#code-deep-dive)
6. [Usage Guide](#usage-guide)
7. [Configuration](#configuration)
8. [Output Structure](#output-structure)
9. [Troubleshooting](#troubleshooting)
10. [Key Considerations](#key-considerations)

---

## Overview

### Purpose

The Multi-Layer PLT (Per-Layer Transcoder) Training Pipeline trains **independent sparse autoencoders** for multiple layers in Boltz2's pairformer trunk simultaneously. This enables:

- **Layer-wise feature analysis**: Compare learned features across network depth
- **Feature evolution tracking**: Understand how representations change from early to late layers
- **Computational efficiency**: Single forward pass collects activations from all target layers
- **Systematic evaluation**: Standardized training and validation across all layers

### Design Philosophy

**Independent Transcoders:** Each layer gets its own PLT model, following the principle that different layers learn different feature spaces. This allows:
- Layer-specific feature dictionaries (2048 features per layer)
- Independent optimization trajectories
- Easy comparison of layer complexity (via sparsity, dead neurons, etc.)

**Separation of Concerns:**
- **Collection** (`collect_multi_layer.py`): Extract activations from Boltz2
- **Training** (`train_multi_layer.py`): Train separate PLTs for each layer
- **Validation** (`validate_multi_layer.py`): Evaluate all trained models

---

## Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BOLTZ2 PAIRFORMER (48 LAYERS)                    │
│                                                                     │
│  Layer 0 ──┐                                                       │
│  Layer 8 ──┼── HOOK & CAPTURE ──→ Activations Collection          │
│  Layer 16 ─┤                      (collect_multi_layer.py)         │
│  Layer 24 ─┤                                                        │
│  Layer 32 ─┤                                                        │
│  Layer 40 ─┘                                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
        ┌───────────────────────────────────────────┐
        │     MULTI-LAYER ACTIVATION STORAGE        │
        │                                           │
        │  layer_00/batch_00000.npz                │
        │  layer_08/batch_00000.npz                │
        │  layer_16/batch_00000.npz                │
        │  layer_24/batch_00000.npz                │
        │  layer_32/batch_00000.npz                │
        │  layer_40/batch_00000.npz                │
        └───────────────────────────────────────────┘
                                ↓
        ┌───────────────────────────────────────────┐
        │         PARALLEL PLT TRAINING             │
        │      (train_multi_layer.py)               │
        │                                           │
        │  Layer 0  → PLT_0  (2048 features)       │
        │  Layer 8  → PLT_8  (2048 features)       │
        │  Layer 16 → PLT_16 (2048 features)       │
        │  Layer 24 → PLT_24 (2048 features)       │
        │  Layer 32 → PLT_32 (2048 features)       │
        │  Layer 40 → PLT_40 (2048 features)       │
        └───────────────────────────────────────────┘
                                ↓
        ┌───────────────────────────────────────────┐
        │       VALIDATION & METRICS                │
        │   (validate_multi_layer.py)               │
        │                                           │
        │  • Reconstruction R²                      │
        │  • MSE/RMSE per layer                     │
        │  • Sparsity analysis                      │
        │  • Dead neuron count                      │
        │  • Weight norm verification               │
        └───────────────────────────────────────────┘
```

### Data Flow

```
FASTA File → Boltz2 Featurizer → Forward Pass → Hook Capture
                                                      ↓
                                    ┌─────────────────────────────┐
                                    │  LayerActivationCollector   │
                                    │  (per-layer storage)        │
                                    └─────────────────────────────┘
                                                ↓
                    ┌───────────────────────────────────────────┐
                    │         Save to NPZ Files                 │
                    │  Keys: input_s, output_s, input_z,       │
                    │        output_z                           │
                    └───────────────────────────────────────────┘
                                                ↓
                    ┌───────────────────────────────────────────┐
                    │  UniversalActivationDataset               │
                    │  (loads batches for training)             │
                    └───────────────────────────────────────────┘
                                                ↓
                    ┌───────────────────────────────────────────┐
                    │  UniversalTranscoder Forward Pass         │
                    │  s1 → [Encoder] → TopK → [Decoder] → y1  │
                    │  s2 → [Encoder] → TopK → [Decoder] → y2  │
                    └───────────────────────────────────────────┘
```

---

## End-to-End Flow

### Phase 1: Activation Collection

**Script:** `collection_scripts/collect_multi_layer.py`

```
1. Load Boltz2 Model
   ├── Load checkpoint from disk
   ├── Initialize model architecture
   └── Move to CUDA device

2. Register Hooks
   ├── For each target layer (0, 8, 16, 24, 32, 40):
   │   ├── Create LayerActivationCollector
   │   ├── Hook transition_s (single representation MLP)
   │   └── Hook transition_z (pair representation MLP)

3. Process FASTA Files
   ├── For each protein:
   │   ├── Parse FASTA → Target
   │   ├── Load MSAs (if available)
   │   ├── Tokenize → Tokens
   │   ├── Featurize → Features
   │   ├── Add batch dimension
   │   ├── Forward pass (activations captured via hooks)
   │   └── Save activations to layer-specific directories

4. Save Activations
   ├── For each layer:
   │   ├── Concatenate batches
   │   ├── Flatten pair representations [B,N,N,128] → [B,N²,128]
   │   └── Save as batch_XXXXX.npz
```

**Key Code Points:**

- **Line 35-50** (`LayerActivationCollector.__init__`): Hook registration per layer
- **Line 158-175** (`MultiLayerActivationCollector`): Manages multiple collectors
- **Line 303-310**: FASTA parsing with correct arguments
- **Line 340-360**: Forward pass with activation capture

### Phase 2: Training

**Script:** `universal_transcoder/train_multi_layer.py`

```
1. Initialize Training
   ├── For each layer index:
   │   ├── Check if activation data exists
   │   ├── Count number of batches
   │   └── Skip if no data

2. Train Single Layer
   ├── Load UniversalActivationDataset
   ├── Create UniversalTranscoder model
   │   ├── Encoder: Linear(384 → 2048)
   │   ├── TopK activation (k=16)
   │   ├── Decoder Y1: Parameter(2048 → 128)
   │   └── Decoder Y2: Parameter(2048 → 128)
   ├── Initialize Adam optimizer
   └── Training loop:
       ├── Dual-pass forward (s1 and s2)
       ├── Compute combined loss:
       │   ├── Reconstruction (4 terms: y1/y2 from s1/s2)
       │   ├── Consistency (2 terms: y1/y2 agreement)
       │   └── AuxK (dead neuron resurrection)
       ├── Backward + optimize
       └── Normalize decoder weights (unit norm)

3. Save Checkpoint
   ├── Model state dict
   ├── Optimizer state
   ├── Hyperparameters
   └── Training metrics history
```

**Key Code Points:**

- **Line 76-140** (`train_single_layer`): Per-layer training orchestration
- **Line 142-250** (`train_multi_layer`): Sequential training across all layers
- **Line 195-230** (in `train_universal.py`): Dual-pass loss computation
- **Line 270**: Weight normalization after each step

### Phase 3: Validation

**Script:** `universal_transcoder/validate_multi_layer.py`

```
1. Load Trained Model
   ├── Load checkpoint
   ├── Reconstruct UniversalTranscoder
   └── Set to eval() mode

2. Verify Weight Norms
   ├── Check encoder norms (should be ~1.2 after training)
   ├── Check decoder norms (should be exactly 1.0)
   └── Report if not unit norm

3. Compute Metrics
   ├── For each validation batch:
   │   ├── Forward pass s1 → y1_pred, y2_pred
   │   ├── Expand predictions to match pair dimensions
   │   ├── Compute MSE & RMSE
   │   ├── Compute R² (variance explained)
   │   └── Record sparsity
   ├── Aggregate across batches
   └── Save validation summary

4. Generate Report
   ├── Print table with per-layer metrics
   ├── Save JSON summary
   └── Flag any anomalies (negative R², high dead neurons)
```

**Key Code Points:**

- **Line 25-50** (`load_transcoder`): Checkpoint loading
- **Line 107-125**: Weight norm verification
- **Line 170-210**: Metric computation loop
- **Line 315-340**: Multi-layer summary generation

---

## Component Documentation

### 1. `collect_multi_layer.py`

**Location:** `collection_scripts/collect_multi_layer.py`  
**Lines of Code:** 443  
**Purpose:** Extract activations from multiple Boltz2 pairformer layers

#### Key Classes

##### `LayerActivationCollector` (Lines 23-135)

```python
class LayerActivationCollector:
    """Collects activations from a single pairformer layer."""
    
    def __init__(self, model, layer_idx, device='cuda'):
        self.target_layer = model.pairformer_module.layers[layer_idx]
        self._register_hooks()
```

**Responsibilities:**
- Register forward hooks on `transition_s` and `transition_z`
- Capture input/output activations during forward pass
- Store activations in CPU memory (to avoid GPU OOM)
- Save collected activations to NPZ files

**Critical Implementation Details:**

```python
# Line 64-67: Hook captures INPUT to transition
def hook_s_input(module, input, output):
    x = input[0].detach().cpu()  # MUST detach & move to CPU
    self.activations['input_s'].append(x)

# Line 119-134: Flatten pair representations before saving
B, N1, N2, D = inp.shape
input_z_list.append(inp.reshape(B, N1 * N2, D))  # [B,N,N,128] → [B,N²,128]
```

**Why flatten?** Training script expects [B, N², 128] format for efficient batching.

##### `MultiLayerActivationCollector` (Lines 158-187)

```python
class MultiLayerActivationCollector:
    """Collects activations from multiple pairformer layers simultaneously."""
    
    def __init__(self, model, layer_indices, device='cuda'):
        self.collectors = {
            idx: LayerActivationCollector(model, idx, device)
            for idx in layer_indices
        }
```

**Why separate collectors?** Each layer needs independent storage. Can't use shared buffer because layers fire at different times during forward pass.

#### Key Functions

##### `collect_activations_multi_layer` (Lines 190-390)

**Critical Code Sections:**

```python
# Line 303-305: Parse FASTA with required args
from boltz.data.parse.fasta import parse_fasta
target = parse_fasta(fasta_file, molecules, moldir, boltz2=True)
```

**⚠️ Common Error:** Forgetting `molecules, moldir, boltz2=True` arguments causes `TypeError`.

```python
# Line 340-350: Forward pass with error handling
try:
    output = model(feats, recycling_steps=recycling_steps)
except Exception as e:
    # Structure module may fail, but activations already captured
    print(f"Note: {e}")
```

**Why try/except?** Boltz2's structure prediction module can fail (e.g., missing atoms), but pairformer activations are captured by hooks BEFORE structure module runs.

### 2. `train_multi_layer.py`

**Location:** `universal_transcoder/train_multi_layer.py`  
**Lines of Code:** 336  
**Purpose:** Orchestrate training of separate PLTs for each layer

#### Key Architecture

```python
class Args:
    """Container for training arguments."""
    # Line 23-27: Simple dataclass pattern
```

**Why custom Args class?** `train_universal_transcoder()` expects an argparse namespace. This allows programmatic creation.

#### Key Functions

##### `train_single_layer` (Lines 30-120)

**Flow Chart:**
```
Check data exists → Create Args → Call train_universal_transcoder()
        ↓                                      ↓
    Skip if empty                    Returns (model, metrics, time)
                                                ↓
                                    Extract final metrics → Return
```

**Error Handling:**

```python
# Line 48-56: Graceful failure
if len(npz_files) == 0:
    return {
        'layer_idx': layer_idx,
        'status': 'skipped',
        'reason': 'no_batches',
    }
```

**Result Structure:**
```python
{
    'layer_idx': 40,
    'status': 'success',
    'training_time_seconds': 2.06,
    'final_loss_total': 1538.45,
    'final_loss_reconstruction': 1357.84,
    'final_loss_consistency': 180.61,
    'final_dead_neurons': 0,
    'checkpoint_dir': 'multi_layer_checkpoints/layer_40'
}
```

##### `train_multi_layer` (Lines 123-280)

**Sequential Training Logic:**

```python
# Line 173-195: Sequential iteration
for layer_idx in layer_indices:
    result = train_single_layer(...)
    
    if result['status'] == 'success':
        successful_layers.append(layer_idx)
    elif result['status'] == 'failed':
        failed_layers.append(layer_idx)
```

**Why sequential?** Could be parallelized (different GPUs), but sequential ensures:
- Predictable memory usage
- Clear error attribution
- Simpler debugging

**Summary Generation:**

```python
# Line 230-250: Detailed results table
print(f"{'Layer':<8} {'Loss':>12} {'Recon':>12} {'Consist':>12}")
for result in all_results:
    if result['status'] == 'success':
        print(f"{layer_idx:<8} {loss:>12.6f} ...")
```

### 3. `validate_multi_layer.py`

**Location:** `universal_transcoder/validate_multi_layer.py`  
**Lines of Code:** 410  
**Purpose:** Validate trained PLTs and compute reconstruction metrics

#### Key Functions

##### `load_transcoder` (Lines 27-56)

```python
def load_transcoder(checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    hparams = checkpoint['hyperparameters']
    
    model = UniversalTranscoder(
        d_model=hparams['d_model'],      # 384
        d_hidden=hparams['d_hidden'],    # 2048
        d_pair=hparams['d_pair'],        # 128
        k=hparams['k'],                  # 16
        auxk=hparams['auxk'],            # 32
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, hparams
```

**Critical:** Must reconstruct model with EXACT same hyperparameters, or state_dict won't load.

##### `compute_r_squared` (Lines 59-66)

```python
def compute_r_squared(y_true, y_pred):
    """Compute R² = 1 - (SS_res / SS_tot)"""
    ss_res = torch.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared.item()
```

**Interpretation:**
- R² = 1.0: Perfect reconstruction
- R² = 0.5: Explains 50% of variance
- R² = 0.0: No better than predicting mean
- R² < 0: Worse than predicting mean (needs more training)

##### `validate_single_layer` (Lines 69-260)

**Weight Norm Verification:**

```python
# Line 113-117: Check decoder unit norms
decoder_y1_norms = torch.norm(model.decoder_y1, dim=0)
decoder_y1_norm_mean = decoder_y1_norms.mean().item()

# Should be exactly 1.0 (PLT requirement)
decoder_y1_is_unit = abs(decoder_y1_norm_mean - 1.0) < 0.01
```

**Why check?** PLT decoders MUST have unit norm for interpretability (feature magnitude = activation strength).

**Metric Computation Loop:**

```python
# Line 185-200: Forward pass and metrics
for batch_file in batch_files:
    # Load batch
    s1 = torch.from_numpy(data['input_s']).to(device)
    
    # Forward pass
    y1_pred, y2_pred, _, _, _ = model(s1_flat)
    
    # Expand to match pair dimensions
    y1_pred_expanded = y1_pred.unsqueeze(1).expand(B*N, N, -1).reshape(B*N_sq, -1)
    
    # Compute metrics
    mse_y1 = F.mse_loss(y1_pred_expanded, y1_true_flat).item()
    r2_y1 = compute_r_squared(y1_true_flat, y1_pred_expanded)
```

**Why expand?** Model predicts [B*N, 128] (per-residue), but target is [B*N², 128] (per-pair). Expansion broadcasts each residue prediction to all N pairs.

---

## Code Deep Dive

### Critical Code Sections You Must Understand

#### 1. Hook Registration Pattern

**File:** `collect_multi_layer.py`  
**Lines:** 48-84

```python
def _register_hooks(self):
    """Register forward hooks on transition_s and transition_z."""
    
    # Hook for transition_s (single representation MLP)
    def hook_s_input(module, input, output):
        x = input[0].detach().cpu()  # ← CRITICAL: detach() prevents backprop, cpu() saves memory
        self.activations['input_s'].append(x)
    
    def hook_s_output(module, input, output):
        x = output.detach().cpu()  # ← Capture OUTPUT of transition
        self.activations['output_s'].append(x)
    
    # Register hooks
    h1 = self.target_layer.transition_s.register_forward_hook(hook_s_input)
    h2 = self.target_layer.transition_s.register_forward_hook(hook_s_output)
    self.hooks.extend([h1, h2])
```

**Why detach()?** Without `.detach()`, PyTorch keeps computation graph alive → GPU memory leak.

**Why cpu()?** Activations accumulate across batches. Keeping on GPU causes OOM.

**Hook lifecycle:**
```
Forward pass starts
    ↓
Layer N reached
    ↓
hook_s_input called  ← Captures INPUT to transition_s
    ↓
transition_s executes
    ↓
hook_s_output called ← Captures OUTPUT from transition_s
    ↓
Forward pass continues
```

#### 2. Dual-Pass Training Loss

**File:** `universal_transcoder/train_universal.py`  
**Lines:** 176-230

```python
# === DUAL-PASS FORWARD ===
# Pass 1: s1 (input_s) → predict y1, y2
y1_pred1, y2_pred1, aux_y1_1, aux_y2_1, dead_mask = model(s1_flat)

# Pass 2: s2 (output_s) → predict y1, y2
y1_pred2, y2_pred2, aux_y1_2, aux_y2_2, _ = model(s2_flat)

# === COMBINED LOSS ===
# 1. Reconstruction Loss (4 terms)
loss_recon_y1_from_s1 = F.mse_loss(y1_pred1_expanded, y1_true_flat)
loss_recon_y2_from_s1 = F.mse_loss(y2_pred1_expanded, y2_true_flat)
loss_recon_y1_from_s2 = F.mse_loss(y1_pred2_expanded, y1_true_flat)
loss_recon_y2_from_s2 = F.mse_loss(y2_pred2_expanded, y2_true_flat)

loss_reconstruction = (
    loss_recon_y1_from_s1 + 
    loss_recon_y2_from_s1 + 
    loss_recon_y1_from_s2 + 
    loss_recon_y2_from_s2
)

# 2. Consistency Loss (2 terms)
# Predictions from s1 and s2 should agree
loss_consistency_y1 = F.mse_loss(y1_pred1, y1_pred2)
loss_consistency_y2 = F.mse_loss(y2_pred1, y2_pred2)

loss_consistency = loss_consistency_y1 + loss_consistency_y2
```

**Why dual-pass?**
- `s1 = input_s`: Activations BEFORE transition MLP
- `s2 = output_s`: Activations AFTER transition MLP

Both should encode similar information about pair representations. Consistency loss forces model to learn shared features.

**Loss breakdown:**
```
Total Loss = Reconstruction + Consistency + AuxK
           = (4 MSE terms) + (2 MSE terms) + (dead neuron penalty)
```

**Note:** This matches the Per-Layer Transcoder (PLT) loss formulation exactly.

#### 3. TopK Activation Function

**File:** `universal_transcoder/universal_model.py`  
**Lines:** 87-104

```python
def topK_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-K activation function.
    
    Args:
        x: Input tensor [..., D]  (e.g., [B*N, 2048])
        k: Number of top activations to keep (e.g., 16)
        
    Returns:
        Sparse tensor with only top-k values (ReLU applied)
    """
    # Find top-k indices
    topk = torch.topk(x, k=k, dim=-1, sorted=False)
    
    # Apply ReLU to top-k values
    values = F.relu(topk.values)  # ← Zero out negative values
    
    # Create sparse output
    result = torch.zeros_like(x)
    result.scatter_(dim=-1, index=topk.indices, src=values)
    
    return result
```

**Example:** Input `x = [-1.2, 3.5, 0.8, -0.3, 2.1]`, `k=2`
```
Step 1: topk(x, k=2) → values=[3.5, 2.1], indices=[1, 4]
Step 2: ReLU(values) → [3.5, 2.1] (already positive)
Step 3: scatter_() → [0, 3.5, 0, 0, 2.1]
```

**Why ReLU after TopK?** Allows negative pre-activations to be selected, but zeros them out. This gives gradient signal during training.

#### 4. Weight Normalization

**File:** `universal_transcoder/universal_model.py`  
**Lines:** 203-213

```python
def norm_weights(self):
    """Normalize decoder weights to unit norm (matching Per-Layer Transcoder)."""
    with torch.no_grad():
        # Normalize each column (feature) to unit norm
        self.decoder_y1.data /= self.decoder_y1.data.norm(dim=0, keepdim=True)
        self.decoder_y2.data /= self.decoder_y2.data.norm(dim=0, keepdim=True)
```

**Why unit norm?**

Per-Layer Transcoder interpretability requires:
- Feature magnitude = activation strength
- Weight vector = feature direction

Without normalization:
```
Feature 1: weight norm = 10 → contributes 10x more even with weak activation
Feature 2: weight norm = 0.1 → barely contributes even with strong activation
```

With normalization:
```
All features: weight norm = 1.0 → contribution proportional to activation only
```

**Called after every training step:**
```python
# Line 278 in train_universal.py
optimizer.step()
model.norm_weights()  # ← Enforce unit norm
```

---

## Usage Guide

### Minimal Test (Recommended First)

```bash
cd /usr/scratch/rmanimaran8/boltz/transcoder

# Test with 1 protein, 1 layer, 10 steps (~6 minutes)
./test_minimal_pipeline.sh

# Check results
cat minimal_test/test.log
cat minimal_test/checkpoints/layer_40/training_metrics.json
```

**Expected Output:**
```
✓ Collection successful: 1 batch(es) saved
✓ Training successful: checkpoint saved
✓ Validation successful
```

### Full Pipeline

```bash
# Run complete pipeline (10 proteins, 6 layers, 100 steps, ~1-2 hours)
./run_multi_layer_pipeline.sh

# Monitor progress
tail -f multi_layer_logs/01_collection.log
tail -f multi_layer_logs/02_training.log
tail -f multi_layer_logs/03_validation.log
```

### Custom Configuration

```bash
# Collection only
cd collection_scripts
python collect_multi_layer.py \
    --checkpoint /path/to/boltz2_conf.ckpt \
    --fasta /path/to/proteins.fasta \
    --output ../my_activations \
    --layers 0 8 16 24 32 40 47 \
    --max-proteins 50 \
    --device cuda

# Training only (if activations already collected)
cd universal_transcoder
python train_multi_layer.py \
    --data_dir ../my_activations \
    --checkpoint_dir ../my_checkpoints \
    --layers 0 8 16 24 32 40 47 \
    --num_steps 500 \
    --batch_size 20 \
    --lr 1e-3

# Validation only
python validate_multi_layer.py \
    --checkpoint_dir ../my_checkpoints \
    --data_dir ../my_activations \
    --layers 0 8 16 24 32 40 47 \
    --device cuda
```

---

## Configuration

### Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--layers` | `[0,8,16,24,32,40]` | 0-47 | Which pairformer layers to analyze |
| `--num_steps` | 100 | 10-10000 | Training iterations per layer |
| `--batch_size` | 10 | 1-100 | Samples per gradient update |
| `--lr` | 1e-3 | 1e-5 to 1e-1 | Learning rate |
| `--d_hidden` | 2048 | 512-8192 | Latent dimension (feature count) |
| `--k` | 16 | 1-128 | TopK sparsity (active features) |
| `--auxk` | 32 | k to d_hidden | Auxiliary K for dead neurons |
| `--max_proteins` | 10 | 1-1000 | Proteins to process |

### Layer Selection Strategy

**Even spacing (default):** `[0, 8, 16, 24, 32, 40]`
- Samples network depth evenly
- Good for feature evolution analysis

**Focus on depth:** `[0, 16, 32, 47]`
- Early, middle-early, middle-late, final layers
- Faster training (4 models instead of 6)

**Dense sampling:** `[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47]`
- Fine-grained feature evolution
- Longer training time

### Memory Considerations

**GPU Memory:**
- Boltz2 model: ~2GB
- Activation storage: ~500MB per protein
- Training batch: ~100MB per layer

**Recommended:**
- 8GB+ GPU for collection
- 4GB+ GPU for training
- 16GB+ GPU for large proteins (>500 residues)

**Disk Space:**
- Activations: ~50MB per protein per layer
- Checkpoints: ~20MB per layer
- Logs: ~1MB per run

Example: 10 proteins × 6 layers × 50MB = **3GB activations**

---

## Output Structure

### Directory Layout

```
transcoder/
├── multi_layer_activations/          # Collected activations
│   ├── layer_00/
│   │   ├── batch_00000.npz           # [B, N, 384] single, [B, N², 128] pair
│   │   ├── batch_00001.npz
│   │   └── ...
│   ├── layer_08/
│   ├── layer_16/
│   ├── layer_24/
│   ├── layer_32/
│   └── layer_40/
│
├── multi_layer_checkpoints/          # Trained models
│   ├── layer_00/
│   │   ├── universal_transcoder_final.pt    # Model weights + optimizer
│   │   └── training_metrics.json            # Loss curves, dead neurons
│   ├── layer_08/
│   ├── layer_16/
│   ├── layer_24/
│   ├── layer_32/
│   └── layer_40/
│
├── multi_layer_logs/                 # Execution logs
│   ├── 01_collection.log             # Activation collection output
│   ├── 02_training.log               # Training progress for all layers
│   └── 03_validation.log             # Validation metrics
│
└── multi_layer_checkpoints/
    ├── multi_layer_training_summary.json    # Aggregate training results
    └── validation_summary.json              # Aggregate validation results
```

### Activation File Format

**File:** `multi_layer_activations/layer_XX/batch_XXXXX.npz`

```python
import numpy as np

data = np.load('batch_00000.npz')

# Keys and shapes:
data['input_s']   # [B, N, 384]    Single rep input to transition_s
data['output_s']  # [B, N, 384]    Single rep output from transition_s
data['input_z']   # [B, N², 128]   Pair rep input to transition_z (FLATTENED)
data['output_z']  # [B, N², 128]   Pair rep output from transition_z (FLATTENED)
```

**Example:**
```python
# Protein with 117 residues
input_s: shape (1, 117, 384)
output_s: shape (1, 117, 384)
input_z: shape (1, 13689, 128)  # 13689 = 117²
output_z: shape (1, 13689, 128)
```

### Checkpoint Format

**File:** `multi_layer_checkpoints/layer_XX/universal_transcoder_final.pt`

```python
import torch

checkpoint = torch.load('universal_transcoder_final.pt')

# Structure:
{
    'model_state_dict': OrderedDict(...),  # encoder, decoder_y1, decoder_y2 weights
    'optimizer_state_dict': {...},         # Adam state
    'step': 100,
    'epoch': 5,
    'training_time': 2.06,
    'final_metrics': {
        'step': 100,
        'loss_total': 1538.45,
        'loss_reconstruction': 1357.84,
        'loss_consistency': 180.61,
        'dead_neurons': 0
    },
    'hyperparameters': {
        'd_model': 384,
        'd_hidden': 2048,
        'd_pair': 128,
        'k': 16,
        'auxk': 32,
        'batch_size': 10,
        'lr': 0.001
    },
    'metrics_history': [...]  # All step metrics
}
```

### Validation Summary

**File:** `multi_layer_checkpoints/validation_summary.json`

```json
{
  "num_layers": 6,
  "successful_layers": [0, 8, 16, 24, 32, 40],
  "failed_layers": [],
  "results": [
    {
      "layer_idx": 0,
      "status": "success",
      "total_samples": 10,
      "mse_y1": 594.26,
      "rmse_y1": 24.38,
      "r2_y1": 0.1285,
      "mse_y2": 78.48,
      "rmse_y2": 8.86,
      "r2_y2": -13.54,
      "avg_sparsity": 16.0,
      "is_unit_norm": false
    }
  ]
}
```

---

## Troubleshooting

### Common Errors

#### 1. `TypeError: parse_fasta() missing 2 required positional arguments`

**Cause:** Incorrect FASTA parsing call

**Fix:**
```python
# ❌ WRONG
target = parse_fasta(fasta_file)

# ✅ CORRECT
target = parse_fasta(fasta_file, molecules, moldir, boltz2=True)
```

**Location:** `collect_multi_layer.py:305`

#### 2. `AttributeError: 'Parameter' object has no attribute 'weight'`

**Cause:** Decoder is a Parameter, not nn.Linear

**Fix:**
```python
# ❌ WRONG
decoder_norms = torch.norm(model.decoder_y1.weight, dim=1)

# ✅ CORRECT (decoder_y1 IS the weight parameter)
decoder_norms = torch.norm(model.decoder_y1, dim=0)
```

**Location:** `validate_multi_layer.py:115`

#### 3. `IndexError: Dimension out of range ... but got 1`

**Cause:** active_mask has wrong dimensions for sparsity calculation

**Fix:**
```python
# ❌ WRONG (active_mask doesn't exist in return)
sparsity = active_mask.float().sum(dim=1).mean().item()

# ✅ CORRECT (TopK ensures exactly k active features)
sparsity = model.k
```

**Location:** `validate_multi_layer.py:206`

#### 4. `ValueError: Invalid FASTA path`

**Cause:** Relative path not resolved from script's working directory

**Fix:**
```python
# ❌ WRONG (relative path breaks when cwd != expected)
fasta_path = Path(fasta_path)

# ✅ CORRECT (resolve to absolute path)
fasta_path = Path(fasta_path).resolve()
```

**Location:** `collect_multi_layer.py:271`

#### 5. Model Loading Stuck (5+ minutes)

**Cause:** Normal behavior - Boltz2 is large (2GB weights)

**Progress indicators:**
```
Loading model... (visible immediately)
Creating model with 50 valid hyperparameters... (30s - 2min)
Loading weights from checkpoint... (2-5min)
✓ Model loaded (appears when done)
```

**Workaround:** Use persistent model cache:
```python
# Load once, reuse for all collections
model = Boltz2.load_from_checkpoint(checkpoint_path)
# ... collect from all proteins ...
```

### Performance Issues

#### Slow Collection (~5min per protein)

**Normal for large proteins.** Breakdown:
- Model loading: 3-5 min (one-time)
- MSA processing: 10-30 sec per protein
- Forward pass: 30-60 sec per protein
- Saving activations: 5-10 sec per protein

**Optimization:**
```bash
# Process multiple proteins in single run (amortizes model loading)
--max-proteins 100  # instead of running 100 times with --max-proteins 1
```

#### High GPU Memory (8GB+)

**Causes:**
- Large protein (>500 residues) → large pair representation (N²)
- Multiple layers hooked → multiple activation buffers
- Gradient accumulation from previous runs

**Solutions:**
```bash
# 1. Process smaller proteins
--max-proteins 10

# 2. Collect fewer layers per run
--layers 0 8 16  # Split into two runs: [0,8,16] and [24,32,40]

# 3. Clear CUDA cache between runs
python -c "import torch; torch.cuda.empty_cache()"
```

#### Negative R² Scores

**Meaning:** Model performs worse than predicting mean value

**Causes:**
1. **Insufficient training:** 10 steps is too few
2. **Insufficient data:** 1 protein doesn't capture distribution
3. **Wrong targets:** y1/y2 mismatch with s1/s2

**Solutions:**
```bash
# Increase training steps
--num_steps 500

# Collect more data
--max-proteins 50

# Check if loss is decreasing (if yes, just needs more steps)
tail training_log.txt
```

---

## Key Considerations

### 1. Layer Selection

**Why layers 0, 8, 16, 24, 32, 40?**

- **Layer 0:** Early features (sequence embeddings, local patterns)
- **Layer 8:** Basic structural motifs
- **Layer 16:** Secondary structure consolidation
- **Layer 24:** Tertiary interactions start
- **Layer 32:** Complex fold patterns
- **Layer 40:** Near-final representations
- **Layer 47:** Final output features (could add for completeness)

**Alternative strategies:**
- **All layers:** `--layers $(seq 0 47)` (48 transcoders, expensive)
- **Powers of 2:** `--layers 0 1 2 4 8 16 32 47` (logarithmic sampling)
- **Last 10 only:** `--layers $(seq 38 47)` (focus on deep features)

### 2. Independent vs Shared Training

**Current approach:** Independent Per-Layer Transcoders (PLTs) per layer

**Advantages:**
- Each layer learns its own feature dictionary
- Easy to compare layer complexity
- Parallelizable (different GPUs)

**Alternative:** Shared encoder across layers

**Advantages:**
- Learns layer-invariant features
- Reduces total parameters
- Can compare layer-specific decoder weights

**Why independent is better:**
Different layers have fundamentally different feature spaces:
- Layer 0: Amino acid identity, charge, hydrophobicity
- Layer 40: Long-range contacts, domain interfaces, binding sites

Shared encoder would be forced to compromise.

### 3. Data Requirements

**Minimal viable:**
- 1 protein, 100 steps → R² = 0.1-0.3 (proof of concept)

**Reasonable:**
- 10 proteins, 500 steps → R² = 0.4-0.6 (usable features)

**Production:**
- 100+ proteins, 5000 steps → R² = 0.7-0.9 (reliable features)

**Scaling law (approximate):**
```
R² ≈ 0.1 + 0.4 * log10(num_proteins) + 0.3 * log10(num_steps / 100)
```

Example:
- 1 protein, 10 steps: R² ≈ 0.1 + 0 - 0.3 = -0.2 (bad)
- 10 proteins, 100 steps: R² ≈ 0.1 + 0.4 + 0 = 0.5 (okay)
- 100 proteins, 1000 steps: R² ≈ 0.1 + 0.8 + 0.3 = 1.2 → clipped to ~0.85 (good)

### 4. Sparsity and Interpretability

**k=16 means:**
- Only 16 out of 2048 features active per forward pass
- Sparsity = 16/2048 = 0.78% (99.22% zeros)

**Trade-off:**
- **Lower k** (e.g., k=8): Sparser features, easier interpretation, lower R²
- **Higher k** (e.g., k=32): Better reconstruction, harder interpretation

**Dead neurons:**

If training shows `Dead neurons: 1500 / 2048`, it means:
- 1500 features NEVER activated during training
- Only 548 features actually learned
- Effective sparsity = 16/548 = 2.9%

**AuxK resurrection:**
- When `dead_neurons > 1000`, model activates auxiliary features
- Gives dead neurons gradient signal to "come alive"
- `auxk=32` means try activating 32 dead neurons per batch

### 5. Consistency Loss Intuition

**Why predict y1 and y2 from both s1 and s2?**

```
s1 (input_s)  ──┐                    ┌──→ y1 (input_z)
                │ Should contain      │
                │ same information    │
s2 (output_s) ──┘                    └──→ y2 (output_z)
```

Transition MLP is supposed to preserve information:
- `s2 = transition_s(s1)` (information-preserving transform)
- Both s1 and s2 should predict same y1, y2

**Consistency loss** forces this:
```python
# Predictions from s1
y1_from_s1, y2_from_s1 = transcoder(s1)

# Predictions from s2
y1_from_s2, y2_from_s2 = transcoder(s2)

# Should agree
loss_consistency = MSE(y1_from_s1, y1_from_s2) + MSE(y2_from_s1, y2_from_s2)
```

This prevents overfitting to input_s vs output_s artifacts.

### 6. Validation Metrics Interpretation

**R² interpretation by value:**

| R² Range | Quality | Recommendation |
|----------|---------|----------------|
| < 0 | Broken | Check data loading, increase training |
| 0.0 - 0.3 | Poor | Need more data or longer training |
| 0.3 - 0.5 | Acceptable | Minimal features, increase steps |
| 0.5 - 0.7 | Good | Usable for analysis |
| 0.7 - 0.9 | Excellent | Production quality |
| > 0.9 | Suspicious | Possible overfitting or data leak |

**RMSE interpretation:**

Depends on data scale. For Boltz2 pair representations:
- Typical value range: [-50, 50]
- RMSE = 10: Reasonable error
- RMSE = 50: Large error (predicting near noise)

**Sparsity verification:**

Should equal `k` parameter:
```python
assert avg_sparsity == model.k  # Should be exactly 16
```

If different, indicates implementation bug.

---

## Related Documentation

### Essential Reading

1. **PLT Architecture:**
   - [PLT_ARCHITECTURE_GUIDE.md](PLT_ARCHITECTURE_GUIDE.md) (if exists)
   - Explains TopK activation, unit norm decoders, AuxK
   - PLT = Per-Layer Transcoder

2. **Original Transcoder:**
   - [TRANSCODER_PROJECT_SUMMARY.md](TRANSCODER_PROJECT_SUMMARY.md) (if exists)
   - Single-layer Per-Layer Transcoder for layer 47 only

3. **Activation Collection:**
   - [DATA_LOADING_GUIDE.md](../transcoder/DATA_LOADING_GUIDE.md)
   - Explains NPZ format, data loading patterns

4. **Boltz2 Architecture:**
   - [BOLTZ_ARCHITECTURE_TECHNICAL_DOCUMENTATION.md](../BOLTZ_ARCHITECTURE_TECHNICAL_DOCUMENTATION.md)
   - Pairformer layers, transition modules, structure

### Code Files to Review

**Priority 1 (Must understand):**

1. **`universal_transcoder/universal_model.py`** (217 lines)
   - UniversalTranscoder class
   - TopK activation
   - Dual-decoder architecture

2. **`collection_scripts/collect_multi_layer.py`** (443 lines)
   - LayerActivationCollector
   - MultiLayerActivationCollector
   - FASTA parsing and forward pass

3. **`universal_transcoder/train_multi_layer.py`** (336 lines)
   - Training orchestration
   - train_single_layer()
   - Summary generation

**Priority 2 (Important):**

4. **`universal_transcoder/validate_multi_layer.py`** (410 lines)
   - Validation metrics
   - R² computation
   - Weight norm checking

5. **`universal_transcoder/train_universal.py`** (404 lines)
   - Dual-pass training loop
   - Combined loss computation
   - AuxK implementation

**Priority 3 (Reference):**

6. **`run_multi_layer_pipeline.sh`** (100 lines)
   - End-to-end pipeline script
   - Error checking
   - Log management

7. **`test_minimal_pipeline.sh`** (144 lines)
   - Minimal test example
   - Good template for custom scripts

---

## FAQ

**Q: What does PLT stand for?**

A: Per-Layer Transcoder - each layer gets its own independent sparse autoencoder.

**Q: Why not train one big transcoder for all layers?**

A: Different layers have different feature spaces. Forcing shared weights would hurt all layers. Independent models allow layer-specific optimization.

**Q: Can I train transcoders on CPU?**

A: Yes, but very slow (10-100x slower). Collection requires GPU (Boltz2 is too large for CPU inference).

**Q: How do I know if training worked?**

A: Check three things:
1. Loss decreasing: `tail training_log.txt`
2. R² > 0.3: `cat validation_summary.json`
3. Dead neurons < 500: `grep "dead_neurons" training_metrics.json`

**Q: Can I use different k for different layers?**

A: Yes, modify train_multi_layer.py:
```python
k_values = {0: 8, 8: 12, 16: 16, 24: 20, 32: 24, 40: 32}
k = k_values[layer_idx]
```

Early layers (simpler features) might need lower k.

**Q: What does PLT stand for?**

A: **Per-Layer Transcoder**. Each pairformer layer gets its own independent sparse autoencoder (transcoder). This is different from "piecewise linear" - the name reflects that we train one transcoder per layer.

**Q: What if validation_summary.json shows negative R²?**

A: Train longer. Quick fix:
```bash
cd universal_transcoder
python train_multi_layer.py \
    --data_dir ../multi_layer_activations \
    --checkpoint_dir ../multi_layer_checkpoints \
    --num_steps 500  # ← Increase from 100
```

**Q: Can I add layer 47 to the default set?**

A: Yes:
```bash
./run_multi_layer_pipeline.sh
# Edit line 14: LAYERS="0 8 16 24 32 40 47"
```

Or command line:
```bash
python collect_multi_layer.py --layers 0 8 16 24 32 40 47 ...
```

**Q: How do I visualize features?**

A: Load decoder weights and sort by norm:
```python
import torch
ckpt = torch.load('multi_layer_checkpoints/layer_40/universal_transcoder_final.pt')
decoder = ckpt['model_state_dict']['decoder_y1']  # [2048, 128]

# Sort features by L2 norm
norms = decoder.norm(dim=1)
top_features = norms.argsort(descending=True)[:10]

# Visualize top 10 features
import matplotlib.pyplot as plt
plt.matshow(decoder[top_features].T.cpu())
plt.colorbar()
plt.title('Top 10 Features (Layer 40)')
plt.show()
```

---

## Appendix: Mathematical Formulation

### Forward Pass

```
Input: s ∈ ℝ^(N×384)  (N = number of residues)

1. Flatten: s_flat ∈ ℝ^(B·N×384)

2. Normalize:
   s_norm = (s_flat - μ) / σ
   s_centered = s_norm - b_pre

3. Encode:
   pre_acts = W_enc · s_centered + b_enc  ∈ ℝ^(B·N×2048)

4. TopK Activation:
   latents = TopK(pre_acts, k=16)  ∈ ℝ^(B·N×2048), only 16 non-zero per row

5. Decode:
   y1_recon = latents · W_dec1 + b_y1  ∈ ℝ^(B·N×128)
   y2_recon = latents · W_dec2 + b_y2  ∈ ℝ^(B·N×128)

6. Denormalize:
   y1_final = y1_recon · σ + μ
   y2_final = y2_recon · σ + μ
```

### Loss Function

```
Given:
  s1, s2 ∈ ℝ^(B·N×384)  (input and output of transition_s)
  y1, y2 ∈ ℝ^(B·N²×128)  (input and output of transition_z, flattened)

Forward passes:
  ŷ1_from_s1, ŷ2_from_s1 = Transcoder(s1)
  ŷ1_from_s2, ŷ2_from_s2 = Transcoder(s2)

Expand to pair dimensions:
  ŷ1_from_s1_exp = repeat(ŷ1_from_s1, N) ∈ ℝ^(B·N²×128)

Reconstruction loss:
  ℒ_recon = ||ŷ1_from_s1_exp - y1||² + ||ŷ2_from_s1_exp - y2||²
          + ||ŷ1_from_s2_exp - y1||² + ||ŷ2_from_s2_exp - y2||²

Consistency loss:
  ℒ_cons = ||ŷ1_from_s1 - ŷ1_from_s2||² + ||ŷ2_from_s1 - ŷ2_from_s2||²

AuxK loss (if dead neurons exist):
  ℒ_auxk = (1/32) · (||aux_ŷ1 - residual_y1||² + ||aux_ŷ2 - residual_y2||²)

Total loss:
  ℒ = ℒ_recon + ℒ_cons + ℒ_auxk
```

### Weight Constraints

```
Decoder weights:
  W_dec1 ∈ ℝ^(2048×128)
  W_dec2 ∈ ℝ^(2048×128)

Constraint (enforced after each step):
  ||W_dec1[:,i]||₂ = 1  ∀i ∈ [0, 2047]  (unit norm per feature)
  ||W_dec2[:,i]||₂ = 1  ∀i ∈ [0, 2047]
```

### Sparsity

```
K = 16 (TopK parameter)
D = 2048 (latent dimension)

Sparsity = K/D = 16/2048 ≈ 0.78%

For a batch of size B·N:
  Total elements: B·N·D
  Non-zero elements: B·N·K
  Activation ratio: K/D
```

---

## Version History

- **v1.0** (March 9, 2026): Initial release
  - Multi-layer collection
  - Independent PLT training
  - Validation suite
  - Minimal test pipeline

---

**For questions or issues, see [TROUBLESHOOTING](#troubleshooting) or check existing documentation in `/documentation/`.**
