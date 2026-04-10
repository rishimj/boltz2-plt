# Multi-Layer PLT Pipeline: Visual Flow & Architecture

**Companion to:** [MULTI_LAYER_PLT_GUIDE.md](MULTI_LAYER_PLT_GUIDE.md)

**Terminology:** PLT = **Per-Layer Transcoder** (independent sparse autoencoder per layer)

---

## System Overview

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                         MULTI-LAYER PLT PIPELINE                          ║
║                                                                           ║
║  Input: FASTA → Collection → Training → Validation → Analysis            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## Detailed Data Flow

### Phase 1: Activation Collection

```
┌──────────────────────────────────────────────────────────────────────────┐
│ INPUT: examples/prot.fasta                                               │
│                                                                          │
│ >protein1                                                                │
│ MKTIIALSYIFCLVFADYKDDDDK...                                             │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
                   ┌────────────────────────────────┐
                   │    parse_fasta()               │
                   │  (with boltz2=True)            │
                   └────────────────────────────────┘
                                    ↓
                    ┌───────────────────────────────┐
                    │   Boltz2Tokenizer            │
                    │   (sequence → token IDs)     │
                    └───────────────────────────────┘
                                    ↓
                    ┌───────────────────────────────┐
                    │   Boltz2Featurizer           │
                    │   (add MSA, structure info)  │
                    └───────────────────────────────┘
                                    ↓
╔═══════════════════════════════════════════════════════════════════════════╗
║                         BOLTZ2 FORWARD PASS                               ║
║                                                                           ║
║  Input Embedding → Pairformer (48 layers) → Structure Module             ║
║                                                                           ║
║  Layer  0: ──┬── [HOOK s] ──→ transition_s ──→ [HOOK s] ────────┐       ║
║              └── [HOOK z] ──→ transition_z ──→ [HOOK z] ────┐   │       ║
║                                                              │   │       ║
║  Layer  8: ──┬── [HOOK s] ──→ transition_s ──→ [HOOK s] ────┼───┼───┐  ║
║              └── [HOOK z] ──→ transition_z ──→ [HOOK z] ────┼─┐ │   │  ║
║                                                              │ │ │   │  ║
║  Layer 16: ──┬── [HOOK s] ──→ transition_s ──→ [HOOK s] ────┼─┼─┼───┼─┐║
║              └── [HOOK z] ──→ transition_z ──→ [HOOK z] ────┼─┼─┼┐  │ │║
║  ...         ...                                            │ │ ││  │ │║
║  Layer 40: ──┬── [HOOK s] ──→ transition_s ──→ [HOOK s] ────┼─┼─┼┼──┼─┼║
║              └── [HOOK z] ──→ transition_z ──→ [HOOK z] ────┼─┼─┼┼─┐│ │║
║                                                              │ │ ││ ││ │║
╚══════════════════════════════════════════════════════════════╪═╪═╪╪═╪╪═╪╝
                                                               │ │ ││ ││ │
                        LayerActivationCollector instances:   │ │ ││ ││ │
                                                               ↓ ↓ ↓↓ ↓↓ ↓
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────────┐
│  Collector #0       │ │  Collector #8       │ │  Collector #40          │
│                     │ │                     │ │                         │
│  activations:       │ │  activations:       │ │  activations:           │
│  ├─ input_s  [list] │ │  ├─ input_s  [list] │ │  ├─ input_s  [list]     │
│  ├─ output_s [list] │ │  ├─ output_s [list] │ │  ├─ output_s [list]     │
│  ├─ input_z  [list] │ │  ├─ input_z  [list] │ │  ├─ input_z  [list]     │
│  └─ output_z [list] │ │  └─ output_z [list] │ │  └─ output_z [list]     │
└─────────────────────┘ └─────────────────────┘ └─────────────────────────┘
         ↓                      ↓                          ↓
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────────┐
│ save_batch()        │ │ save_batch()        │ │ save_batch()            │
│                     │ │                     │ │                         │
│ Concatenate lists   │ │ Concatenate lists   │ │ Concatenate lists       │
│ Flatten pair dim    │ │ Flatten pair dim    │ │ Flatten pair dim        │
│ Save to NPZ         │ │ Save to NPZ         │ │ Save to NPZ             │
└─────────────────────┘ └─────────────────────┘ └─────────────────────────┘
         ↓                      ↓                          ↓
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────────┐
│ layer_00/           │ │ layer_08/           │ │ layer_40/               │
│  batch_00000.npz    │ │  batch_00000.npz    │ │  batch_00000.npz        │
│  batch_00001.npz    │ │  batch_00001.npz    │ │  batch_00001.npz        │
│  ...                │ │  ...                │ │  ...                    │
└─────────────────────┘ └─────────────────────┘ └─────────────────────────┘
```

### NPZ File Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│  File: layer_XX/batch_00000.npz                                     │
│                                                                     │
│  Keys:                                                              │
│    'input_s'   → ndarray [B, N, 384]      (single rep input)       │
│    'output_s'  → ndarray [B, N, 384]      (single rep output)      │
│    'input_z'   → ndarray [B, N², 128]     (pair rep input)         │
│    'output_z'  → ndarray [B, N², 128]     (pair rep output)        │
│                                                                     │
│  Example (protein with 117 residues):                              │
│    input_s:  (1, 117, 384)    ← per-residue single representation  │
│    output_s: (1, 117, 384)                                         │
│    input_z:  (1, 13689, 128)  ← 13689 = 117² (all residue pairs)   │
│    output_z: (1, 13689, 128)                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 2: Training Architecture

### Sequential Training Flow

```
train_multi_layer.py
         │
         ├──→ For layer in [0, 8, 16, 24, 32, 40]:
         │         │
         │         ├── Check if data exists (layer_XX/*.npz)
         │         │
         │         ├── train_single_layer(layer_idx)
         │         │         │
         │         │         └──→ ┌──────────────────────────────────┐
         │         │              │  train_universal_transcoder()    │
         │         │              │                                  │
         │         │              │  1. Load data from NPZ          │
         │         │              │  2. Create model                │
         │         │              │  3. Training loop (100 steps)   │
         │         │              │  4. Save checkpoint             │
         │         │              └──────────────────────────────────┘
         │         │                          │
         │         │                          ↓
         │         │              ┌──────────────────────────────────┐
         │         │              │  Returns:                        │
         │         │              │  - trained model                 │
         │         │              │  - metrics history               │
         │         │              │  - training time                 │
         │         │              └──────────────────────────────────┘
         │         │
         │         ├── Extract final metrics
         │         ├── Save to results list
         │         └── Continue to next layer
         │
         └──→ Generate summary JSON
```

### Single Layer Training Detail

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                      TRAINING ONE TRANSCODER (e.g., Layer 40)             ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌────────────────────────────────────────────────────────────────────────┐
│  1. DATA LOADING                                                       │
│                                                                        │
│  UniversalActivationDataset(layer_40/batch_*.npz)                     │
│                                                                        │
│  Loads 4 arrays per sample:                                           │
│    s1 = input_s   [N, 384]    ← Input to transition_s                │
│    s2 = output_s  [N, 384]    ← Output from transition_s             │
│    y1 = input_z   [N², 128]   ← Input to transition_z                │
│    y2 = output_z  [N², 128]   ← Output from transition_z             │
│                                                                        │
│  Creates DataLoader with batch_size=10                                │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────────────────────────────┐
│  2. MODEL INITIALIZATION                                               │
│                                                                        │
│  UniversalTranscoder(                                                  │
│    d_model  = 384,    ← Input dimension (from s1/s2)                  │
│    d_hidden = 2048,   ← Latent dimension (feature dictionary size)    │
│    d_pair   = 128,    ← Output dimension (to y1/y2)                   │
│    k        = 16,     ← TopK sparsity                                 │
│    auxk     = 32      ← Auxiliary K for dead neuron resurrection      │
│  )                                                                     │
│                                                                        │
│  Architecture:                                                         │
│                                                                        │
│    ┌────────────────────────────────────────────────────────────┐    │
│    │  Encoder:  Linear(384 → 2048) + bias                       │    │
│    │            + pre-bias, LayerNorm params                    │    │
│    └────────────────────────────────────────────────────────────┘    │
│                          ↓                                            │
│    ┌────────────────────────────────────────────────────────────┐    │
│    │  TopK Activation: Keep only k=16 largest values per row    │    │
│    └────────────────────────────────────────────────────────────┘    │
│                          ↓                                            │
│    ┌────────────────────────────────────────────────────────────┐    │
│    │  Decoder Y1: Parameter(2048 → 128) + bias                  │    │
│    │  Decoder Y2: Parameter(2048 → 128) + bias                  │    │
│    │  (Unit norm constraint enforced after each step)           │    │
│    └────────────────────────────────────────────────────────────┘    │
│                                                                        │
│  Total parameters: ~1.6M                                               │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────────────────────────────┐
│  3. TRAINING LOOP (100 steps)                                          │
│                                                                        │
│  for step in range(100):                                              │
│      for batch in dataloader:                                         │
│          # Unpack batch                                               │
│          s1, s2, y1, y2 = batch  # All on GPU                         │
│                                                                        │
│          # === DUAL-PASS FORWARD ===                                  │
│          y1_pred1, y2_pred1, aux_y1_1, aux_y2_1, dead_mask = model(s1)│
│          y1_pred2, y2_pred2, aux_y1_2, aux_y2_2, _ = model(s2)        │
│          #                ↑                                            │
│          #   Two forward passes through SAME model                    │
│                                                                        │
│          # === EXPANSION ===                                          │
│          # Expand [B*N, 128] → [B*N², 128] to match pair dimension    │
│          y1_pred1_exp = expand_to_pair_dim(y1_pred1)                  │
│          y1_pred2_exp = expand_to_pair_dim(y1_pred2)                  │
│          # (same for y2_pred)                                         │
│                                                                        │
│          # === LOSS COMPUTATION ===                                   │
│          loss_recon = (                                               │
│              MSE(y1_pred1_exp, y1) +  # Predict y1 from s1           │
│              MSE(y2_pred1_exp, y2) +  # Predict y2 from s1           │
│              MSE(y1_pred2_exp, y1) +  # Predict y1 from s2           │
│              MSE(y2_pred2_exp, y2)    # Predict y2 from s2           │
│          )                                                            │
│                                                                        │
│          loss_consistency = (                                         │
│              MSE(y1_pred1, y1_pred2) +  # s1 and s2 agree on y1      │
│              MSE(y2_pred1, y2_pred2)    # s1 and s2 agree on y2      │
│          )                                                            │
│                                                                        │
│          # If many dead neurons, add AuxK loss                        │
│          if num_dead > threshold:                                     │
│              loss_auxk = MSE(aux_y1, residual_y1) + ...              │
│          else:                                                        │
│              loss_auxk = 0                                            │
│                                                                        │
│          total_loss = loss_recon + loss_consistency + loss_auxk       │
│                                                                        │
│          # === OPTIMIZATION ===                                       │
│          optimizer.zero_grad()                                        │
│          total_loss.backward()                                        │
│          optimizer.step()                                             │
│                                                                        │
│          # === ENFORCE CONSTRAINTS ===                                │
│          model.norm_weights()  # Unit norm for decoders               │
│                                                                        │
│      # Log metrics every 10 steps                                     │
│      if step % 10 == 0:                                               │
│          print(f"Step {step}: loss={total_loss:.2f}")                 │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────────────────────────────┐
│  4. CHECKPOINT SAVING                                                  │
│                                                                        │
│  Save to: layer_40/universal_transcoder_final.pt                      │
│                                                                        │
│  Contents:                                                             │
│    - model_state_dict: {encoder.weight, decoder_y1, decoder_y2, ...}  │
│    - optimizer_state_dict: {param_groups, state, ...}                 │
│    - hyperparameters: {d_model, d_hidden, k, auxk, lr, ...}           │
│    - final_metrics: {loss_total, loss_recon, dead_neurons, ...}       │
│    - metrics_history: [{step: 0, loss: ...}, {step: 1, ...}, ...]     │
│    - training_time: 2.06 seconds                                      │
│                                                                        │
│  Also saves:                                                           │
│    - training_metrics.json (detailed metrics)                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 3: Validation Pipeline

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                         VALIDATION ARCHITECTURE                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

validate_multi_layer.py --layers 0 8 16 24 32 40
         │
         ├──→ For each layer:
         │         │
         │         ├── 1. LOAD MODEL
         │         │       │
         │         │       ├── Read checkpoint
         │         │       ├── Extract hyperparameters
         │         │       ├── Reconstruct UniversalTranscoder
         │         │       └── Load state_dict, set eval()
         │         │
         │         ├── 2. VERIFY WEIGHTS
         │         │       │
         │         │       ├── Check encoder norms
         │         │       │   Expected: ~1.2 (learned during training)
         │         │       │
         │         │       └── Check decoder norms
         │         │           Expected: exactly 1.0 (enforced constraint)
         │         │
         │         ├── 3. LOAD VALIDATION DATA
         │         │       │
         │         │       └── Load all layer_XX/batch_*.npz files
         │         │
         │         ├── 4. COMPUTE METRICS
         │         │       │
         │         │       └──→ ┌──────────────────────────────────────┐
         │         │            │  For each batch:                     │
         │         │            │                                      │
         │         │            │  # Forward pass                      │
         │         │            │  s1 = batch['input_s']               │
         │         │            │  y1_true = batch['input_z']          │
         │         │            │  y1_pred, y2_pred, _, _, _ = model(s1)│
         │         │            │                                      │
         │         │            │  # Expand predictions                │
         │         │            │  y1_pred_exp = expand(y1_pred)       │
         │         │            │                                      │
         │         │            │  # Compute metrics                   │
         │         │            │  MSE_y1 = mean((y1_pred - y1_true)²) │
         │         │            │  RMSE_y1 = √MSE_y1                   │
         │         │            │  R²_y1 = 1 - (SS_res / SS_tot)       │
         │         │            │                                      │
         │         │            │  # Aggregate                         │
         │         │            │  all_mse.append(MSE_y1)              │
         │         │            │  all_r2.append(R²_y1)                │
         │         │            └──────────────────────────────────────┘
         │         │                          │
         │         │                          ↓
         │         │       ┌──────────────────────────────────────┐
         │         │       │  Average metrics across batches:     │
         │         │       │                                      │
         │         │       │  avg_mse_y1  = mean(all_mse_y1)      │
         │         │       │  avg_r2_y1   = mean(all_r2_y1)       │
         │         │       │  avg_sparsity = k (constant: 16)     │
         │         │       └──────────────────────────────────────┘
         │         │
         │         ├── 5. SAVE RESULTS
         │         │       │
         │         │       └── layer_40/validation_summary.json
         │         │
         │         └── Continue to next layer
         │
         └──→ Generate multi-layer summary
```

### R² Computation Details

```
┌─────────────────────────────────────────────────────────────────────────┐
│  R² (Coefficient of Determination) Calculation                          │
│                                                                         │
│  Given:                                                                 │
│    y_true: Ground truth values [B*N², 128]                             │
│    y_pred: Model predictions  [B*N², 128]                              │
│                                                                         │
│  Compute:                                                               │
│                                                                         │
│    SS_res = Σ(y_true - y_pred)²     ← Residual sum of squares         │
│    SS_tot = Σ(y_true - mean(y_true))² ← Total sum of squares          │
│                                                                         │
│    R² = 1 - (SS_res / SS_tot)                                          │
│                                                                         │
│  Interpretation:                                                        │
│    R² = 1.0   Perfect reconstruction                                   │
│    R² = 0.5   Explains 50% of variance                                 │
│    R² = 0.0   No better than predicting mean                           │
│    R² < 0     Worse than predicting mean (undertraining)               │
│                                                                         │
│  Example:                                                               │
│    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]                                  │
│    y_pred = [1.1, 2.2, 2.9, 3.8, 5.1]                                  │
│                                                                         │
│    mean(y_true) = 3.0                                                  │
│                                                                         │
│    SS_res = (1.0-1.1)² + (2.0-2.2)² + ... = 0.27                       │
│    SS_tot = (1.0-3.0)² + (2.0-3.0)² + ... = 10.0                       │
│                                                                         │
│    R² = 1 - (0.27 / 10.0) = 0.973  ← Excellent fit                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            FILE DEPENDENCIES                             │
└──────────────────────────────────────────────────────────────────────────┘

collection_scripts/collect_multi_layer.py
    │
    ├── Imports:
    │   ├── torch
    │   ├── boltz.model.Boltz2
    │   ├── boltz.data.tokenize.Boltz2Tokenizer
    │   ├── boltz.data.feature.Boltz2Featurizer
    │   ├── boltz.data.parse.fasta.parse_fasta  ← CRITICAL: needs 4 args
    │   └── boltz.data.parse.ccd.load_canonicals
    │
    └── Creates:
        └── layer_XX/batch_XXXXX.npz (NPZ files)

universal_transcoder/train_multi_layer.py
    │
    ├── Imports:
    │   ├── universal_transcoder/train_universal.py  ← Main training logic
    │   └── universal_transcoder/universal_model.py  ← Model definition
    │
    ├── Reads:
    │   └── layer_XX/batch_*.npz (from collection)
    │
    └── Creates:
        └── layer_XX/
            ├── universal_transcoder_final.pt  ← Model checkpoint
            └── training_metrics.json          ← Metrics history

universal_transcoder/validate_multi_layer.py
    │
    ├── Imports:
    │   └── universal_transcoder/universal_model.py
    │
    ├── Reads:
    │   ├── layer_XX/universal_transcoder_final.pt  (from training)
    │   └── layer_XX/batch_*.npz                     (from collection)
    │
    └── Creates:
        └── layer_XX/validation_summary.json

run_multi_layer_pipeline.sh
    │
    ├── Calls:
    │   ├── 1. collection_scripts/collect_multi_layer.py
    │   ├── 2. universal_transcoder/train_multi_layer.py
    │   └── 3. universal_transcoder/validate_multi_layer.py
    │
    └── Creates:
        └── multi_layer_logs/
            ├── 01_collection.log
            ├── 02_training.log
            └── 03_validation.log
```

---

## Memory and Computation Flow

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        RESOURCE USAGE BREAKDOWN                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

COLLECTION PHASE:
┌─────────────────────────────────────────────────────────────────────────┐
│  GPU Memory:                                                            │
│    ├── Boltz2 Model:        ~2.0 GB  (static)                          │
│    ├── Input Features:      ~0.5 GB  (protein size × 50MB)             │
│    ├── Forward Pass Tensors: ~1.0 GB  (temporary)                      │
│    └── Hook Buffers (6 layers): ~0.3 GB × 6 = 1.8 GB                   │
│                                                                         │
│  Total GPU: ~5.3 GB (peak)                                              │
│                                                                         │
│  CPU Memory:                                                            │
│    └── Accumulated activations: ~50 MB per protein per layer           │
│        → 6 layers × 10 proteins × 50 MB = 3 GB                          │
│                                                                         │
│  Disk:                                                                  │
│    └── NPZ files: ~3 GB (compressed)                                    │
│                                                                         │
│  Time:                                                                  │
│    ├── Model loading (one-time): 3-5 minutes                           │
│    ├── Per-protein processing: 30-60 seconds                           │
│    └── Total for 10 proteins: ~10-15 minutes                           │
└─────────────────────────────────────────────────────────────────────────┘

TRAINING PHASE (per layer):
┌─────────────────────────────────────────────────────────────────────────┐
│  GPU Memory:                                                            │
│    ├── Model Parameters:     ~6.5 MB                                   │
│    ├── Batch Data (batch_size=10):                                     │
│    │   ├── s1: 10 × 384 × N    ~100 KB                                 │
│    │   ├── s2: 10 × 384 × N    ~100 KB                                 │
│    │   ├── y1: 10 × N² × 128   ~5 MB (for N=117)                       │
│    │   └── y2: 10 × N² × 128   ~5 MB                                   │
│    ├── Forward Pass Activations: ~50 MB                                │
│    ├── Gradients:            ~6.5 MB                                   │
│    └── Optimizer State:      ~13 MB (Adam momentum + variance)         │
│                                                                         │
│  Total GPU: ~100 MB per layer                                           │
│                                                                         │
│  Disk:                                                                  │
│    ├── Checkpoint: ~20 MB per layer                                    │
│    └── Metrics JSON: ~10 KB per layer                                  │
│                                                                         │
│  Time (100 steps):                                                      │
│    ├── Data loading: ~1 second                                         │
│    ├── Training loop: 0.2 sec/step × 100 = 20 seconds                  │
│    ├── Checkpoint save: ~1 second                                      │
│    └── Total per layer: ~22 seconds                                    │
│                                                                         │
│  6 Layers Sequential: ~2.2 minutes                                      │
└─────────────────────────────────────────────────────────────────────────┘

VALIDATION PHASE:
┌─────────────────────────────────────────────────────────────────────────┐
│  GPU Memory:                                                            │
│    ├── Model: ~6.5 MB                                                  │
│    └── Data batch: ~10 MB                                              │
│                                                                         │
│  Total GPU: ~20 MB per layer                                            │
│                                                                         │
│  Time: ~5 seconds per layer (6 layers = 30 seconds)                     │
└─────────────────────────────────────────────────────────────────────────┘

TOTAL PIPELINE (10 proteins, 6 layers, 100 steps):
┌─────────────────────────────────────────────────────────────────────────┐
│  Time:  Collection (15 min) + Training (2.2 min) + Validation (0.5 min) │
│       = ~18 minutes                                                      │
│                                                                         │
│  Disk:  Activations (3 GB) + Checkpoints (120 MB) + Logs (1 MB)         │
│       = ~3.2 GB                                                          │
│                                                                         │
│  Peak GPU Memory: ~5.3 GB (collection phase)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Error Propagation & Debugging Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     COMMON ERROR PATHS & SOLUTIONS                       │
└──────────────────────────────────────────────────────────────────────────┘

ERROR 1: Collection fails with "Invalid FASTA path"
    │
    ├── File: collect_multi_layer.py:271
    ├── Cause: Relative path not resolved
    │
    └── Flow:
            Script run from collection_scripts/
                 ↓
            Relative path "../examples/prot.fasta"
                 ↓
            Working dir ≠ expected dir
                 ↓
            Path.exists() returns False
                 ↓
            ValueError raised
                 
            FIX: Path(fasta_path).resolve()  ← Convert to absolute

ERROR 2: Collection fails with "parse_fasta() missing arguments"
    │
    ├── File: collect_multi_layer.py:305
    ├── Cause: Boltz2 requires 4 arguments, not 1
    │
    └── Flow:
            parse_fasta(fasta_file)  ← OLD, wrong signature
                 ↓
            Boltz2's parse_fasta expects:
              (path, molecules: CCD, moldir: Path, boltz2: bool)
                 ↓
            TypeError: missing 2 required positional arguments
                 
            FIX: parse_fasta(fasta_file, molecules, moldir, boltz2=True)

ERROR 3: Validation fails with "AttributeError: 'Parameter' object..."
    │
    ├── File: validate_multi_layer.py:115
    ├── Cause: decoder_y1 IS the Parameter, not a module with .weight
    │
    └── Flow:
            model.decoder_y1.weight  ← Assumes nn.Linear
                 ↓
            But decoder_y1 = nn.Parameter([2048, 128])
                 ↓
            Parameter has no .weight attribute
                 ↓
            AttributeError raised
                 
            FIX: torch.norm(model.decoder_y1, dim=0)  ← Use Parameter directly

ERROR 4: Validation fails with "IndexError: Dimension out of range"
    │
    ├── File: validate_multi_layer.py:206
    ├── Cause: active_mask is [D] shaped, not [B, D]
    │
    └── Flow:
            active_mask = ~dead_mask  ← [2048] boolean array
                 ↓
            active_mask.sum(dim=1)  ← Try to sum over dim 1
                 ↓
            But active_mask only has dim 0!
                 ↓
            IndexError: expected range [-1, 0], got 1
                 
            FIX: sparsity = model.k  ← TopK guarantees exactly k active

ERROR 5: Training produces negative R²
    │
    ├── File: Training output, visible in validation
    ├── Cause: Insufficient training (too few steps or data)
    │
    └── Flow:
            Model trained for 10 steps
                 ↓
            Weights barely updated from initialization
                 ↓
            Predictions ~ random
                 ↓
            SS_res > SS_tot
                 ↓
            R² = 1 - (SS_res / SS_tot) < 0
                 
            FIX: Increase --num_steps to 500+ and/or --max_proteins to 50+
```

---

## Feature Interpretation Workflow

```
┌──────────────────────────────────────────────────────────────────────────┐
│               USING TRAINED TRANSCODERS FOR ANALYSIS                     │
└──────────────────────────────────────────────────────────────────────────┘

After training, you can analyze learned features:

1. LOAD TRAINED MODEL
   │
   ├── import torch
   ├── from universal_transcoder.universal_model import UniversalTranscoder
   │
   └── checkpoint = torch.load('layer_40/universal_transcoder_final.pt')
       model = UniversalTranscoder(**checkpoint['hyperparameters'])
       model.load_state_dict(checkpoint['model_state_dict'])
       model.eval()

2. EXTRACT FEATURE DICTIONARY
   │
   ├── encoder = model.encoder.weight  # [2048, 384]
   ├── decoder_y1 = model.decoder_y1   # [2048, 128]
   └── decoder_y2 = model.decoder_y2   # [2048, 128]
       
       Each column of decoder = one learned feature
       Feature i: decoder_y1[i, :] is a 128-dim vector

3. ANALYZE FEATURE ACTIVATIONS
   │
   └── # Forward pass on new protein
       s1 = ...  # New protein activations [N, 384]
       y1_pred, y2_pred, _, _, _ = model(s1)
       
       # Get latent activations (after TopK)
       pre_acts = model.encoder(normalize(s1))
       latents = topK_activation(pre_acts, k=16)  # [N, 2048]
       
       # Find which features are active
       active_features = (latents > 0).nonzero()  # [N*16, 2]
       
       # For each residue, see which 16 features activated

4. FEATURE VISUALIZATION
   │
   ├── Plot decoder weights as heatmaps
   ├── Cluster features by similarity
   ├── Correlate with structure (DSSP, contacts)
   └── Compare across layers (early vs late features)

5. INTERVENTION EXPERIMENTS
   │
   ├── Amplify specific features: latents[:, feature_id] *= 2.0
   ├── Suppress features: latents[:, feature_id] = 0.0
   ├── Decode modified latents: y_modified = decoder @ latents.T
   └── Replace Boltz2's internal activations with y_modified
       → Measure impact on final structure prediction
```

---

## Comparison: Single Layer vs Multi-Layer

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                    ORIGINAL (SINGLE LAYER 47)                             ║
╚═══════════════════════════════════════════════════════════════════════════╝

collect_with_boltz_cli.py → Activations from layer 47 only
train.py                  → One Per-Layer Transcoder for final layer
                            (2048 features for output representations)

Pros:
  ✓ Simple, focused analysis
  ✓ Fast training (one model)
  ✓ Clear interpretation (final features)

Cons:
  ✗ No depth analysis
  ✗ Can't see feature evolution
  ✗ Limited intervention points

╔═══════════════════════════════════════════════════════════════════════════╗
║                    NEW (MULTI-LAYER 0, 8, 16, 24, 32, 40)                 ║
╚═══════════════════════════════════════════════════════════════════════════╝

collect_multi_layer.py    → Simultaneous collection from 6 layers
train_multi_layer.py      → Sequential training of 6 independent Per-Layer Transcoders
                            (2048 features × 6 layers = 12,288 total features)
validate_multi_layer.py   → Compare quality across layers

Pros:
  ✓ Feature evolution analysis
  ✓ Layer-wise intervention
  ✓ Depth-dependent patterns
  ✓ Computational efficiency (single forward pass collects all)

Cons:
  ✗ 6× storage (120 MB checkpoints vs 20 MB)
  ✗ 6× training time (still only ~2 minutes)
  ✗ More complex analysis

Use Cases:
  • Study how features become more abstract with depth
  • Identify which layer is best for specific tasks
  • Intervene at optimal depth for structure modification
  • Compare early (sequence) vs late (structure) features
```

---

## Pipeline Execution Timeline

```
START: ./run_multi_layer_pipeline.sh
│
├─ [T+0:00] Start collection
│   ├─ [T+0:00 - T+3:00] Load Boltz2 model (one-time)
│   ├─ [T+3:00 - T+4:00] Process protein 1
│   ├─ [T+4:00 - T+5:00] Process protein 2
│   │   ...
│   └─ [T+12:00 - T+13:00] Process protein 10
│       → Output: 60 NPZ files (6 layers × 10 batches)
│
├─ [T+13:00] Start training
│   ├─ [T+13:00 - T+13:22] Train layer 0 (100 steps × 0.2s)
│   ├─ [T+13:22 - T+13:44] Train layer 8
│   ├─ [T+13:44 - T+14:06] Train layer 16
│   ├─ [T+14:06 - T+14:28] Train layer 24
│   ├─ [T+14:28 - T+14:50] Train layer 32
│   └─ [T+14:50 - T+15:12] Train layer 40
│       → Output: 6 checkpoints (~120 MB total)
│
├─ [T+15:12] Start validation
│   ├─ [T+15:12 - T+15:17] Validate layer 0
│   ├─ [T+15:17 - T+15:22] Validate layer 8
│   ├─ [T+15:22 - T+15:27] Validate layer 16
│   ├─ [T+15:27 - T+15:32] Validate layer 24
│   ├─ [T+15:32 - T+15:37] Validate layer 32
│   └─ [T+15:37 - T+15:42] Validate layer 40
│       → Output: validation_summary.json
│
└─ [T+15:42] COMPLETE
    Total time: 15.7 minutes
    
    Generated files:
      - multi_layer_activations/     (3 GB)
      - multi_layer_checkpoints/     (120 MB)
      - multi_layer_logs/            (1 MB)
      - multi_layer_training_summary.json
      - validation_summary.json
```

---

## Next Steps After Pipeline Completion

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        POST-TRAINING ANALYSIS                            │
└──────────────────────────────────────────────────────────────────────────┘

1. CHECK RESULTS
   │
   ├── cat multi_layer_checkpoints/validation_summary.json
   │   → Look for R² > 0.5 on most layers
   │
   ├── cat multi_layer_checkpoints/multi_layer_training_summary.json
   │   → Check dead neurons < 500 for all layers
   │
   └── tail multi_layer_logs/02_training.log
       → Verify losses decreased

2. COMPARE LAYERS
   │
   └── python -c "
       import json
       with open('multi_layer_checkpoints/validation_summary.json') as f:
           results = json.load(f)['results']
       
       print('Layer | R² (Y1) | R² (Y2) | Dead Neurons')
       for r in results:
           print(f\"{r['layer_idx']:5d} | {r['r2_y1']:7.3f} | 
                   {r['r2_y2']:7.3f} | {r.get('dead_neurons', 'N/A'):5s}\")
       "

3. FEATURE ANALYSIS (Optional)
   │
   ├── Load layer 0 features (early) vs layer 40 features (late)
   ├── Compute cosine similarity between features
   ├── Cluster features by activation patterns
   └── Correlate with structural properties

4. INTERVENTION (Optional)
   │
   ├── Replace Boltz2's layer 24 activations with transcoder output
   ├── Amplify specific features
   └── Measure impact on final structure prediction

5. ITERATE (If needed)
   │
   ├── If R² < 0.5: Increase --num_steps to 500
   ├── If high dead neurons: Increase --max_proteins to 50
   └── Re-run pipeline with better parameters
```

---

**For detailed implementation, see [MULTI_LAYER_PLT_GUIDE.md](MULTI_LAYER_PLT_GUIDE.md)**
