# Data Loading Integration Guide

## Why Data Loading Needs Work

The Boltz codebase has **complete** data loading infrastructure in `src/boltz/data/module/trainingv2.py`, but integrating it requires understanding several components that work together:

### The Boltz2 Data Pipeline

```
Raw Structure (.npz) → Input → Tokenized → Featurized → Model Input Dict
```

#### 1. **Structure Files** (`*.npz`)
Located in `target_dir/structures/`, each file contains:
- `atoms`: Atom coordinates and features
- `bonds`: Bond connectivity
- `residues`: Residue information  
- `chains`: Chain assignments
- `connections`: Inter-chain connections
- `interfaces`: Interface definitions
- `mask`: Valid atom mask

#### 2. **MSA Files** (`.npz`)
Located in `msa_dir/`, indexed by chain MSA ID:
- Multiple sequence alignments per chain
- Used for evolutionary features

#### 3. **Manifest** (`manifest.json`)
Lists all structures with metadata:
```json
{
  "records": [
    {
      "id": "structure_001",
      "chains": [...],
      ...
    }
  ]
}
```

## What's Already Available

### ✅ BoltzTrainingDataModule
Full PyTorch Lightning DataModule in `src/boltz/data/module/trainingv2.py`:

```python
from boltz.data.module.trainingv2 import BoltzTrainingDataModule

# Initialize with config
data_module = BoltzTrainingDataModule(config)
data_module.setup()

# Get dataloader
dataloader = data_module.train_dataloader()

# Iterate
for batch in dataloader:
    # batch is dict with all features ready for model
    output = model(batch)
```

**The dataloader already:**
- Loads structures from .npz files ✓
- Loads MSAs ✓
- Tokenizes structures ✓
- Featurizes for model input ✓
- Handles batching and padding ✓
- Returns model-ready feature dicts ✓

### ✅ Model Loading
```python
from boltz.model.models.boltz2 import Boltz2

model = Boltz2.load_from_checkpoint("checkpoint.ckpt")
model.eval()
```

## Why collect_activations.py Has TODOs

The issue is **NOT** that we can't load data - it's that the full pipeline requires:

### Missing Integration Pieces

1. **Config Creation**: `BoltzTrainingDataModule` needs a `DataConfig` object with:
   - Dataset paths
   - Tokenizer config
   - Featurizer config
   - Cropper config
   - Filter configs
   - Many hyperparameters

2. **Tokenizer & Featurizer Setup**: These need to be instantiated with proper configs

3. **Manifest**: Need to either:
   - Use existing manifest.json files
   - Create manifest from directory of structures
   - Use the dataset's existing manifests

## Two Approaches to Fix This

### Approach 1: Use Existing Training Infrastructure (Recommended)

Create a simple config and use the existing data module:

```python
from boltz.data.module.trainingv2 import BoltzTrainingDataModule, DataConfig, DatasetConfig
from boltz.data.tokenize.tokenizer import BoltzTokenizer
from boltz.data.feature.featurizer import BoltzFeaturizer
# ... other imports

# Create minimal config
config = DataConfig(
    datasets=[
        DatasetConfig(
            target_dir="/path/to/structures",
            msa_dir="/path/to/msa",
            prob=1.0,
            sampler=SimpleSampler(),  # Need to define
            cropper=SimpleCropper(),   # Need to define
        )
    ],
    tokenizer=BoltzTokenizer(...),
    featurizer=BoltzFeaturizer(...),
    # ... many other required fields
)

data_module = BoltzTrainingDataModule(config)
dataloader = data_module.train_dataloader()
```

**Pros:**
- Uses battle-tested infrastructure
- Handles all edge cases
- Proper tokenization/featurization

**Cons:**
- Requires understanding full config structure
- Many dependencies to set up correctly

### Approach 2: Simple Direct Loading (Current Placeholder)

Load .npz files directly and build features manually:

```python
# Load structure
structure_data = np.load("structure.npz")

# Manually create features (SIMPLIFIED)
feats = {
    'atom_pos': structure_data['atoms'][:, :3],
    'atom_mask': structure_data['mask'],
    # ... need to manually create ALL required features
}

# Run model
output = model(feats)
```

**Pros:**
- Simple, direct
- Good for quick testing

**Cons:**
- Have to manually replicate tokenization
- Have to manually replicate featurization  
- Easy to miss required features
- Won't match training pipeline exactly

## Recommended Next Steps

### Option A: Quick & Dirty (Get Something Working Fast)

1. Find a dataset that already has manifest.json
2. Look at existing training configs in `scripts/train/configs/`
3. Adapt one of those configs for your use case
4. Use BoltzTrainingDataModule directly

Example:
```python
# Use existing dataset
dataset_path = "/path/to/training/data"
manifest = dataset_path / "manifest.json"
structures = dataset_path / "structures"
msa = dataset_path / "msa"

# Load with existing config style
# (you'd need to create proper config object)
```

### Option B: Understand and Implement Properly

1. Study `src/boltz/data/module/trainingv2.py` carefully
2. Study `scripts/train/configs/full.yaml` for config structure
3. Create minimal working config for your pilot
4. Implement proper integration

## What I've Done So Far

The current `collect_activations.py`:

✅ **Working:**
- Hook registration on correct layer
- Activation capture mechanism
- NPZ file saving
- Model loading

⚠️ **Placeholder:**
- Data loading (shows where it goes)
- Feature preparation (marked as TODO)
- Forward pass execution (structure there, needs data)

**Why it's a placeholder:** To properly integrate requires either:
1. ~100 lines of config setup code (if using existing infrastructure)
2. ~200 lines of manual feature creation (if doing it custom)

I wanted to show you the structure first rather than make assumptions about which approach you prefer.

## Immediate Action Items

**To make this work, you need to:**

1. **Locate your training data:**
   - Where are the .npz structure files?
   - Where are the MSA files?
   - Is there a manifest.json?

2. **Choose approach:**
   - Use existing DataModule (more robust)
   - Or write custom loader (faster to prototype)

3. **I can then implement** whichever approach you choose

## Example: Using Existing DataModule

If you have a training dataset at `/path/to/boltz2_data/` with manifest.json, structures/, and msa/, I can write:

```python
# This would be the real implementation
from boltz.data.module.trainingv2 import BoltzTrainingDataModule
import yaml

# Load config from training
with open("scripts/train/configs/full.yaml") as f:
    config = yaml.safe_load(f)

# Adapt for inference
config['datasets'][0]['target_dir'] = "/path/to/your/data"
# ... modify other paths

data_module = BoltzTrainingDataModule.from_config(config)
dataloader = data_module.train_dataloader()

# Now we can iterate
for batch in dataloader:
    output = model(batch)  # Works!
```

Would you like me to:
1. Show you how to find/create a manifest.json for your data?
2. Implement the full DataModule integration?
3. Write a simpler custom loader?
4. Help you locate existing Boltz2 training data?
