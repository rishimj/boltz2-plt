# Where is Boltz2 Training Data?

## Short Answer

**The Boltz2 training data is NOT in this repository.** It needs to be downloaded separately from AWS S3.

## The Training Data Sources

Boltz-1 (and Boltz-2 uses similar data) was trained on:

### 1. **RCSB/PDB Dataset** (Protein Data Bank)
- **Source:** Public protein structures from https://www.rcsb.org/
- **Pre-processed download:**
  ```bash
  # Structures (~250GB total storage needed)
  wget https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_targets.tar
  tar -xf rcsb_processed_targets.tar
  
  # MSAs
  wget https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_msa.tar
  tar -xf rcsb_processed_msa.tar
  ```

### 2. **OpenFold Distillation Dataset**
- **Source:** Synthetic data from AlphaFold2/OpenFold predictions
- **Pre-processed download:**
  ```bash
  # Structures
  wget https://boltz1.s3.us-east-2.amazonaws.com/openfold_processed_targets.tar
  tar -xf openfold_processed_targets.tar
  
  # MSAs
  wget https://boltz1.s3.us-east-2.amazonaws.com/openfold_processed_msa.tar
  tar -xf openfold_processed_msa.tar
  ```

## Data Format

Once downloaded and extracted, you get:

```
processed_targets/
├── structures/
│   ├── 1abc.npz
│   ├── 1def.npz
│   └── ...
└── manifest.json

processed_msa/
├── hash1.npz
├── hash2.npz
└── ...
```

Each structure `.npz` file contains:
- `atoms`: Atom coordinates and features
- `bonds`: Bond connectivity
- `residues`: Residue information
- `chains`: Chain assignments
- `connections`: Inter-chain connections
- `interfaces`: Interface definitions
- `mask`: Valid atom mask

## For Your Transcoder Project

### Option 1: Use Pre-processed Data (Recommended)

Download the processed datasets:
```bash
cd /usr/scratch/rmanimaran8/boltz
mkdir -p data
cd data

# Download structures (required)
wget https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_targets.tar
tar -xf rcsb_processed_targets.tar

# Download MSAs (optional for activation collection)
wget https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_msa.tar
tar -xf rcsb_processed_msa.tar
```

Then update your activation collection script:
```python
python collect_activations.py \
    --checkpoint /path/to/boltz2_checkpoint.ckpt \
    --structures ./data/rcsb_processed_targets/structures \
    --msa ./data/rcsb_processed_msa \
    --output pilot_activations \
    --max-structures 100
```

### Option 2: Use Test/Example Data (Fast)

The repo already has small test examples in:
```
/usr/scratch/rmanimaran8/boltz/test_output/boltz_results_prot/processed/
├── structures/
│   └── prot.npz
└── msa/
    └── prot_0.npz
```

You can use these for initial testing!

### Option 3: Process Your Own Data

If you have your own PDB/mmCIF files, follow the data processing pipeline in `docs/training.md`:
1. Download raw PDB files
2. Process them with `scripts/process/rcsb.py`
3. Generate MSAs with ColabFold
4. Process MSAs with `scripts/process/msa.py`

## Why No Data in Repo?

The full training dataset is **~250GB+**, which is:
- Too large for GitHub
- Available on AWS S3 instead
- Processed from public PDB database
- Can be re-created from scratch using the processing scripts

## For Pilot Testing

I recommend using the small test examples already in the repo first, then downloading a subset of the full data if needed:

```bash
# Quick test with existing data
python collect_activations.py \
    --checkpoint boltz2_checkpoint.ckpt \
    --structures test_output/boltz_results_prot/processed/structures \
    --msa test_output/boltz_results_prot/processed/msa \
    --output pilot_activations \
    --max-structures 1
```

This will verify the pipeline works before committing to downloading 250GB!
