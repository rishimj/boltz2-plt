# Transcoder Directory Structure

**Last organized:** April 30, 2026

This directory contains the active transcoder research code for analyzing Boltz internal representations.

For the current high-level status, read `PROJECT_STATE.md` first.

---

## Directory Organization

```
transcoder/
├── universal_transcoder/          MAIN MODEL (ACTIVE)
│   ├── universal_model.py            - Transcoder architecture
│   ├── train_universal.py            - Training script
│   ├── checkpoints/
│   │   ├── universal_transcoder_final.pt  - Trained model (16 MB)
│   │   └── training_metrics.json          - Training history
│   └── evaluation_results/
│
├── real_activations/              TRAINING DATA (GENERATED, MAY BE LOCAL ONLY)
│   ├── batch_00000.npz               - Protein 1 (117 residues, 13 MB)
│   ├── batch_00001.npz               - Protein 2 (80 residues, 6.1 MB)
│   ├── protein_001.npz               - Original format
│   └── protein_002.npz               - Original format
│
├── analysis_output/               SMALL TRACKED RESULTS
│   └── analysis_results.json         - Detailed analysis metrics
│
├── collection_scripts/            DATA COLLECTION
│   ├── collect_direct.py             - Single protein collection (working)
│   ├── collect_batch.py              - Batch collection (working)
│   ├── create_batches.py             - Format conversion
│   ├── collect_activations.py        - Early version
│   ├── collect_activations_fixed.py  - Fixed version
│   ├── collect_from_fasta.py         - FASTA-based collection
│   ├── collect_real_activations.py   - Real data collection
│   ├── collect_simple.py             - Simplified collection
│   ├── collect_with_boltz_cli.py     - CLI-based collection
│   └── create_synthetic_activations.py - Synthetic data generation
│
├── training_scripts/              TRAINING AND ANALYSIS
│   ├── analyze_transcoder.py         - Analysis script (CURRENT)
│   ├── train.py                      - Old training script
│   ├── train_dynamic.py              - Dynamic training variant
│   └── run_pilot.py                  - Pilot experiment runner
│
├── validation_scripts/            PIPELINE VALIDATION
│   ├── verify_pipeline_reproducibility.py  - Reproducibility & intervention tests
│   ├── run_validation.sh             - Quick validation runner
│   └── README.md                     - Validation guide
│
├── documentation/                 PROJECT DOCUMENTATION
│   ├── PROJECT_STATE.md              - Current repo/project state
│   ├── TRANSCODER_PROJECT_SUMMARY.md - Main project summary
│   ├── PLT_ARCHITECTURE_GUIDE.md     - PLT theory and implementation
│   ├── NEXT_STEPS_IMPLEMENTATION_PLAN.md - Future work roadmap
│   ├── README.md                     - Original README
│   ├── QUICKSTART.md                 - Quick start guide
│   ├── DATA_LOADING_GUIDE.md         - Data loading instructions
│   ├── PILOT_RESULTS.md              - Pilot experiment results
│   └── WHERE_IS_DATA.md              - Data location guide
│
├── shell_scripts/                 AUTOMATION SCRIPTS
│   ├── run_pipeline.sh               - Full pipeline runner
│   ├── run_full_pipeline.sh          - Complete workflow
│   ├── run_examples.sh               - Example runs
│   ├── test_collection.sh            - Collection testing
│   ├── test_model.sh                 - Model testing
│   ├── try_boltz_cli.sh              - Boltz CLI experiments
│   └── check_status.sh               - Status checker
│
├── logs/                          EXECUTION LOGS (GENERATED)
│   ├── analysis.log                  - Analysis output
│   ├── collection_*.log              - Collection runs
│   ├── training_*.log                - Training runs
│   └── pipeline_*.log                - Pipeline execution
│
├── data/                          INPUT DATA
│   └── test_protein.fasta            - Test protein sequence
│
├── old_models/                    LEGACY CODE
│   ├── model.py                      - Original model architecture
│   └── ...                           - Historical reference material only
│
├── pid_files/                     PROCESS IDS (GENERATED)
│
├── __pycache__/                   PYTHON CACHE (GENERATED)
│
└── DIRECTORY_STRUCTURE.md         This file

```

---

## Quick Access

### To Train the Model:
```bash
cd universal_transcoder/
python train_universal.py
```

### To Analyze Results:
```bash
cd training_scripts/
python analyze_transcoder.py
```

### To Collect New Activations:
```bash
cd collection_scripts/
python collect_batch.py
```

### To Read Documentation:
```bash
cd documentation/
cat PROJECT_STATE.md
```

---

## Current Status

| Component | Status | Location |
|-----------|--------|----------|
| **Trained Model** | Ready | `universal_transcoder/checkpoints/universal_transcoder_final.pt` |
| **Training Data** | Present locally/tracked historically | `real_activations/batch_*.npz` |
| **Analysis Results** | Present | `analysis_output/analysis_results.json` |
| **Documentation** | Active | `documentation/PROJECT_STATE.md` |
| **Collection Pipeline** | Working | `collection_scripts/collect_batch.py` |

---

## Key Files

1. **Current Model:** `universal_transcoder/checkpoints/universal_transcoder_final.pt` (16 MB)
2. **Training Script:** `universal_transcoder/train_universal.py` (404 lines)
3. **Model Architecture:** `universal_transcoder/universal_model.py` (217 lines)
4. **Analysis Script:** `training_scripts/analyze_transcoder.py` (350 lines)
5. **Collection Script:** `collection_scripts/collect_batch.py` (215 lines)
6. **Project State:** `documentation/PROJECT_STATE.md`

---

## File Naming Conventions

### Activation Data Files:
- `batch_XXXXX.npz` - Training-ready batch format
- `protein_XXX.npz` - Individual protein activations

### Log Files:
- `collection_*.log` - Data collection logs
- `training_*.log` - Training execution logs
- `pipeline_*.log` - Full pipeline logs

### Scripts:
- `collect_*.py` - Data collection scripts
- `train_*.py` - Training scripts
- `run_*.sh` - Shell automation scripts
- `test_*.sh` - Testing scripts

---

## Next Steps

1. **Collect more proteins:** Use `collection_scripts/collect_batch.py`
2. **Retrain with more data:** Run `universal_transcoder/train_universal.py`
3. **Re-analyze:** Use `training_scripts/analyze_transcoder.py`
4. **Document findings:** Update `documentation/TRANSCODER_PROJECT_SUMMARY.md`

---

## Notes

- **Active development:** `universal_transcoder/` directory
- **Generated state:** logs, pid files, caches, checkpoints, and local outputs should generally stay out of git
- **Legacy:** `old_models/` is historical reference material
- **Logs:** All execution logs in `logs/` directory
- **Documentation:** All markdown files in `documentation/` directory

For current orientation, start with: `documentation/PROJECT_STATE.md`
