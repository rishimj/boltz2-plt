# Boltz Inference Timing Scripts

This directory contains scripts to benchmark and time Boltz inference performance across different example types.

## Files

- **`time_inference.py`**: Full timing script that runs actual Boltz predictions
- **`time_inference_demo.py`**: Demo version showing workflow with simulated timings
- **`timing_results.csv`**: Output file with detailed timing measurements (generated after running)
- **`timing_output/`**: Directory containing prediction outputs from timing runs

## Quick Start

### Prerequisites

You need Python >= 3.9 and Boltz installed:

```bash
# Create a fresh Python environment (recommended)
python3 -m venv boltz_timing_env
source boltz_timing_env/bin/activate  # On Windows: boltz_timing_env\Scripts\activate

# Install Boltz with CUDA support
pip install boltz[cuda]

# Or for CPU-only (slower):
# pip install boltz
```

### Running the Demo (No Installation Required)

To see the workflow without installing Boltz:

```bash
python3 scripts/eval/time_inference_demo.py
```

This shows simulated timings for all examples.

### Running Real Timing Experiments

Once Boltz is installed:

```bash
python3 scripts/eval/time_inference.py
```

This will:
1. **Warm-up**: Run one prediction to initialize GPU and load model
2. **Single-sample tests**: Time all 8 examples with `--diffusion_samples 1`
3. **Multi-sample tests**: Time 3 representative examples with `--diffusion_samples 5`
4. **Generate CSV**: Save detailed results to `timing_results.csv`

## What Gets Measured

For each example, the script measures:

- **Total execution time** (includes MSA generation + structure prediction)
- **Peak GPU memory usage** (using PyTorch CUDA memory tracking)
- **Success/failure status**
- **Per-sample timing** (for multi-sample runs)

## Examples Tested

| Example File | Type | Description |
|--------------|------|-------------|
| `prot.yaml` | Protein | Basic single-chain protein |
| `prot_no_msa.yaml` | Protein (no MSA) | Protein without MSA search |
| `prot_custom_msa.yaml` | Protein (custom MSA) | Protein with pre-computed MSA |
| `ligand.yaml` | Protein-ligand | Protein with small molecule |
| `affinity.yaml` | Affinity prediction | Protein-ligand with binding affinity |
| `multimer.yaml` | Multimer | Multi-chain protein complex |
| `pocket.yaml` | Pocket-constrained | Docking with binding site constraints |
| `cyclic_prot.yaml` | Cyclic peptide | Peptide with cyclic topology |

## Output Format

The `timing_results.csv` contains:

```
input_file,diffusion_samples,total_time_sec,peak_gpu_memory_gb,status,error,example_type,time_per_sample_sec
prot.yaml,1,123.45,8.32,success,,protein,
ligand.yaml,1,145.67,9.21,success,,protein_ligand,
prot.yaml,5,567.89,42.10,success,,protein,113.58
```

Summary statistics are also printed to console:
- Mean/median/std timing per example type
- Peak memory usage
- Success rates

## Configuration

Edit the script to customize:

- `diffusion_samples`: Number of structure samples to generate (default: 1 for single, 5 for multi)
- `recycling_steps`: Number of recycling iterations (default: 3)
- `sampling_steps`: Diffusion sampling steps (default: 200)
- `use_msa_server`: Whether to use MMseqs2 MSA server (default: True)
- `model`: Model version - "boltz1" or "boltz2" (default: boltz2)

## Understanding Timing

**Diffusion Samples**: Each sample is an independent structure prediction starting from different random noise. More samples = better quality but slower:

- `--diffusion_samples 1`: Fastest, good for quick predictions
- `--diffusion_samples 5`: Balanced quality/speed
- `--diffusion_samples 25`: AlphaFold3-like quality (much slower)

**MSA Generation**: When using `--use_msa_server`, the first run includes MSA search time. Subsequent runs on the same sequence will be faster if cached.

**GPU Memory**: Scales with:
- Number of parallel samples (capped by `--max_parallel_samples 5`)
- Protein/complex size
- Model version

## Troubleshooting

### `ModuleNotFoundError: No module named 'boltz'`

Boltz is not installed. Run:
```bash
pip install boltz[cuda]
```

### `CUDA out of memory`

Reduce parallel samples:
```bash
# Edit time_inference.py line ~87:
max_parallel_samples: Optional[int] = 1,  # Reduce from 5 to 1
```

### Python version too old

Boltz requires Python >= 3.9. Create a new environment:
```bash
conda create -n boltz_timing python=3.10
conda activate boltz_timing
pip install boltz[cuda]
```

## Expected Timing Ranges (RTX A6000 GPU)

Rough estimates per example (single sample):

- Simple protein (~100-200 residues): 60-120 seconds
- Protein-ligand: 100-180 seconds  
- Multimer (2-3 chains): 150-300 seconds
- With MSA generation: +30-120 seconds (first run only)

Multi-sample (5x) typically takes 4-5x the single-sample time due to parallelization.

## Citation

If using Boltz for your research, please cite:

```bibtex
@article{boltz2024,
  title={Boltz-1: A Family of Molecular Models for Biomolecular Interaction Prediction},
  author={Wohlwend, Jeremy and ...},
  journal={bioRxiv},
  year={2024}
}
```

For questions or issues, join the [Boltz Slack channel](https://boltz.bio/join-slack).
