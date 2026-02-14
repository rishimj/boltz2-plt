#!/usr/bin/env python3
"""
Time inference on Boltz examples to benchmark performance.

This script runs inference on all example YAML files and measures:
- Total execution time (including MSA generation)
- Structure prediction time
- Peak GPU memory usage
- Scaling with multiple diffusion samples
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

# Add parent directories to path to import profiling utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tests"))
try:
    from profiling import clear_memory, get_gpu_memory_gb
except ImportError:
    print("Warning: Could not import profiling utilities. Memory tracking disabled.")
    
    def clear_memory(device):
        """Fallback clear_memory if import fails."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def get_gpu_memory_gb(device):
        """Fallback get_gpu_memory_gb if import fails."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(device) / 1e9
        return 0.0


def run_boltz_predict(
    input_path: Path,
    output_dir: Path,
    diffusion_samples: int = 1,
    use_msa_server: bool = True,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    accelerator: str = "gpu",
    devices: int = 1,
    model: str = "boltz2",
) -> Dict:
    """
    Run boltz predict command and measure timing.
    
    Args:
        input_path: Path to input YAML file
        output_dir: Output directory for predictions
        diffusion_samples: Number of diffusion samples to generate
        use_msa_server: Whether to use MSA server
        recycling_steps: Number of recycling steps
        sampling_steps: Number of diffusion sampling steps
        accelerator: Device type (gpu/cpu)
        devices: Number of devices
        model: Model version (boltz1/boltz2)
    
    Returns:
        Dict with timing results and metadata
    """
    # Build command
    cmd = [
        "boltz",
        "predict",
        str(input_path),
        "--out_dir",
        str(output_dir),
        "--diffusion_samples",
        str(diffusion_samples),
        "--recycling_steps",
        str(recycling_steps),
        "--sampling_steps",
        str(sampling_steps),
        "--accelerator",
        accelerator,
        "--devices",
        str(devices),
        "--model",
        model,
    ]
    
    if use_msa_server:
        cmd.append("--use_msa_server")
    
    # Clear GPU memory before run
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        clear_memory(device)
        initial_memory = get_gpu_memory_gb(device)
    else:
        device = None
        initial_memory = 0.0
    
    # Run command and measure time
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    # Set cache directories to writable locations
    env = os.environ.copy()
    base_dir = Path(__file__).parent.parent.parent
    
    # Override HOME to writable directory to avoid /nethome permission issues
    env['HOME'] = '/usr/scratch/rmanimaran8'
    
    # Set BOLTZ_CACHE
    if 'BOLTZ_CACHE' not in env:
        cache_dir = base_dir / '.boltz_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        env['BOLTZ_CACHE'] = str(cache_dir)
    
    # Set XDG_CACHE_HOME for triton and other cache needs
    if 'XDG_CACHE_HOME' not in env:
        xdg_cache = base_dir / '.cache'
        xdg_cache.mkdir(parents=True, exist_ok=True)
        env['XDG_CACHE_HOME'] = str(xdg_cache)
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
            env=env,
        )
        success = True
        error_msg = None
    except subprocess.CalledProcessError as e:
        success = False
        error_msg = e.stderr if hasattr(e, 'stderr') else str(e)
        result = e
    
    # Ensure CUDA operations complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Get peak GPU memory
    if torch.cuda.is_available():
        peak_memory = get_gpu_memory_gb(device)
    else:
        peak_memory = 0.0
    
    return {
        "input_file": input_path.name,
        "diffusion_samples": diffusion_samples,
        "total_time_sec": total_time,
        "peak_gpu_memory_gb": peak_memory,
        "status": "success" if success else "failed",
        "error": error_msg,
        "use_msa_server": use_msa_server,
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "model": model,
    }


def run_warmup(examples_dir: Path, output_dir: Path) -> None:
    """
    Run a warm-up prediction to initialize GPU and load model checkpoint.
    
    Args:
        examples_dir: Directory containing example files
        output_dir: Output directory for warm-up run
    """
    print("\n" + "=" * 80)
    print("WARM-UP RUN")
    print("=" * 80)
    
    warmup_input = examples_dir / "prot.yaml"
    if not warmup_input.exists():
        print(f"Warning: Warm-up file {warmup_input} not found. Skipping warm-up.")
        return
    
    warmup_dir = output_dir / "warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running warm-up on {warmup_input.name}...")
    result = run_boltz_predict(
        input_path=warmup_input,
        output_dir=warmup_dir,
        diffusion_samples=1,
        use_msa_server=True,
    )
    
    print(f"Warm-up completed in {result['total_time_sec']:.2f}s")
    print(f"Status: {result['status']}")
    if result['status'] == 'failed':
        print(f"Error: {result['error']}")
    print("=" * 80 + "\n")


def get_example_type(yaml_path: Path) -> str:
    """
    Determine the type of example from the filename.
    
    Args:
        yaml_path: Path to YAML file
    
    Returns:
        String describing example type
    """
    name = yaml_path.stem.lower()
    
    type_map = {
        "prot": "protein",
        "prot_no_msa": "protein_no_msa",
        "prot_custom_msa": "protein_custom_msa",
        "ligand": "protein_ligand",
        "affinity": "protein_ligand_affinity",
        "multimer": "protein_multimer",
        "pocket": "pocket_constrained",
        "cyclic_prot": "cyclic_peptide",
    }
    
    for key, value in type_map.items():
        if key in name:
            return value
    
    return "unknown"


def run_single_sample_experiments(
    examples_dir: Path,
    output_dir: Path,
    skip_fasta: bool = True,
) -> List[Dict]:
    """
    Run single-sample timing experiments on all examples.
    
    Args:
        examples_dir: Directory containing example YAML files
        output_dir: Output directory for predictions
        skip_fasta: Whether to skip FASTA files (deprecated format)
    
    Returns:
        List of result dictionaries
    """
    print("\n" + "=" * 80)
    print("SINGLE-SAMPLE EXPERIMENTS")
    print("=" * 80)
    
    results = []
    
    # Find all YAML files
    yaml_files = sorted(examples_dir.glob("*.yaml"))
    
    if skip_fasta:
        # Filter out FASTA-related files (deprecated)
        yaml_files = [f for f in yaml_files if "fasta" not in f.stem.lower()]
    
    print(f"Found {len(yaml_files)} example files")
    
    for yaml_file in yaml_files:
        print(f"\n{'-' * 80}")
        print(f"Processing: {yaml_file.name}")
        print(f"Type: {get_example_type(yaml_file)}")
        print(f"{'-' * 80}")
        
        run_dir = output_dir / "single_sample" / yaml_file.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        
        result = run_boltz_predict(
            input_path=yaml_file,
            output_dir=run_dir,
            diffusion_samples=1,
            use_msa_server=True,
        )
        
        result["example_type"] = get_example_type(yaml_file)
        results.append(result)
        
        print(f"Time: {result['total_time_sec']:.2f}s")
        print(f"Memory: {result['peak_gpu_memory_gb']:.2f} GB")
        print(f"Status: {result['status']}")
    
    print("=" * 80 + "\n")
    return results


def run_multi_sample_experiments(
    examples_dir: Path,
    output_dir: Path,
    diffusion_samples: int = 5,
) -> List[Dict]:
    """
    Run multi-sample scaling experiments on representative examples.
    
    Args:
        examples_dir: Directory containing example YAML files
        output_dir: Output directory for predictions
        diffusion_samples: Number of samples to generate
    
    Returns:
        List of result dictionaries
    """
    print("\n" + "=" * 80)
    print(f"MULTI-SAMPLE EXPERIMENTS ({diffusion_samples} samples)")
    print("=" * 80)
    
    results = []
    
    # Representative examples to test scaling
    representative_examples = ["prot.yaml", "ligand.yaml", "multimer.yaml"]
    
    for example_name in representative_examples:
        yaml_file = examples_dir / example_name
        
        if not yaml_file.exists():
            print(f"Warning: {example_name} not found. Skipping.")
            continue
        
        print(f"\n{'-' * 80}")
        print(f"Processing: {yaml_file.name} ({diffusion_samples} samples)")
        print(f"Type: {get_example_type(yaml_file)}")
        print(f"{'-' * 80}")
        
        run_dir = output_dir / f"multi_sample_{diffusion_samples}" / yaml_file.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        
        result = run_boltz_predict(
            input_path=yaml_file,
            output_dir=run_dir,
            diffusion_samples=diffusion_samples,
            use_msa_server=True,
        )
        
        result["example_type"] = get_example_type(yaml_file)
        
        # Calculate per-sample time (approximate)
        result["time_per_sample_sec"] = result["total_time_sec"] / diffusion_samples
        
        results.append(result)
        
        print(f"Total time: {result['total_time_sec']:.2f}s")
        print(f"Time per sample: {result['time_per_sample_sec']:.2f}s")
        print(f"Memory: {result['peak_gpu_memory_gb']:.2f} GB")
        print(f"Status: {result['status']}")
    
    print("=" * 80 + "\n")
    return results


def save_results(results: List[Dict], output_file: Path) -> None:
    """
    Save results to CSV file with summary statistics.
    
    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file
    """
    df = pd.DataFrame(results)
    
    # Save detailed results
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Group by example type and diffusion samples
    if len(df) > 0:
        summary = df.groupby(["example_type", "diffusion_samples"]).agg({
            "total_time_sec": ["mean", "median", "std", "min", "max"],
            "peak_gpu_memory_gb": ["mean", "max"],
            "status": lambda x: (x == "success").sum(),
        }).round(2)
        
        print("\nTiming Statistics (seconds):")
        print(summary["total_time_sec"])
        
        print("\nMemory Statistics (GB):")
        print(summary["peak_gpu_memory_gb"])
        
        print("\nSuccess Count:")
        print(summary["status"])
    
    print("=" * 80 + "\n")


def main():
    """Main execution function."""
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    examples_dir = project_root / "examples"
    output_dir = script_dir / "timing_output"
    results_file = script_dir / "timing_results.csv"
    
    print("=" * 80)
    print("BOLTZ INFERENCE TIMING BENCHMARK")
    print("=" * 80)
    print(f"Examples directory: {examples_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Results file: {results_file}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available. Running on CPU.")
    
    print("=" * 80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # 1. Warm-up run
    run_warmup(examples_dir, output_dir)
    
    # 2. Single-sample experiments
    single_results = run_single_sample_experiments(examples_dir, output_dir)
    all_results.extend(single_results)
    
    # 3. Multi-sample scaling experiments
    multi_results = run_multi_sample_experiments(examples_dir, output_dir, diffusion_samples=5)
    all_results.extend(multi_results)
    
    # 4. Save results
    save_results(all_results, results_file)
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Total experiments: {len(all_results)}")
    print(f"Successful: {sum(1 for r in all_results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in all_results if r['status'] == 'failed')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
