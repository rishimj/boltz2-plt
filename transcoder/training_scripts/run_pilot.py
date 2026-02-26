"""Pilot run script for testing the full pipeline."""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully!")
    return result


def main():
    parser = argparse.ArgumentParser(description='Run pilot transcoder training pipeline')
    parser.add_argument('--model', type=str, required=True, help='Path to Boltz model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to data manifest')
    parser.add_argument('--skip-collection', action='store_true', help='Skip activation collection')
    parser.add_argument('--skip-training', action='store_true', help='Skip training')
    
    args = parser.parse_args()
    
    # Paths
    transcoder_dir = Path(__file__).parent
    activation_dir = transcoder_dir / "pilot_activations"
    checkpoint_dir = transcoder_dir / "pilot_checkpoints"
    log_file = transcoder_dir / "pilot_training_log.json"
    
    print(f"\n{'='*80}")
    print("PILOT TRANSCODER TRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"Transcoder directory: {transcoder_dir}")
    print(f"Activation directory: {activation_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Step 1: Collect activations
    if not args.skip_collection:
        collect_cmd = [
            sys.executable,
            str(transcoder_dir / "collect_activations.py"),
            "--model", args.model,
            "--data", args.data,
            "--output", str(activation_dir),
            "--max-structures", "100",
            "--batch-size", "1",
            "--layer", "47",
        ]
        run_command(collect_cmd, "Step 1: Collecting activations")
    else:
        print("\n⏭️  Skipping activation collection")
    
    # Step 2: Test the transcoder model
    print(f"\n{'='*80}")
    print("Step 2: Testing transcoder model")
    print(f"{'='*80}\n")
    
    test_cmd = [
        sys.executable,
        str(transcoder_dir / "model.py"),
    ]
    run_command(test_cmd, "Testing transcoder model")
    
    # Step 3: Train transcoder
    if not args.skip_training:
        train_cmd = [
            sys.executable,
            str(transcoder_dir / "train.py"),
            "--activations", str(activation_dir),
            "--checkpoints", str(checkpoint_dir),
            "--log", str(log_file),
            "--latent-dim", "2048",
            "--l1-coeff", "0.0001",
            "--lr", "0.001",
            "--batch-size", "32",
            "--epochs", "10",  # Short pilot run
            "--checkpoint-every", "500",
        ]
        run_command(train_cmd, "Step 3: Training transcoder")
    else:
        print("\n⏭️  Skipping training")
    
    # Summary
    print(f"\n{'='*80}")
    print("✓ PILOT RUN COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults:")
    print(f"  - Activations: {activation_dir}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Training log: {log_file}")
    print(f"\nNext steps:")
    print(f"  1. Review training log and metrics")
    print(f"  2. Inspect activation files and shapes")
    print(f"  3. Verify checkpoint files were saved")
    print(f"  4. If successful, scale up to full dataset")
    print()


if __name__ == "__main__":
    main()
