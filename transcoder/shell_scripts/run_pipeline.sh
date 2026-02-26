#!/bin/bash
# Complete pipeline: Collect activations → Train transcoder → Analyze

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=== Boltz Transcoder Training Pipeline ==="
echo "Start time: $(date)"
echo ""

# Step 1: Collect activations
echo "STEP 1: Collecting activations..."
python collect_activations_fixed.py \
    --checkpoint ../boltz2_checkpoint.ckpt \
    --predictions_dir ../test_output/boltz_results_prot/processed \
    --output_dir real_activations \
    --device cuda

if [ $? -ne 0 ]; then
    echo "✗ Activation collection failed!"
    exit 1
fi

# Verify activations were created
if [ ! -d "real_activations" ] || [ -z "$(ls -A real_activations/*.npz 2>/dev/null)" ]; then
    echo "✗ No activation files found!"
    exit 1
fi

echo "✓ Activations collected successfully"
echo ""

# Step 2: Train transcoder
echo "STEP 2: Training transcoder on real data..."
python train.py \
    --activations real_activations \
    --checkpoints real_model_checkpoints \
    --log training_log_real.txt \
    --epochs 100 \
    --batch-size 1 \
    --lr 1e-3 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "✗ Training failed!"
    exit 1
fi

echo "✓ Training completed successfully"
echo ""

# Step 3: Compare results
echo "STEP 3: Comparing synthetic vs real training..."
echo ""
echo "=== Synthetic Data Results ==="
tail -15 training_log.txt 2>/dev/null || echo "No synthetic log"
echo ""
echo "=== Real Data Results ==="
tail -15 training_log_real.txt 2>/dev/null || echo "No real log"
echo ""

echo "=== Pipeline Complete ==="
echo "End time: $(date)"
echo ""
echo "Files created:"
echo "  - real_activations/*.npz        # Activation data"
echo "  - real_model_checkpoints/       # Training checkpoints"
echo "  - training_log_real.txt         # Training metrics"
echo "  - transcoder_real_final.pt      # Final model (if saved)"
