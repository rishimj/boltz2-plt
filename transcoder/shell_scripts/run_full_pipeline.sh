#!/bin/bash
# Complete pipeline with monitoring

set -e  # Exit on error

LOGDIR=/usr/scratch/rmanimaran8/boltz/transcoder/logs
mkdir -p $LOGDIR

echo "======================================"
echo "TRANSCODER TRAINING PIPELINE"
echo "Started at: $(date)"
echo "======================================"

cd /usr/scratch/rmanimaran8/boltz
source boltz_env/bin/activate

# Step 1: Use existing prediction data
echo ""
echo "[STEP 1/3] Setting up prediction data..."
if [ -d "test_output/boltz_results_prot/processed" ]; then
    echo "✓ Using existing prediction: test_output/boltz_results_prot"
    mkdir -p transcoder/example_predictions
    # Create symlink if it doesn't exist
    if [ ! -e "transcoder/example_predictions/prot" ]; then
        ln -s ../../test_output/boltz_results_prot transcoder/example_predictions/prot
    fi
    echo "✓ Linked existing prediction data"
else
    echo "No existing predictions found. Please run boltz predict first."
    exit 1
fi
echo "✓ Step 1 complete at $(date)"

# Step 2: Collect activations
echo ""
echo "[STEP 2/3] Collecting activations from layer 48..."
cd transcoder
python collect_activations_fixed.py \
    --checkpoint ../boltz2_checkpoint.ckpt \
    --predictions_dir example_predictions \
    --output_dir real_activations \
    --device cuda \
    2>&1 | tee $LOGDIR/collection_$(date +%Y%m%d_%H%M%S).log

# Check if activations were collected
NUM_ACTIVATIONS=$(ls -1 real_activations/*.npz 2>/dev/null | wc -l)
if [ $NUM_ACTIVATIONS -eq 0 ]; then
    echo "ERROR: No activations collected!"
    exit 1
fi
echo "✓ Collected $NUM_ACTIVATIONS activation files"
echo "✓ Step 2 complete at $(date)"

# Step 3: Train transcoder
echo ""
echo "[STEP 3/3] Training transcoder..."
python train_dynamic.py \
    --activation_dir real_activations \
    --checkpoint_dir real_model \
    --log_file training_log_real.json \
    --num_epochs 100 \
    --batch_size 4 \
    --learning_rate 1e-3 \
    --device cuda \
    2>&1 | tee $LOGDIR/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "======================================"
echo "PIPELINE COMPLETE!"
echo "Finished at: $(date)"
echo "======================================"
echo ""
echo "Results:"
echo "  - Activations: transcoder/real_activations/"
echo "  - Trained model: transcoder/transcoder_real_final.pt"
echo "  - Checkpoints: transcoder/real_model/"
echo "  - Training log: transcoder/training_log_real.json"
echo "  - Full logs: transcoder/logs/"
