#!/bin/bash
# Run Universal Transcoder Training Pipeline
# 
# This script:
# 1. Generates synthetic activation data
# 2. Trains the universal transcoder
# 3. Reports timing and results

set -e  # Exit on error

echo "=================================="
echo "UNIVERSAL TRANSCODER TRAINING PIPELINE"
echo "=================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
DATA_DIR="data"
CHECKPOINT_DIR="checkpoints"
LOG_DIR="logs"
NUM_BATCHES=10
SAMPLES_PER_BATCH=10
NUM_STEPS=100
BATCH_SIZE=10

# Create directories
mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

echo "Step 1: Generating synthetic data..."
echo "-----------------------------------"
python create_synthetic_data.py \
    --output_dir "$DATA_DIR" \
    --num_batches $NUM_BATCHES \
    --samples_per_batch $SAMPLES_PER_BATCH \
    --num_tokens 117

echo ""
echo "Step 2: Training universal transcoder..."
echo "-----------------------------------"

# Record start time
TOTAL_START=$(date +%s)

# Run training
python train_universal.py \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --lr 1e-3 \
    --log_every 10 \
    --d_model 384 \
    --d_hidden 2048 \
    --d_pair 128 \
    --k 16 \
    --auxk 32 \
    --dead_steps_threshold 10000 \
    2>&1 | tee "$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

# Record end time
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "=================================="
echo "PIPELINE COMPLETE"
echo "=================================="
echo "Total pipeline time: $TOTAL_TIME seconds"
echo ""
echo "Results saved to:"
echo "  - Checkpoint: $CHECKPOINT_DIR/universal_transcoder_final.pt"
echo "  - Metrics: $CHECKPOINT_DIR/training_metrics.json"
echo "  - Log: $LOG_DIR/"
echo ""
echo "To view training metrics:"
echo "  cat $CHECKPOINT_DIR/training_metrics.json | python -m json.tool"
echo ""
