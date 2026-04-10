#!/bin/bash
#
# Complete Multi-Layer PLT Training Pipeline
# 
# This script runs the full pipeline:
# 1. Collect activations from multiple pairformer layers
# 2. Train separate transcoders for each layer
# 3. Validate all trained transcoders
#
# Quick validation mode: ~10 proteins, 100 training steps per layer
# Expected time: ~30-60 minutes total

set -e  # Exit on error

# Configuration
LAYERS="0 8 16 24 32 40"
NUM_PROTEINS=10
TRAINING_STEPS=100
BATCH_SIZE=10
DEVICE="cuda"

# Directories
FASTA_FILE="/usr/scratch/rmanimaran8/boltz/examples/multi_protein.fasta"
ACTIVATION_DIR="multi_layer_10p_activations"
CHECKPOINT_DIR="multi_layer_10p_checkpoints"
LOG_DIR="multi_layer_10p_logs"

# Boltz2 checkpoint
BOLTZ_CHECKPOINT="/usr/scratch/rmanimaran8/boltz/.boltz_cache/boltz2_conf.ckpt"

# Create log directory
mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "MULTI-LAYER PLT TRAINING PIPELINE"
echo "========================================================================"
echo "Layers: $LAYERS"
echo "Proteins: $NUM_PROTEINS"
echo "Training steps per layer: $TRAINING_STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "========================================================================"
echo ""

# Activate environment
echo "Activating Boltz environment..."
source ../boltz_env/bin/activate
echo "✓ Environment activated"
echo ""

# Step 1: Collect activations
echo "========================================================================"
echo "STEP 1: COLLECT MULTI-LAYER ACTIVATIONS"
echo "========================================================================"
echo ""

cd collection_scripts

python collect_multi_layer.py \
    --checkpoint "$BOLTZ_CHECKPOINT" \
    --fasta "$FASTA_FILE" \
    --output "../$ACTIVATION_DIR" \
    --layers $LAYERS \
    --max-proteins $NUM_PROTEINS \
    --device $DEVICE \
    2>&1 | tee "../$LOG_DIR/01_collection.log"

cd ..

echo ""
echo "✓ Activation collection complete"
echo ""

# Step 2: Train transcoders
echo "========================================================================"
echo "STEP 2: TRAIN MULTI-LAYER TRANSCODERS"
echo "========================================================================"
echo ""

cd universal_transcoder

python train_multi_layer.py \
    --data_dir "../$ACTIVATION_DIR" \
    --checkpoint_dir "../$CHECKPOINT_DIR" \
    --layers $LAYERS \
    --batch_size $BATCH_SIZE \
    --num_steps $TRAINING_STEPS \
    --lr 1e-3 \
    --log_every 10 \
    2>&1 | tee "../$LOG_DIR/02_training.log"

cd ..

echo ""
echo "✓ Training complete"
echo ""

# Step 3: Validate transcoders
echo "========================================================================"
echo "STEP 3: VALIDATE TRANSCODERS"
echo "========================================================================"
echo ""

cd universal_transcoder

python validate_multi_layer.py \
    --checkpoint_dir "../$CHECKPOINT_DIR" \
    --data_dir "../$ACTIVATION_DIR" \
    --layers $LAYERS \
    --device $DEVICE \
    2>&1 | tee "../$LOG_DIR/03_validation.log"

cd ..

echo ""
echo "✓ Validation complete"
echo ""

# Summary
echo "========================================================================"
echo "PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "Results:"
echo "  Activations: $ACTIVATION_DIR/"
echo "  Checkpoints: $CHECKPOINT_DIR/"
echo "  Logs: $LOG_DIR/"
echo ""
echo "Training summary: $CHECKPOINT_DIR/multi_layer_training_summary.json"
echo "Validation summary: $CHECKPOINT_DIR/validation_summary.json"
echo ""
echo "View logs:"
echo "  Collection:  cat $LOG_DIR/01_collection.log"
echo "  Training:    cat $LOG_DIR/02_training.log"
echo "  Validation:  cat $LOG_DIR/03_validation.log"
echo ""
echo "========================================================================"
