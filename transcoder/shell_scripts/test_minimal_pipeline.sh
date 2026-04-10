#!/bin/bash
#
# MINIMAL PIPELINE TEST
# Tests the entire pipeline with absolute minimum resources:
# - 1 protein
# - 1 layer (layer 40)
# - 10 training steps (instead of 100)
# - Small batch size
#
# Expected time: ~10-15 minutes total
#

set -e

echo "========================================================================"
echo "MINIMAL PIPELINE TEST (1 protein, 1 layer, 10 steps)"
echo "========================================================================"
echo ""

# Configuration
TEST_LAYER=40
NUM_PROTEINS=1
TRAINING_STEPS=10
BATCH_SIZE=1
DEVICE="cuda"

# Directories
FASTA_FILE="/usr/scratch/rmanimaran8/boltz/examples/prot.fasta"
TEST_DIR="minimal_test"
ACTIVATION_DIR="${TEST_DIR}/activations"
CHECKPOINT_DIR="${TEST_DIR}/checkpoints"
LOG_FILE="${TEST_DIR}/test.log"

# Boltz2 checkpoint
BOLTZ_CHECKPOINT="/usr/scratch/rmanimaran8/boltz/.boltz_cache/boltz2_conf.ckpt"

# Create test directory
mkdir -p "$TEST_DIR"

# Activate environment
echo "Activating environment..."
source ../boltz_env/bin/activate

echo ""
echo "========================================================================"
echo "STEP 1/3: Collect activations (1 protein, layer $TEST_LAYER)"
echo "========================================================================"
echo ""

cd collection_scripts

python -u collect_multi_layer.py \
    --checkpoint "$BOLTZ_CHECKPOINT" \
    --fasta "$FASTA_FILE" \
    --output "../$ACTIVATION_DIR" \
    --layers $TEST_LAYER \
    --max-proteins $NUM_PROTEINS \
    --device $DEVICE \
    2>&1 | tee "../${LOG_FILE}"

cd ..

# Check if data was collected
LAYER_DIR="${ACTIVATION_DIR}/layer_${TEST_LAYER}"
if [ ! -d "$LAYER_DIR" ]; then
    echo "❌ ERROR: No activation directory created at $LAYER_DIR"
    exit 1
fi

NUM_BATCHES=$(ls -1 "$LAYER_DIR"/*.npz 2>/dev/null | wc -l)
if [ "$NUM_BATCHES" -eq 0 ]; then
    echo "❌ ERROR: No activation batches saved"
    exit 1
fi

echo ""
echo "✓ Collection successful: $NUM_BATCHES batch(es) saved"
echo ""

echo "========================================================================"
echo "STEP 2/3: Train transcoder ($TRAINING_STEPS steps)"
echo "========================================================================"
echo ""

cd universal_transcoder

python -u train_multi_layer.py \
    --data_dir "../$ACTIVATION_DIR" \
    --checkpoint_dir "../$CHECKPOINT_DIR" \
    --layers $TEST_LAYER \
    --batch_size $BATCH_SIZE \
    --num_steps $TRAINING_STEPS \
    --lr 1e-3 \
    --log_every 2 \
    2>&1 | tee -a "../${LOG_FILE}"

cd ..

# Check if checkpoint was created
CHECKPOINT_FILE="${CHECKPOINT_DIR}/layer_${TEST_LAYER}/universal_transcoder_final.pt"
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "❌ ERROR: No checkpoint created at $CHECKPOINT_FILE"
    exit 1
fi

echo ""
echo "✓ Training successful: checkpoint saved"
echo ""

echo "========================================================================"
echo "STEP 3/3: Validate transcoder"
echo "========================================================================"
echo ""

cd universal_transcoder

python -u validate_multi_layer.py \
    --checkpoint_dir "../$CHECKPOINT_DIR" \
    --data_dir "../$ACTIVATION_DIR" \
    --layers $TEST_LAYER \
    --device $DEVICE \
    --max_batches $NUM_BATCHES \
    2>&1 | tee -a "../${LOG_FILE}"

cd ..

echo ""
echo "========================================================================"
echo "MINIMAL TEST COMPLETE! ✓"
echo "========================================================================"
echo ""
echo "Results:"
echo "  Test directory: $TEST_DIR/"
echo "  Activations: $ACTIVATION_DIR/layer_${TEST_LAYER}/ ($NUM_BATCHES batches)"
echo "  Checkpoint: $CHECKPOINT_FILE"
echo "  Full log: $LOG_FILE"
echo ""
echo "Validation summary: ${CHECKPOINT_DIR}/validation_summary.json"
echo ""
echo "If this test succeeded, you can run the full pipeline:"
echo "  ./run_multi_layer_pipeline.sh"
echo ""
echo "========================================================================"
