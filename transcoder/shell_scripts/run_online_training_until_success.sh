#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/usr/scratch/rmanimaran8/boltz/transcoder"
PYTHON_BIN="/usr/scratch/rmanimaran8/boltz/boltz_env/bin/python3"
TRAIN_SCRIPT="$ROOT_DIR/universal_transcoder/train_online_multi_layer.py"
LOG_DIR="$ROOT_DIR/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="online_train_${RUN_TS}"
CHECKPOINT_DIR="$ROOT_DIR/minimal_test/${RUN_NAME}_checkpoints"
RUN_LOG="$LOG_DIR/${RUN_NAME}.log"
STATUS_FILE="$LOG_DIR/${RUN_NAME}.status"

FASTA_PATH="/usr/scratch/rmanimaran8/boltz/examples/prot.fasta"
LAYERS="0"
NUM_STEPS="1"
BATCH_SIZE="1"
LOG_EVERY="1"
DEVICE="cuda"
MAX_ATTEMPTS="0"
SLEEP_BETWEEN_ATTEMPTS="20"

mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "START_TIME=$(date -Is)" | tee -a "$RUN_LOG"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR" | tee -a "$RUN_LOG"
echo "STATUS=RUNNING" > "$STATUS_FILE"

action_success() {
  local summary_file="$CHECKPOINT_DIR/online_multi_layer_training_summary.json"
  local ckpt_file="$CHECKPOINT_DIR/layer_00/universal_transcoder_final.pt"

  if [[ ! -f "$summary_file" ]]; then
    return 1
  fi

  if [[ ! -f "$ckpt_file" ]]; then
    return 1
  fi

  return 0
}

attempt=0
while true; do
  attempt=$((attempt + 1))

  if [[ "$MAX_ATTEMPTS" -gt 0 && "$attempt" -gt "$MAX_ATTEMPTS" ]]; then
    echo "STATUS=FAILED" > "$STATUS_FILE"
    echo "FAILED_TIME=$(date -Is)" >> "$STATUS_FILE"
    echo "Attempts exhausted after $((attempt - 1)) tries" | tee -a "$RUN_LOG"
    exit 1
  fi

  echo "" | tee -a "$RUN_LOG"
  echo "===== ATTEMPT $attempt at $(date -Is) =====" | tee -a "$RUN_LOG"

  set +e
  "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    --fasta "$FASTA_PATH" \
    --layers $LAYERS \
    --num_steps "$NUM_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --device "$DEVICE" \
    --log_every "$LOG_EVERY" >> "$RUN_LOG" 2>&1
  cmd_exit=$?
  set -e

  echo "Attempt $attempt exit code: $cmd_exit" | tee -a "$RUN_LOG"

  if action_success; then
    echo "STATUS=SUCCESS" > "$STATUS_FILE"
    echo "SUCCESS_TIME=$(date -Is)" >> "$STATUS_FILE"
    echo "ATTEMPTS=$attempt" >> "$STATUS_FILE"
    echo "LOG=$RUN_LOG" >> "$STATUS_FILE"
    echo "CHECKPOINT_DIR=$CHECKPOINT_DIR" >> "$STATUS_FILE"
    echo "Training succeeded on attempt $attempt" | tee -a "$RUN_LOG"
    exit 0
  fi

  echo "Attempt $attempt did not produce valid outputs. Retrying in ${SLEEP_BETWEEN_ATTEMPTS}s..." | tee -a "$RUN_LOG"
  sleep "$SLEEP_BETWEEN_ATTEMPTS"
done
