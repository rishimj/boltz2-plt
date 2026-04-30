#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/usr/scratch/rmanimaran8/boltz/transcoder"
PYTHON_BIN="/usr/scratch/rmanimaran8/boltz/boltz_env/bin/python3"
TRAIN_SCRIPT="$ROOT_DIR/universal_transcoder/train_online_multi_layer.py"
DATASET_DIR="/usr/scratch/rmanimaran8/boltz/examples/multi_protein_split"
LOG_DIR="$ROOT_DIR/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="online_train_split10_full_${RUN_TS}"
CHECKPOINT_DIR="$ROOT_DIR/overnight_runs/${RUN_NAME}_checkpoints"
RUN_LOG="$LOG_DIR/${RUN_NAME}.log"
STATUS_FILE="$LOG_DIR/${RUN_NAME}.status"

LAYERS=(0 8 16 24 32 40)
EPOCHS=20
MAX_PROTEINS=10
BATCH_SIZE=10
LOG_EVERY=20
DEVICE="cuda"
SEED=42
MAX_ATTEMPTS=0
SLEEP_BETWEEN_ATTEMPTS=30

mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

mapfile -t FASTA_FILES < <(find "$DATASET_DIR" -maxdepth 1 \( -name '*.fasta' -o -name '*.fa' \) | sort)
if [[ "${#FASTA_FILES[@]}" -eq 0 ]]; then
  echo "No FASTA files found in $DATASET_DIR" >&2
  exit 1
fi

if [[ "$MAX_PROTEINS" -gt 0 && "${#FASTA_FILES[@]}" -gt "$MAX_PROTEINS" ]]; then
  EFFECTIVE_PROTEINS="$MAX_PROTEINS"
else
  EFFECTIVE_PROTEINS="${#FASTA_FILES[@]}"
fi

NUM_STEPS=$((EPOCHS * EFFECTIVE_PROTEINS))

{
  echo "START_TIME=$(date -Is)"
  echo "RUN_NAME=$RUN_NAME"
  echo "DATASET_DIR=$DATASET_DIR"
  echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
  echo "LAYERS=${LAYERS[*]}"
  echo "EPOCHS=$EPOCHS"
  echo "MAX_PROTEINS=$MAX_PROTEINS"
  echo "EFFECTIVE_PROTEINS=$EFFECTIVE_PROTEINS"
  echo "NUM_STEPS=$NUM_STEPS"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "SEED=$SEED"
} | tee -a "$RUN_LOG"

{
  echo "STATUS=RUNNING"
  echo "START_TIME=$(date -Is)"
  echo "RUN_NAME=$RUN_NAME"
  echo "RUN_LOG=$RUN_LOG"
  echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
  echo "DATASET_DIR=$DATASET_DIR"
  echo "EPOCHS=$EPOCHS"
  echo "NUM_STEPS=$NUM_STEPS"
} > "$STATUS_FILE"

action_success() {
  local summary_file="$CHECKPOINT_DIR/online_multi_layer_training_summary.json"
  [[ -f "$summary_file" ]] || return 1

  for layer in "${LAYERS[@]}"; do
    local ckpt="$CHECKPOINT_DIR/layer_$(printf '%02d' "$layer")/universal_transcoder_final.pt"
    [[ -f "$ckpt" ]] || return 1
  done

  return 0
}

attempt=0
while true; do
  attempt=$((attempt + 1))

  if [[ "$MAX_ATTEMPTS" -gt 0 && "$attempt" -gt "$MAX_ATTEMPTS" ]]; then
    {
      echo "STATUS=FAILED"
      echo "FAILED_TIME=$(date -Is)"
      echo "ATTEMPTS=$((attempt - 1))"
      echo "RUN_LOG=$RUN_LOG"
      echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
    } > "$STATUS_FILE"
    echo "Attempts exhausted after $((attempt - 1)) tries" | tee -a "$RUN_LOG"
    exit 1
  fi

  echo "" | tee -a "$RUN_LOG"
  echo "===== ATTEMPT $attempt at $(date -Is) =====" | tee -a "$RUN_LOG"

  set +e
  "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    --fasta "$DATASET_DIR" \
    --layers "${LAYERS[@]}" \
    --num_steps "$NUM_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --device "$DEVICE" \
    --log_every "$LOG_EVERY" \
    --max-proteins "$MAX_PROTEINS" \
    --seed "$SEED" >> "$RUN_LOG" 2>&1
  cmd_exit=$?
  set -e

  echo "Attempt $attempt exit code: $cmd_exit" | tee -a "$RUN_LOG"

  if action_success; then
    {
      echo "STATUS=SUCCESS"
      echo "SUCCESS_TIME=$(date -Is)"
      echo "ATTEMPTS=$attempt"
      echo "RUN_LOG=$RUN_LOG"
      echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
      echo "DATASET_DIR=$DATASET_DIR"
      echo "EPOCHS=$EPOCHS"
      echo "NUM_STEPS=$NUM_STEPS"
    } > "$STATUS_FILE"
    echo "Training succeeded on attempt $attempt" | tee -a "$RUN_LOG"
    exit 0
  fi

  echo "Attempt $attempt did not produce valid outputs. Retrying in ${SLEEP_BETWEEN_ATTEMPTS}s..." | tee -a "$RUN_LOG"
  sleep "$SLEEP_BETWEEN_ATTEMPTS"
done
