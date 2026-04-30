#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/usr/scratch/rmanimaran8/boltz/transcoder"
RUNNER="$ROOT_DIR/shell_scripts/run_online_training_split10_full.sh"
PID_DIR="$ROOT_DIR/pid_files"
LAUNCH_TS="$(date +%Y%m%d_%H%M%S)"
PID_FILE="$PID_DIR/online_train_split10_full_${LAUNCH_TS}.pid"

mkdir -p "$PID_DIR"

nohup setsid "$RUNNER" >/dev/null 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"

echo "Launched overnight PLT training"
echo "PID: $PID"
echo "PID file: $PID_FILE"
echo "Runner: $RUNNER"
echo ""
echo "Check logs:"
echo "  ls -lt $ROOT_DIR/logs | head"
echo "Check status:"
echo "  cat \$ROOT_DIR/logs/*.status"
echo "Stop job:"
echo "  kill $PID"
