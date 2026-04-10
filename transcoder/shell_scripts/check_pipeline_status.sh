#!/bin/bash
# Quick status checker for the multi-layer PLT pipeline

echo "=========================================="
echo "MULTI-LAYER PLT PIPELINE STATUS"
echo "=========================================="
echo ""

# Check if process is running
PID=$(cat pipeline.pid 2>/dev/null)
if [ -n "$PID" ]; then
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Pipeline is RUNNING (PID: $PID)"
        echo ""
        
        # Show what stage it's on
        if ps aux | grep -q "[c]ollect_multi_layer"; then
            echo "Current stage: COLLECTION (Step 1/3)"
        elif ps aux | grep -q "[t]rain_multi_layer"; then
            echo "Current stage: TRAINING (Step 2/3)"
        elif ps aux | grep -q "[v]alidate_multi_layer"; then
            echo "Current stage: VALIDATION (Step 3/3)"
        else
            echo "Current stage: UNKNOWN (check logs)"
        fi
    else
        echo "✗ Pipeline is NOT running (may have completed or failed)"
    fi
else
    echo "✗ No pipeline PID found"
fi

echo ""
echo "=========================================="
echo "RECENT LOG OUTPUT (last 40 lines):"
echo "=========================================="
tail -40 full_pipeline.log

echo ""
echo "=========================================="
echo "MONITORING COMMANDS:"
echo "=========================================="
echo "Watch live updates:       tail -f full_pipeline.log"
echo "Check collection logs:    tail -f multi_layer_logs/01_collection.log"
echo "Check training logs:      tail -f multi_layer_logs/02_training.log"
echo "Check validation logs:    tail -f multi_layer_logs/03_validation.log"
echo "Kill pipeline:            kill \$(cat pipeline.pid)"
echo ""
