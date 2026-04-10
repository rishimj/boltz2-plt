#!/bin/bash
# Status checker for 10-protein multi-layer PLT pipeline

echo "=========================================="
echo "10-PROTEIN PLT PIPELINE STATUS"
echo "=========================================="
echo ""

# Check if process is running
PID=$(cat pipeline_10p.pid 2>/dev/null)
if [ -n "$PID" ]; then
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Pipeline is RUNNING (PID: $PID)"
        echo ""
        
        # Show what stage it's on
        if ps aux | grep -q "[c]ollect_multi_layer"; then
            echo "Current stage: COLLECTION (Step 1/3)"
            # Try to show progress
            if [ -f multi_layer_10p_logs/01_collection.log ]; then
                PROTEINS_DONE=$(grep -c "Processing protein" multi_layer_10p_logs/01_collection.log 2>/dev/null || echo "0")
                echo "  Proteins processed: ~$PROTEINS_DONE/10"
            fi
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
tail -40 full_pipeline_10proteins.log

echo ""
echo "=========================================="
echo "OUTPUT DIRECTORIES:"
echo "=========================================="
if [ -d multi_layer_10p_activations ]; then
    echo "Activations: $(du -sh multi_layer_10p_activations 2>/dev/null | cut -f1)"
    ls -d multi_layer_10p_activations/layer_* 2>/dev/null | wc -l | xargs -I {} echo "  Layers collected: {}/6"
fi
if [ -d multi_layer_10p_checkpoints ]; then
    echo "Checkpoints: $(du -sh multi_layer_10p_checkpoints 2>/dev/null | cut -f1)"
    ls -d multi_layer_10p_checkpoints/layer_* 2>/dev/null | wc -l | xargs -I {} echo "  Layers trained: {}/6"
fi

echo ""
echo "=========================================="
echo "MONITORING COMMANDS:"
echo "=========================================="
echo "Live updates:         tail -f full_pipeline_10proteins.log"
echo "Collection log:       tail -f multi_layer_10p_logs/01_collection.log"
echo "Training log:         tail -f multi_layer_10p_logs/02_training.log"
echo "Validation log:       tail -f multi_layer_10p_logs/03_validation.log"
echo "Kill pipeline:        kill \$(cat pipeline_10p.pid)"
echo ""
