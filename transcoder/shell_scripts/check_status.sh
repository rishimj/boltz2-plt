#!/bin/bash
# Quick status check for activation collection

echo "=== Boltz Transcoder Collection Status ==="
echo "Time: $(date)"
echo ""

if [ -f collection.pid ]; then
    PID=$(cat collection.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Process running (PID: $PID)"
        ps -p $PID -o pid,etime,cmd --no-headers
    else
        echo "✗ Process not running"
    fi
else
    echo "✗ No PID file found"
fi

echo ""
echo "=== Log (last 20 lines) ==="
tail -20 collection_fixed.log 2>/dev/null || echo "No log yet"

echo ""
echo "=== Output Files ==="
ls -lh real_activations/ 2>/dev/null || echo "Directory not created yet"

echo ""
echo "=== Quick Check Commands ==="
echo "  tail -f collection_fixed.log    # Watch log in real-time"
echo "  ps -p \$(cat collection.pid)      # Check if running"
echo "  kill \$(cat collection.pid)       # Stop collection"
