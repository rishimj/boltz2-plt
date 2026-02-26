#!/bin/bash
# Test the activation collection with small protein

cd /usr/scratch/rmanimaran8/boltz
source boltz_env/bin/activate
cd transcoder

echo "Testing activation collection with small protein..."
echo ""

# Check if checkpoint exists
CHECKPOINT="../boltz2_checkpoint.ckpt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Warning: Checkpoint not found at $CHECKPOINT"
    echo "Please download Boltz2 checkpoint first:"
    echo "  wget https://model-gateway.boltz.bio/boltz2_conf.ckpt -O boltz2_checkpoint.ckpt"
    echo ""
    echo "For now, you can try with any .ckpt file you have, or we'll skip model loading"
    read -p "Enter path to checkpoint (or press Enter to skip): " user_checkpoint
    if [ -n "$user_checkpoint" ]; then
        CHECKPOINT="$user_checkpoint"
    else
        echo "Skipping test - need checkpoint to run"
        exit 1
    fi
fi

# Run collection on test data
python collect_activations.py \
    --checkpoint "$CHECKPOINT" \
    --structures ../test_output/boltz_results_prot/processed/structures \
    --msa ../test_output/boltz_results_prot/processed/msa \
    --output pilot_activations \
    --max-structures 1 \
    --layer 47 \
    --device cuda \
    --recycling-steps 0

echo ""
echo "Collection test complete!"
echo "Check pilot_activations/ for output files"
