#!/bin/bash
#
# Quick test: Collect activations from 1 protein to verify the pipeline works
#

set -e

echo "Quick test: Multi-layer activation collection"
echo "=============================================="
echo ""

# Activate environment
source ../boltz_env/bin/activate

# Run collection on just 1 protein
cd collection_scripts

python collect_multi_layer.py \
    --checkpoint /usr/scratch/rmanimaran8/boltz/.boltz_cache/boltz2_conf.ckpt \
    --fasta ../examples/prot.fasta \
    --output ../test_multi_layer_activations \
    --layers 0 8 16 24 32 40 \
    --max-proteins 1 \
    --device cuda

cd ..

echo ""
echo "Test complete! Check test_multi_layer_activations/"
echo ""
echo "Collected batches:"
for layer in 00 08 16 24 32 40; do
    dir="test_multi_layer_activations/layer_$layer"
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir"/*.npz 2>/dev/null | wc -l)
        echo "  Layer $layer: $count batches"
    else
        echo "  Layer $layer: No data"
    fi
done
