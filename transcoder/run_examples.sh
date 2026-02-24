#!/bin/bash
# File: transcoder/run_examples.sh

cd /usr/scratch/rmanimaran8/boltz
source boltz_env/bin/activate

mkdir -p transcoder/example_predictions
mkdir -p transcoder/logs

echo "=== Processing Example Proteins ==="
echo "Started at: $(date)"

# Process each example
for yaml in examples/prot.yaml examples/multimer.yaml examples/cyclic_prot.yaml examples/pocket.yaml; do
    name=$(basename $yaml .yaml)
    echo "Processing $name..."
    boltz predict $yaml --out_dir transcoder/example_predictions/$name --devices 0 2>&1 | tee transcoder/logs/predict_$name.log
    echo "✓ Completed $name at $(date)"
done

echo "=== All predictions complete at $(date) ==="
echo "Processed structures available in: transcoder/example_predictions/*/processed/"
