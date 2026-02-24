#!/bin/bash
# Alternative approach: Use Boltz CLI to generate predictions, then collect activations

cd /usr/scratch/rmanimaran8/boltz
source boltz_env/bin/activate

echo "Step 1: Running Boltz prediction on test protein..."
echo "This will load the model and process the structure"
echo ""

# Use the existing test example
cd examples
boltz predict prot.yaml --out_dir ../transcoder/boltz_pred_output --recycling_steps 0 --override_use_previous_msa

echo ""
echo "Step 2: Check output..."
ls -lh ../transcoder/boltz_pred_output/

echo ""
echo "Note: To collect activations, we need to modify the Boltz source code"
echo "to add hooks during the prediction process."
echo ""
echo "The model checkpoint version mismatch prevents direct loading."
echo "Recommendation: Either update the Boltz codebase or use an older checkpoint."
