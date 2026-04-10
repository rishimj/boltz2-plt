#!/bin/bash
# Quick reproducibility test for Boltz2 (no transcoder)

set -e

cd "$(dirname "$0")/.."
source ../boltz_env/bin/activate

FASTA="${1:-../examples/prot.fasta}"

echo "Testing Boltz2 reproducibility on: $FASTA"
echo ""
echo "This will run Boltz2 TWICE and verify outputs are identical."
echo "No transcoder involved - just a sanity check."
echo ""

python validation_scripts/test_boltz_reproducibility.py \
    --fasta "$FASTA" \
    --device cuda

echo ""
echo "Test complete!"
