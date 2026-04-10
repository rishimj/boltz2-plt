#!/bin/bash
# Quick validation test script

set -e  # Exit on error

cd "$(dirname "$0")/.."
source ../boltz_env/bin/activate

echo "==========================================="
echo "Boltz2 Pipeline Validation Test"
echo "==========================================="
echo ""

# Default paths
FASTA="${1:-../examples/prot.fasta}"
OUTPUT_DIR="${2:-validation_output}"

echo "Input FASTA: $FASTA"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Test 1: Reproducibility only (no transcoder)
echo "=========================================="
echo "TEST 1: Reproducibility Check"
echo "=========================================="
echo "This verifies Boltz2 produces identical outputs on repeated runs"
echo ""

python validation_scripts/verify_pipeline_reproducibility.py \
    --fasta "$FASTA" \
    --no-transcoder \
    --output "$OUTPUT_DIR/reproducibility"

echo ""
echo "✓ Reproducibility test complete"
echo ""

# Test 2: With transcoder intervention
echo "=========================================="
echo "TEST 2: Transcoder Intervention"
echo "=========================================="
echo "This tests if transcoder preserves functional information"
echo ""

if [ -f "universal_transcoder/checkpoints/universal_transcoder_final.pt" ]; then
    python validation_scripts/verify_pipeline_reproducibility.py \
        --fasta "$FASTA" \
        --transcoder universal_transcoder/checkpoints/universal_transcoder_final.pt \
        --output "$OUTPUT_DIR/intervention"
    
    echo ""
    echo "✓ Intervention test complete"
else
    echo "⚠️ Transcoder checkpoint not found - skipping intervention test"
    echo "   Expected: universal_transcoder/checkpoints/universal_transcoder_final.pt"
fi

echo ""
echo "==========================================="
echo "ALL TESTS COMPLETE"
echo "==========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/*/validation_results_*.json"
echo ""
