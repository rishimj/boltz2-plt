# Quick Start Guide: Testing with Small Protein

## Step 1: Get Boltz2 Checkpoint

You need a Boltz2 model checkpoint. Download it:

```bash
cd /usr/scratch/rmanimaran8/boltz
wget https://model-gateway.boltz.bio/boltz2_conf.ckpt -O boltz2_checkpoint.ckpt
```

Or use the HuggingFace mirror if the above fails:
```bash
wget https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt -O boltz2_checkpoint.ckpt
```

**Note:** This file is ~1-2GB, download may take a few minutes.

## Step 2: Test Activation Collection

Run the test script:

```bash
cd /usr/scratch/rmanimaran8/boltz/transcoder
./test_collection.sh
```

This will:
1. Load the Boltz2 model
2. Process the small test protein (`test_output/boltz_results_prot/processed/structures/prot.npz`)
3. Collect activations from layer 48
4. Save to `pilot_activations/batch_00000.npz`

## Step 3: Verify Activation Collection

Check the output:

```bash
ls -lh pilot_activations/
python -c "import numpy as np; d = np.load('pilot_activations/batch_00000.npz'); print('Keys:', list(d.keys())); [print(f'{k}: {d[k].shape}') for k in d.keys()]"
```

You should see:
- `input_s`: Shape [B, N, 384] - single representation inputs
- `output_s`: Shape [B, N, 384] - single representation outputs
- `input_z`: Shape [B, N*N, 128] - pair representation inputs (flattened)
- `output_z`: Shape [B, N*N, 128] - pair representation outputs (flattened)

## Step 4: Train Transcoder

Once activations are collected, train the transcoder:

```bash
python train.py \
    --activations pilot_activations \
    --checkpoints pilot_checkpoints \
    --log pilot_training_log.json \
    --epochs 10 \
    --batch-size 32
```

## Troubleshooting

### "No module named 'boltz'"
Make sure you're in the boltz_env:
```bash
source /usr/scratch/rmanimaran8/boltz/boltz_env/bin/activate
```

### "Checkpoint not found"
Download the Boltz2 checkpoint (see Step 1)

### "CUDA out of memory"
Try with CPU:
```bash
python collect_activations.py ... --device cpu
```

### "No activations were collected"
The hooks might not be triggering. Check:
1. Model architecture matches expected (has pairformer_module.layers)
2. Layer index is correct (47 for layer 48)
3. Model actually runs forward pass successfully

## Next Steps

After successful pilot:

1. **Scale up data collection**: Process more structures (100-1000)
2. **Download full dataset**: Get RCSB processed data (~250GB)
3. **Train longer**: 100+ epochs for full convergence
4. **Analyze results**: Check reconstruction quality and sparsity metrics
