<div align="center">
  <h1>Boltz2 PLT</h1>
  <p><b>Per-Layer Transcoders for interpreting Boltz2 protein structure prediction</b></p>
  <a href="https://boltz2-plt.20.25.227.252.sslip.io/"><b>Live Demo</b></a> ·
  <a href="https://docs.google.com/presentation/d/e/2PACX-1vTLHgXL7Q1hIYD7Hdb7uVUBhktBvkhM-GIPkFLfeD9rVm3-nBfRNfwPm7mtGHoZHA/pub?start=false&loop=false&delayms=3000"><b>Presentation</b></a> ·
  <a href="transcoder/documentation"><b>Docs</b></a>
</div>

---

Boltz2 is a state-of-the-art biomolecular structure prediction model. This project opens up its
Pairformer trunk with **sparse Per-Layer Transcoders (PLTs)**: small models that rewrite each
layer's activations as a handful of human-readable features, so you can see what the network is
computing at every depth.

- **Sparse, interpretable features:** each 384-d activation is expressed with just **16 of 2,048** learned features (0.8% active).
- **Full-depth coverage:** independent transcoders for Pairformer layers **0, 8, 16, 24, 32 and 40**.
- **End-to-end pipeline:** streams activations straight from live Boltz2 predictions, trains, validates, and can splice the PLT back into the model's forward pass.
- **Reproducible by design:** fixed seeds, deterministic cuDNN, and a verified deterministic Boltz2 baseline.

## Results

Trained on 10 diverse protein chains (all 6 layers in ~4.3 GPU-hours) and benchmarked per layer:

| Pairformer layer | 0 | 8 | 16 | 24 | 32 | 40 |
|---|---|---|---|---|---|---|
| Reconstruction R² | **0.95** | 0.74 | 0.72 | 0.71 | 0.71 | 0.74 |
| Active features | 16 / 2048 | 16 / 2048 | 16 / 2048 | 16 / 2048 | 16 / 2048 | 16 / 2048 |

All six transcoders trained successfully with unit-norm decoders and zero failed layers.

## How it was validated

1. **Deterministic baseline:** two identical Boltz2 forward passes must match to within 1e-6 on activations and 1e-5 Å on structure ([`deterministic_baseline.py`](transcoder/scripts/deterministic_baseline.py)).
2. **Reconstruction benchmark:** per-layer MSE, RMSE and R² plus sparsity and decoder-norm checks ([`validate_multi_layer.py`](transcoder/universal_transcoder/validate_multi_layer.py)).
3. **In-model verification:** the trained PLT replaces the real layer inside Boltz2, and the predicted structure is compared with the original by RMSD and pLDDT ([`verify_plt_structure.py`](transcoder/scripts/verify_plt_structure.py)).

## Quick start

Requires Python 3.10+, PyTorch 2 and a CUDA GPU (24 GB+ recommended).

```bash
git clone https://github.com/rishimj/boltz2-plt.git && cd boltz2-plt
python -m venv .venv && source .venv/bin/activate
pip install "boltz[cuda]" -U && pip install pytorch-lightning einops einx scipy
wget https://model-gateway.boltz.bio/boltz2_conf.ckpt -O boltz2_conf.ckpt
cd transcoder
```

**Train** transcoders on all six layers, streaming activations from Boltz2:

```bash
python universal_transcoder/train_online_multi_layer.py \
  --checkpoint ../boltz2_conf.ckpt --fasta ../examples/multi_protein_split \
  --layers 0 8 16 24 32 40 --checkpoint_dir plt_checkpoints --seed 42
```

**Benchmark** reconstruction quality:

```bash
python collection_scripts/collect_multi_layer.py \
  --checkpoint ../boltz2_conf.ckpt --fasta ../examples/multi_protein_split --output activations
python universal_transcoder/validate_multi_layer.py \
  --checkpoint_dir plt_checkpoints --data_dir activations
```

**Verify** inside Boltz2 by swapping a layer for its transcoder:

```bash
python scripts/verify_plt_structure.py \
  --checkpoint ../boltz2_conf.ckpt --fasta ../examples/prot.fasta --plt-checkpoints plt_checkpoints --layers 0 --output verification
```

## Architecture

```
Boltz2 layer activation (384) → normalize → encoder → TopK (16 of 2048) → decoder → reconstruction
```

Trained with reconstruction and consistency losses, an auxiliary TopK loss that revives dead
features, and unit-norm decoder weights. Full details are in the
[architecture guide](transcoder/documentation/PLT_ARCHITECTURE_GUIDE.md).

## Built on

[Boltz2](https://github.com/jwohlwend/boltz) (MIT). Released under the MIT License.
