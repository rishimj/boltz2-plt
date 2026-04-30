<div align="center">
  <h1>Per-Layer Transcoder (PLT)</h1>
  <h3>Sparse Autoencoders for Boltz2 Protein Structure Prediction Interpretability</h3>
  <br>
  <a href="https://docs.google.com/presentation/d/e/2PACX-1vTLHgXL7Q1hIYD7Hdb7uVUBhktBvkhM-GIPkFLfeD9rVm3-nBfRNfwPm7mtGHoZHA/pub?start=false&loop=false&delayms=3000">📊 Presentation</a> • 
  <a href="transcoder/documentation">📖 Documentation</a> •
  <a href="https://github.com/jwohlwend/boltz">🧬 Boltz2 Repository</a>
</div>

---

## What is PLT?

**Per-Layer Transcoder (PLT)** is a sparse autoencoder framework for discovering interpretable features in Boltz2's neural network activations. By training independent transcoders on each layer of Boltz2's Pairformer trunk, we can:

- **Decode what the model learns**: Discover interpretable features like secondary structure patterns, contact predictions, and structural motifs
- **Track feature evolution**: Compare how representations change across network depth (layers 0, 8, 16, 24, 32, 40)
- **Achieve high reconstruction**: Reconstruct layer activations with sparse, human-understandable features (only 16 active per example)
- **Enable mechanistic interpretability**: Move beyond black-box neural networks toward understanding how structure prediction works

### Key Features

- ✅ **Sparse encoding**: 384-dimensional activations → 2048 sparse features (TopK=16 active)
- ✅ **Multi-layer training**: Independent transcoders for 6 key Pairformer layers
- ✅ **Online training**: Stream data from Boltz2 predictions during training
- ✅ **Deterministic**: Reproducible training with fixed seeds and cuDNN settings
- ✅ **Dead neuron resurrection**: Automatically revive inactive features
- ✅ **Unit-norm decoder weights**: Stable learned representations

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- Boltz2 (installed as base dependency)
- GPU with sufficient VRAM (24GB+ recommended)

### Setup

```bash
# Clone and navigate to the repository
git clone https://github.com/rishimj/boltz2-plt.git
cd boltz2-plt

# Create and activate virtual environment
python -m venv plt_env
source plt_env/bin/activate

# Install Boltz2 and dependencies
pip install boltz[cuda] -U
pip install torch pytorch-lightning einops einx numpy scipy

# Download Boltz2 checkpoint (required for activation collection)
wget https://model-gateway.boltz.bio/boltz2_conf.ckpt -O boltz2_checkpoint.ckpt
```

---

## Quick Start: Training PLT

### Step 1: Prepare Your Data

PLT works with protein FASTA sequences. The system will automatically:
1. Run Boltz2 predictions on the sequences
2. Collect activations from specified layers
3. Train sparse autoencoders on those activations

Prepare a directory with `.fasta` files:
```bash
# Example
examples/
├── protein_A.fasta
├── protein_B.fasta
└── protein_C.fasta
```

Or use the included multi-protein split dataset:
```bash
examples/multi_protein_split/
├── A.fasta  (protein chains)
├── B.fasta
├── C.fasta
└── ... (10 proteins total)
```

### Step 2: Collect Activations & Train PLT

```bash
cd transcoder
python universal_transcoder/train_online_multi_layer.py \
    --fasta-dir ../examples/multi_protein_split \
    --checkpoint-dir ./trained_plt_checkpoints \
    --layers 0 8 16 24 32 40 \
    --epochs 20 \
    --batch-size 10 \
    --learning-rate 0.001 \
    --seed 42
```

**Key Parameters:**
- `--fasta-dir`: Directory containing FASTA files
- `--layers`: Which Pairformer layers to train on (0, 8, 16, 24, 32, 40 recommended)
- `--epochs`: Training epochs (20-100 typical)
- `--batch-size`: Proteins per batch (smaller = less VRAM)
- `--learning-rate`: Learning rate (0.0001-0.001 typical)
- `--seed`: Seed for reproducibility (42 used in our experiments)

### Step 3: Validate & Analyze Results

```bash
python validation_scripts/validate_multi_layer.py \
    --checkpoint-dir ./trained_plt_checkpoints \
    --fasta-dir ../examples/multi_protein_split \
    --layers 0 8 16 24 32 40
```

This generates:
- **Reconstruction R²**: How well sparse features recover original activations
- **Sparsity metrics**: Mean active features, dead neuron count
- **Per-layer analysis**: Feature distribution and complexity trends

---

## Trained Models & Results

We've trained PLT models on a dataset of 10 diverse proteins with 20 epochs:

**Training Configuration:**
```yaml
Proteins: 10 (diverse chains: A-J from multi_protein_split)
Layers: 0, 8, 16, 24, 32, 40 (6 independent transcoders)
Epochs: 20
Batch Size: 10
Learning Rate: 0.001
Seed: 42
Model Dimensions:
  - Input (single representation): 384
  - Hidden (feature space): 2048
  - TopK sparsity: 16 active features
  - Pair representation output: 128
```

**Model Checkpoint Locations:**
```
transcoder/overnight_runs/online_train_split10_full_20260409_224033_checkpoints/
├── layer_00/model.pt
├── layer_08/model.pt
├── layer_16/model.pt
├── layer_24/model.pt
├── layer_32/model.pt
└── layer_40/model.pt
```

---

## Architecture

### Per-Layer Transcoder (PLT) Model

```
Input: x ∈ ℝ^384 (single representation from Boltz2 layer)
  ↓
[Normalize] x̂ = (x - μ) / σ
  ↓
[Encode] h = W_enc @ x̂ + b_enc  →  h ∈ ℝ^2048
  ↓
[TopK Sparsity] z = TopK(h, k=16)  →  only 16 values active
  ↓
[Decode] y = W_dec @ z + b_dec  →  y ∈ ℝ^128 (pair representation)
  ↓
[Denormalize] ŷ = y * σ + μ
  ↓
Output: ŷ ∈ ℝ^128 (reconstructed pair representation)

Loss: MSE(ŷ, target) + λ * L1_norm(z)
```

### Key Design Choices

| Component | Design | Rationale |
|-----------|--------|-----------|
| **Encoder** | Linear layer | Fast, interpretable; non-linearity from ReLU after |
| **TopK Sparsity** | Keep top 16 of 2048 | ~1% active (interpretable, prevents overfitting) |
| **Decoder** | Unit-norm weights | Prevents feature collapse and unbounded growth |
| **Dead Neuron Resurrection** | Reinitialize unused features | Prevents feature redundancy over training |
| **Learned centering bias** | Per-layer preprocessing | Allows features to learn relative to layer statistics |

---

## Documentation

For detailed technical information:

- **[PLT Architecture Guide](transcoder/documentation/PLT_ARCHITECTURE_GUIDE.md)** — Mathematical formulation, design principles, and implementation details
- **[Multi-Layer PLT Guide](transcoder/documentation/MULTI_LAYER_PLT_GUIDE.md)** — Full training pipeline, data flow, and component breakdown  
- **[Quickstart Guide](transcoder/documentation/QUICKSTART.md)** — Step-by-step setup and testing with small dataset
- **[Deep Reader Guide](transcoder/documentation/PLT_DEEP_READER_GUIDE.md)** — In-depth technical exploration

### Key Files

| File | Purpose |
|------|---------|
| `universal_transcoder/train_online_multi_layer.py` | Main training loop for multi-layer PLT |
| `collection_scripts/collect_multi_layer.py` | Collect Boltz2 activations from multiple layers |
| `validation_scripts/validate_multi_layer.py` | Evaluate trained PLT models |
| `transcoder/documentation/` | Full technical documentation |

---

## Project Structure

```
boltz2-plt/
├── transcoder/
│   ├── collection_scripts/      # Activation collection from Boltz2
│   ├── universal_transcoder/    # PLT training implementation
│   ├── validation_scripts/      # Evaluation and analysis
│   ├── overnight_runs/          # Trained checkpoints & logs
│   ├── documentation/           # Technical guides
│   │   ├── PLT_ARCHITECTURE_GUIDE.md
│   │   ├── MULTI_LAYER_PLT_GUIDE.md
│   │   ├── QUICKSTART.md
│   │   └── ...
│   └── shell_scripts/           # Training automation
├── examples/                    # Example protein data
│   ├── multi_protein_split/     # 10 diverse test proteins
│   └── ...
├── boltz2_checkpoint.ckpt       # Boltz2 model (required)
└── README.md                    # This file
```

---

## Dependencies: Boltz2

PLT operates on activations from **Boltz2**, a state-of-the-art biomolecular structure prediction model:

- **Paper**: [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://doi.org/10.1101/2025.06.14.659707)
- **Repository**: [jwohlwend/boltz](https://github.com/jwohlwend/boltz)
- **License**: MIT

Our PLT framework is architecture-agnostic and can be adapted to work with other structure prediction models.

---

## Citation

If you use PLT in your research, please cite:

```bibtex
@misc{plt_sparse_autoencoders,
  title={Per-Layer Transcoder: Sparse Autoencoders for Biomolecular Structure Prediction Interpretability},
  author={Manimaran, Rishi},
  year={2026},
  url={https://github.com/rishimj/boltz2-plt}
}
```

Also cite Boltz2 if you use its activations:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}
```

---

## License

MIT License — freely available for academic and commercial use.

---

## Troubleshooting

**"CUDA out of memory"**
```bash
# Reduce batch size
python universal_transcoder/train_online_multi_layer.py ... --batch-size 4
```

**"No module named 'boltz'"**
```bash
# Ensure environment is activated and boltz installed
source plt_env/bin/activate
pip install boltz[cuda] -U
```

**"Checkpoint not found"**
```bash
# Download Boltz2 checkpoint
wget https://model-gateway.boltz.bio/boltz2_conf.ckpt -O boltz2_checkpoint.ckpt
```

**"Error in compute_ptms"**
This is a known issue with certain protein sequences. The training continues and skips problematic proteins. Check the log file for details.

---

## Related Work

- **Sparse Autoencoders for Interpretability**: [Anthropic's SAE work](https://www.anthropic.com/research/scalable-interpretability-via-sparse-autoencoders)
- **Boltz2 Structure Prediction**: [Boltz GitHub](https://github.com/jwohlwend/boltz)
- **Neural Network Interpretability**: [Interpretability in the Wild (IITW)](https://arxiv.org/abs/2312.04782)

---

## Contact & Questions

For questions, suggestions, or contributions, please open an issue or contact the maintainers through GitHub.

**Project Links:**
- 📊 [PLT Presentation](https://docs.google.com/presentation/d/e/2PACX-1vTLHgXL7Q1hIYD7Hdb7uVUBhktBvkhM-GIPkFLfeD9rVm3-nBfRNfwPm7mtGHoZHA/pub?start=false&loop=false&delayms=3000)
- 📖 [Full Documentation](transcoder/documentation/)
- 🧬 [Boltz2 Repository](https://github.com/jwohlwend/boltz)
