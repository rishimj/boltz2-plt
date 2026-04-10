# Boltz Architecture and Transcoder Extension: Deep Technical Reference

## 1. Purpose and Scope

This document gives a detailed, code-aligned explanation of:

1. Boltz2 architecture, with emphasis on the representational path used for interpretability.
2. Background and motivation for transcoders in mechanistic analysis.
3. Why a normal transcoder is not sufficient for Boltz representations.
4. The concrete extension used in this repository to make transcoder training work for Boltz.
5. Exact mathematical definitions for latent mapping, decoder mapping, and losses.
6. Direct equation-level comparison against standard sparse transcoders.
7. Practical implications, complexity, and limitations.

This is intended as a technical reference for future model and loss modifications.

---

## 2. Notation and Tensor Shapes

We use the following notation throughout.

| Symbol | Meaning | Typical value in current setup |
|---|---|---|
| $B$ | Batch size | 1 to 10 |
| $N$ | Number of residues (tokens) | Protein-dependent |
| $H$ | Single representation size | 384 |
| $P$ | Pair representation size | 128 |
| $D$ | Latent feature dimension | 2048 |
| $k$ | TopK active features per token | 16 |
| $k_{aux}$ | AuxK dead-neuron activation budget | up to 32 or bounded by dead count |

Key tensors:

1. Single stream tensors:
- $s_1, s_2 \in \mathbb{R}^{B\times N \times H}$

2. Pair stream tensors (flattened pair axis):
- $y_1, y_2 \in \mathbb{R}^{B\times N^2 \times P}$

3. Flattened forms used by the transcoder training step:
- $s_1^{flat}, s_2^{flat} \in \mathbb{R}^{(B\cdot N)\times H}$
- $y_1^{flat}, y_2^{flat} \in \mathbb{R}^{(B\cdot N^2)\times P}$

---

## 3. Boltz2 Architecture in Depth (Interpretability-Relevant Path)

### 3.1 High-Level Pipeline

Boltz2 can be viewed as:

1. Input feature encoding.
2. Trunk processing (MSA + pair/single interaction updates).
3. Structure generation (distogram and diffusion).
4. Confidence and optional affinity heads.

For transcoder training, we focus on trunk internals, especially pairformer transition modules.

### 3.2 Input Embedding and Core States

Boltz maintains two coupled representation spaces:

1. Single/token space: per-residue state, dimension $H=384$.
2. Pair space: residue-residue relational state, dimension $P=128$.

These two spaces are repeatedly updated, enabling Boltz to propagate local sequence context into global geometric constraints.

### 3.3 MSA and Pairformer Interaction

MSA features enrich pair states through sequence-family context, while pairformer blocks refine both token and pair streams. Even without full structural decoding, these intermediate states are rich and suitable for mechanistic probing.

### 3.4 Why Transition Blocks Are a Good Hook Point

The transition modules inside pairformer layers are MLP-like transforms that:

1. Concentrate nonlinear feature mixing.
2. Have clean input/output boundaries for hook capture.
3. Preserve direct correspondence to token/pair channels.

For interpretation, they offer a high signal-to-noise location: rich enough to contain semantics, but structured enough to map with sparse features.

---

## 4. Background: Transcoder Perspective for Mechanistic Analysis

### 4.1 From Autoencoder to Transcoder

A standard sparse autoencoder (SAE) aims to reconstruct the same space:

$$
x \rightarrow z \rightarrow \hat{x}
$$

A transcoder generalizes this to map one representation into another:

$$
x \rightarrow z \rightarrow \hat{y}
$$

This distinction matters for Boltz because explanatory targets are often in pair space, while the sparse basis is learned from single space.

### 4.2 Why Sparse Latents

Sparse latent units are useful because:

1. They encourage feature specialization.
2. They reduce overlap between concepts.
3. They make attribution easier when probing biological patterns.

With TopK, each residue activates exactly a small number of latent features, creating a controlled and comparable sparsity regime.

---

## 5. Why a Normal Transcoder Fails on Boltz Without Extension

### 5.1 Geometry Mismatch: Token Space vs Pair Space

A normal setup assumes same-axis reconstruction:

$$
x \in \mathbb{R}^{B\times N\times H}, \quad \hat{x}\in\mathbb{R}^{B\times N\times H}
$$

Boltz interpretability here requires:

$$
s \in \mathbb{R}^{B\times N\times H} \Rightarrow \hat{y}\in\mathbb{R}^{B\times N^2\times P}
$$

The $N \rightarrow N^2$ structural relation is not representable with a naive same-axis decoder.

### 5.2 Dynamics Mismatch: One State vs Two State Transition

In each pairformer layer, both pre-transition and post-transition single states carry signal. A one-input transcoder misses this transition-consistency constraint.

### 5.3 Objective Mismatch: Reconstruction Alone vs Mechanistic Agreement

Token-only reconstruction can produce low MSE yet fail to encode pair semantics. For Boltz, we need both:

1. Pair-target reconstruction quality.
2. Agreement between pathways generated from $s_1$ and $s_2$.

### 5.4 Capacity Utilization and Dead Features

TopK training can leave many latent units permanently inactive. Without dead-neuron mechanisms, interpretability degrades due to reduced effective dictionary size.

### 5.5 Data Collection Constraint

Boltz activation extraction can be expensive and sequence-dependent. An online streaming pipeline is operationally better than expecting large pre-collected offline activation corpora.

---

## 6. Current Extension: Universal Transcoder for Boltz

### 6.1 Architectural Design

Current design components:

1. One encoder from single space to latent space.
2. TopK activation for strict sparsity.
3. Two decoders:
  - Decoder 1 predicts pair-input side target ($y_1$).
  - Decoder 2 predicts pair-output side target ($y_2$).
4. Optional AuxK path using dead-neuron subset.

### 6.2 Input/Target Layout per Batch

Training step receives:

1. $s_1$: input single activations.
2. $s_2$: output single activations.
3. $y_1$: input pair activations.
4. $y_2$: output pair activations.

Both $s_1$ and $s_2$ pass through the same encoder and decoders, enforcing shared latent semantics.

### 6.3 Pair Expansion Bridge

Decoder outputs are token-indexed. Pair targets are pair-indexed. Current bridge expands token predictions across pair rows so losses can be computed in pair space.

Operationally, this acts like a broadcast from $B\cdot N$ to $B\cdot N^2$ rows. It is a practical compatibility layer between token-latent representation and pair supervision.

### 6.4 PLT-Compatible Stability Mechanics

The implementation includes:

1. Unit-norm decoder columns.
2. Gradient projection helper for manifold-compatible updates.
3. Dead-neuron tracking with inactivity counters.
4. Auxiliary residual fitting path for dead-feature recovery.

Even when not all options are used in every step, the model is structurally aligned with PLT-style training.

---

## 7. Full Mathematical Formulation

### 7.1 Layer Normalization and Centering

For each flattened token vector $s \in \mathbb{R}^{H}$:

$$
\mu(s) = \frac{1}{H}\sum_{j=1}^{H} s_j,
\quad
\sigma(s) = \sqrt{\frac{1}{H}\sum_{j=1}^{H}(s_j-\mu(s))^2}
$$

$$
\operatorname{LN}(s)=\frac{s-\mu(s)}{\sigma(s)+\epsilon}
$$

$$
  ilde{s}=\operatorname{LN}(s)-b_{pre}
$$

### 7.2 Encoder and Sparse Latent Function

$$
h = W_{enc}\tilde{s}+b_{enc}, \quad h\in\mathbb{R}^{D}
$$

$$
z = \operatorname{TopK}(\operatorname{ReLU}(h), k), \quad z\in\mathbb{R}^{D}
$$

Equivalent latent map:

$$
f_{latent}(s) = \operatorname{TopK}(\operatorname{ReLU}(W_{enc}(\operatorname{LN}(s)-b_{pre})+b_{enc}), k)
$$

### 7.3 Dual Decoder Heads

For pathway $i\in\{1,2\}$:

$$
\hat{y}_{1,i}^{token}=z_iW_{dec,1}+b_{pre,1}
$$

$$
\hat{y}_{2,i}^{token}=z_iW_{dec,2}+b_{pre,2}
$$

Implementation detail: outputs are scaled by input-wise $(\mu,\sigma)$ from the LN step. This introduces per-token scalar re-scaling before pair-space comparison.

### 7.4 Token-to-Pair Expansion Operator

Define expansion operator $\mathcal{E}:\mathbb{R}^{B\cdot N\times P}\rightarrow\mathbb{R}^{B\cdot N^2\times P}$ by row repetition over pair indices:

$$
\hat{y}_{m,i}=\mathcal{E}(\hat{y}_{m,i}^{token}), \quad m\in\{1,2\}
$$

So losses compare:

$$
\hat{y}_{1,i} \text{ vs } y_1, \quad \hat{y}_{2,i} \text{ vs } y_2
$$

### 7.5 NMSE Definition

$$
\operatorname{NMSE}(a,b)=\frac{\operatorname{MSE}(a,b)}{\operatorname{Var}(b)+\epsilon}
$$

This normalizes error by target variance to reduce scale sensitivity across layers and proteins.

### 7.6 Reconstruction Loss

$$
\mathcal{L}_{recon}=
\operatorname{NMSE}(\hat{y}_{1,1},y_1)+
\operatorname{NMSE}(\hat{y}_{2,1},y_2)+
\operatorname{NMSE}(\hat{y}_{1,2},y_1)+
\operatorname{NMSE}(\hat{y}_{2,2},y_2)
$$

Interpretation:

1. Both single pathways must explain both pair targets.
2. The shared latent basis is pressured to encode robust, transition-invariant features.

### 7.7 Consistency Loss

Current consistency is computed on token-level decoder outputs:

$$
\mathcal{L}_{cons}=
\frac{\operatorname{MSE}(\hat{y}_{1,1}^{token},\hat{y}_{1,2}^{token})}{\operatorname{Var}(y_1)+\epsilon}
+
\frac{\operatorname{MSE}(\hat{y}_{2,1}^{token},\hat{y}_{2,2}^{token})}{\operatorname{Var}(y_2)+\epsilon}
$$

Interpretation: if a concept is stable across $s_1$ and $s_2$, both pathways should decode similarly.

### 7.8 Auxiliary Dead-Neuron Residual Loss

Dead units are selected by inactivity threshold and used to fit residuals.

Residual definitions:

$$
r_{1,i}=y_1-\hat{y}_{1,i}, \quad r_{2,i}=y_2-\hat{y}_{2,i}
$$

Aux predictions from dead-neuron activations:

$$
\hat{r}_{1,i},\hat{r}_{2,i}
$$

Loss:

$$
\mathcal{L}_{aux}=\alpha\sum_{i\in\{1,2\}}\left[\operatorname{NMSE}(\hat{r}_{1,i},r_{1,i})+\operatorname{NMSE}(\hat{r}_{2,i},r_{2,i})\right]
$$

with $\alpha=1/32$ in the current trainer.

### 7.9 Total Loss

$$
\mathcal{L}_{total}=\mathcal{L}_{recon}+\mathcal{L}_{cons}+\mathcal{L}_{aux}
$$

---

## 8. Direct Equation-Level Comparison: Standard vs Boltz Extension

### 8.1 Mapping Definitions

Standard sparse transcoder:

$$
z=\phi(x), \quad \hat{x}=g(z)
$$

Boltz extension:

$$
z_1=\phi(s_1),\ z_2=\phi(s_2)
$$

$$
\hat{y}_{1,1}=\mathcal{E}(g_1(z_1)),\ \hat{y}_{2,1}=\mathcal{E}(g_2(z_1))
$$

$$
\hat{y}_{1,2}=\mathcal{E}(g_1(z_2)),\ \hat{y}_{2,2}=\mathcal{E}(g_2(z_2))
$$

### 8.2 Loss Comparison

Standard:

$$
\mathcal{L}_{std}=\operatorname{MSE}(\hat{x},x)+\lambda\Omega(z)
$$

Boltz extension:

$$
\mathcal{L}_{boltz}=\sum_{\text{4 recon terms}}\operatorname{NMSE}(\cdot)+\sum_{\text{2 consistency terms}}\operatorname{NMSE}(\cdot)+\mathcal{L}_{aux}
$$

### 8.3 Conceptual Comparison Table

| Category | Standard | Boltz Extension |
|---|---|---|
| Input streams | 1 | 2 ($s_1,s_2$) |
| Decoder heads | 1 | 2 ($y_1,y_2$) |
| Target axis | $N$ | $N^2$ (via $\mathcal{E}$) |
| Objective type | Reconstruction-centric | Reconstruction + pathway consistency + dead-feature residual fit |
| Scale handling | Raw MSE often sufficient | NMSE required for heterogeneous pair scales |
| Streaming support | Optional | Core workflow assumption |

---

## 9. Online Multi-Layer Training Workflow

### 9.1 Pipeline

1. Parse FASTA inputs.
2. Build Boltz features (tokenizer + featurizer + MSA parsing).
3. Run Boltz forward with activation collection hooks.
4. Pop per-layer batches from collector.
5. Train each layer-specific transcoder immediately (streaming update).

### 9.2 Determinism and Reproducibility Choices

Current trainer sets:

1. Python, NumPy, and PyTorch seeds.
2. Deterministic CUDA/cuDNN options.
3. Optional disabling of MSA subsampling in Boltz MSA module.

This improves repeatability for regression and metric comparison across runs.

### 9.3 Current Scaled Run Snapshot

From the summary artifact dated 2026-04-03:

| Layer | Final total | Final recon | Final consistency | Final aux | Dead neurons |
|---|---:|---:|---:|---:|---:|
| 0 | 544.66 | 514.39 | 30.26 | 0.00 | 0 |
| 8 | 1251.86 | 1214.32 | 37.54 | 0.00 | 0 |
| 16 | 1602.14 | 1553.83 | 48.31 | 0.00 | 0 |
| 24 | 1819.28 | 1763.08 | 56.20 | 0.00 | 0 |
| 32 | 1731.54 | 1679.49 | 52.05 | 0.00 | 0 |
| 40 | 1126.47 | 1079.74 | 46.73 | 0.00 | 0 |

Observed pattern:

1. Intermediate-deeper layers can be harder to fit than very early layers.
2. Consistency contributes a nontrivial but smaller fraction than reconstruction.
3. AuxK was 0 in this run, consistent with zero dead-neuron count at reported checkpoints.

---

## 10. Complexity and Scaling Considerations

### 10.1 Computational Cost Drivers

1. Boltz forward + activation extraction dominates wall time.
2. Pair tensors scale as $O(N^2\cdot P)$.
3. Transcoder forward scales roughly with $O(B\cdot N\cdot H\cdot D)$ for encoding and $O(B\cdot N\cdot D\cdot P)$ for decoding, plus expansion and pair-space losses.

### 10.2 Memory Pressure

Main memory pressure points:

1. Pair tensors of size $B\times N^2\times P$.
2. Multi-layer buffering if collector queues are not drained promptly.
3. Intermediate expanded prediction tensors for all four recon terms.

### 10.3 Practical Controls

1. Keep batch size modest when $N$ is large.
2. Use streamed updates instead of large offline stores.
3. Select a sparse subset of layers for broad scans, then refine around interesting layers.

---

## 11. Interpretation Strategy for Learned Features

Given latent matrix $Z\in\mathbb{R}^{(B\cdot N)\times D}$:

1. Activation frequency per feature:

$$
f_j = \frac{1}{B\cdot N}\sum_{t=1}^{B\cdot N} \mathbf{1}[Z_{t,j} > 0]
$$

2. Feature contribution vectors to each pair target:

$$
v_{1,j}=W_{dec,1}[j,:], \quad v_{2,j}=W_{dec,2}[j,:]
$$

3. Candidate mechanistic probes:
- Correlate feature activation maps with residue type classes.
- Correlate with secondary-structure annotations.
- Compare feature prevalence across layer index.
- Measure agreement/disagreement between decoder heads per feature.

This gives a path from sparse latent activity to biologically meaningful hypotheses.

---

## 12. Limitations and Known Gaps

1. Expansion operator is a coarse bridge from token predictions to pair supervision; richer pair-aware decoding could improve mechanistic fidelity.
2. Consistency normalization currently uses pair-target variance while comparing token-level predictions; this is practical but not the only possible normalization choice.
3. AuxK behavior depends on dead-neuron dynamics and thresholding; short runs may never trigger meaningful AuxK updates.
4. Feature interpretability still requires downstream probing and does not emerge automatically from low loss.

---

## 13. Suggested Next Mathematical Extensions

1. Pair-aware decoder with explicit $(i,j)$ conditioning:

$$
\hat{y}_{ij}=g(z_i, z_j, e_{ij})
$$

2. Symmetry-aware pair losses when appropriate:

$$
\mathcal{L}_{sym}=\operatorname{MSE}(\hat{y}_{ij},\hat{y}_{ji})
$$

3. Feature orthogonality / decorrelation regularization:

$$
\mathcal{L}_{decor}=\left\|\frac{1}{T}Z^\top Z-I\right\|_F^2
$$

4. Layer-coupled objective for cross-layer dictionary alignment.

---

## 14. Source Anchors (Code and Artifacts)

Core code files:

1. Boltz model wiring:
- src/boltz/model/models/boltz2.py

2. Trunk and MSA/pair modules:
- src/boltz/model/modules/trunk.py

3. Universal transcoder model:
- transcoder/universal_transcoder/universal_model.py

4. Online multi-layer trainer:
- transcoder/universal_transcoder/train_online_multi_layer.py

Referenced run artifact:

1. Scaled training summary:
- transcoder/minimal_test/online_train_scaled_20260403_002652_checkpoints/online_multi_layer_training_summary.json

This document should be updated whenever decoder geometry, consistency loss definition, or activation collection interfaces change.