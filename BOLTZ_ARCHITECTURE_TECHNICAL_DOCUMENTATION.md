# Boltz-1: Comprehensive Technical Documentation of Model Architecture

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture Summary](#model-architecture-summary)
3. [Trunk Module](#trunk-module)
4. [Diffusion Module](#diffusion-module)
5. [Confidence Module](#confidence-module)
6. [Validation Module](#validation-module)
7. [Hidden Dimensions and Parameter Counts](#hidden-dimensions-and-parameter-counts)
8. [Training Configuration](#training-configuration)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Implementation Details](#implementation-details)

---

## Overview

Boltz-1 is an open-source biomolecular structure prediction model that approaches AlphaFold3 accuracy. The model consists of four main modules:
- **Trunk Module**: Processes MSA and generates pairwise representations
- **Diffusion Module**: Predicts 3D atomic coordinates through denoising diffusion
- **Confidence Module**: Estimates prediction quality metrics
- **Validation Module**: Provides distogram predictions for validation

The complete model contains approximately **~1.3 billion parameters** distributed across these modules.

---

## Model Architecture Summary

```
Input Features
    ↓
[Input Embedder]
    ↓
[MSA Module] (4 blocks) ──→ Pair Representation (z)
    ↓
[Pairformer Module] (48 blocks) ──→ Single (s) & Pair (z) Representations
    ↓                                  ↓
[Distogram Head]              [Diffusion Module]
    ↓                                  ↓
Validation Loss              3D Coordinates
                                      ↓
                            [Confidence Module]
                                      ↓
                            Quality Metrics (pLDDT, PAE, PDE)
```

---

## Trunk Module

The Trunk Module is responsible for processing evolutionary information and generating rich single and pairwise representations. It consists of two sub-components: the MSA Module and the Pairformer Module.

### Architecture Overview

**Total Parameters**: ~520M
**Components**:
- Input Embedder with Atom Attention Encoder (3 layers, 4 heads)
- MSA Module (4 blocks)
- Pairformer Module (48 blocks)

### Input Embedder

The Input Embedder projects input features into token-level representations using an Atom Attention Encoder.

**Configuration**:
- Atom encoder depth: 3 layers
- Atom encoder heads: 4
- Atom single dimension: 128
- Atom pair dimension: 16

**Input Features**:
```python
s_input = concat([
    atom_features,           # From atom encoder
    res_type,               # Residue type (one-hot)
    profile,                # MSA profile
    deletion_mean,          # Deletion statistics
    pocket_feature          # Pocket conditioning
])
```

### MSA Module (MSA Processing Stack)

The MSA module processes multiple sequence alignments to extract evolutionary information and update pairwise representations.

**Configuration**:
- Number of blocks: 4
- MSA embedding dimension: 64
- Pair representation dimension: 128
- MSA dropout: 0.15
- Pair dropout: 0.25

#### MSA Block Architecture

Each MSA block consists of the following operations applied sequentially:

##### 1. Pair-Weighted Averaging (MSA Row Attention)

Updates MSA representations by aggregating information across sequences with pair-biased attention.

**Mathematical Formulation**:

```math
m_{si} \leftarrow m_{si} + \text{PWA}(m_s, z)
```

Where Pair-Weighted Averaging is computed as:

```math
\text{PWA}(m, z) = \text{LayerNorm}(m) \cdot W_o \left( \sum_h \text{Attention}_h \right)
```

For each head $h$:

```math
v_h = \text{LayerNorm}(m) \cdot W_{v,h}
```

```math
b_h = \text{LayerNorm}(z) \cdot W_{b,h}
```

```math
w_h = \text{softmax}(b_h - \text{mask} \cdot \infty)
```

```math
g_h = \sigma(\text{LayerNorm}(m) \cdot W_{g,h})
```

```math
\text{Attention}_h = g_h \odot (w_h \cdot v_h)
```

**Parameters**:
- Number of heads: 8
- Head width: 32
- Total attention dimension: 256

##### 2. MSA Transition

Feed-forward network applied to MSA representations.

```math
m_{si} \leftarrow m_{si} + \text{Transition}(m_{si})
```

```math
\text{Transition}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot \text{LayerNorm}(x))
```

- Hidden dimension: $4 \times \text{msa\_s} = 256$

##### 3. Outer Product Mean

Aggregates MSA information into pairwise representation.

```math
z_{ij} \leftarrow z_{ij} + \text{OuterProductMean}(m)
```

```math
\text{OuterProductMean}(m) = W_o \left( \frac{1}{N_{\text{seq}}} \sum_{s} (a_s \otimes b_s) \right)
```

Where:
```math
a_s = \text{LayerNorm}(m_s) \cdot W_a, \quad b_s = \text{LayerNorm}(m_s) \cdot W_b
```

**Parameters**:
- Hidden dimension: 32
- Output dimension: 128 (token_z)

##### 4. Triangle Multiplication (Outgoing)

Updates pair representation through multiplicative interactions along triangular edges.

```math
z_{ij} \leftarrow z_{ij} + \text{TriMulOut}(z)
```

```math
\text{TriMulOut}(z) = \text{LayerNorm}_{\text{out}}(g \odot [\text{LN}_{\text{in}}(z) \cdot W_a] \cdot [\text{LN}_{\text{in}}(z) \cdot W_b]^T) \cdot W_o
```

Where:
```math
g = \sigma(\text{LayerNorm}_{\text{in}}(z) \cdot W_g)
```

- Projects to dimension: $2 \times \text{token\_z} = 256$

##### 5. Triangle Multiplication (Incoming)

Similar to outgoing but aggregates information from different edge direction.

```math
z_{ij} \leftarrow z_{ij} + \text{TriMulIn}(z)
```

Computed similarly to TriMulOut but with transposed aggregation pattern.

##### 6. Triangle Attention (Starting Node)

Attention mechanism along triangle edges (row-wise attention on pair representation).

```math
z_{ij} \leftarrow z_{ij} + \text{TriAttnStart}(z)
```

```math
\text{TriAttn}(z) = \text{MultiHeadAttention}(\text{LayerNorm}(z))
```

**Parameters**:
- Number of heads: 4
- Head width: 32

##### 7. Triangle Attention (Ending Node)

Attention mechanism along triangle edges (column-wise attention on pair representation).

```math
z_{ij} \leftarrow z_{ij} + \text{TriAttnEnd}(z^T)^T
```

**Parameters**:
- Number of heads: 4
- Head width: 32

##### 8. Pair Transition

Feed-forward network applied to pair representation.

```math
z_{ij} \leftarrow z_{ij} + \text{Transition}(z_{ij})
```

- Hidden dimension: $4 \times \text{token\_z} = 512$

### Pairformer Module (Structure Module)

The Pairformer processes single and pair representations through deep attention blocks.

**Configuration**:
- Number of blocks: 48
- Token single dimension: 384
- Token pair dimension: 128
- Dropout: 0.25
- Activation checkpointing: True

#### Pairformer Block Architecture

Each Pairformer block contains:

##### 1. Triangle Operations (Same as MSA)

The pairformer applies the same triangle operations on the pair representation:
- Triangle Multiplication (Outgoing)
- Triangle Multiplication (Incoming)  
- Triangle Attention (Starting Node)
- Triangle Attention (Ending Node)
- Pair Transition

##### 2. Attention with Pair Bias

Updates single representation with pair-biased self-attention.

```math
s_i \leftarrow s_i + \text{AttentionPairBias}(s, z)
```

**Mathematical Formulation**:

```math
\text{AttentionPairBias}(s, z) = W_o \left( \sum_h g_h \odot \text{Attention}_h \right)
```

For each head $h$:

```math
q_h = \text{LayerNorm}(s) \cdot W_{q,h}
```

```math
k_h = \text{LayerNorm}(s) \cdot W_{k,h}
```

```math
v_h = \text{LayerNorm}(s) \cdot W_{v,h}
```

```math
b_h = \text{LayerNorm}(z) \cdot W_{b,h}
```

```math
\text{Attention}_h = \text{softmax}\left(\frac{q_h k_h^T}{\sqrt{d_h}} + b_h\right) \cdot v_h
```

```math
g_h = \sigma(\text{LayerNorm}(s) \cdot W_{g,h})
```

**Parameters**:
- Number of heads: 16
- Head dimension: 24 (384 / 16)
- Output dimension: 384

##### 3. Single Transition

Feed-forward network applied to single representation.

```math
s_i \leftarrow s_i + \text{Transition}(s_i)
```

- Hidden dimension: $4 \times \text{token\_s} = 1536$

### Recycling

The trunk supports recycling where outputs are fed back as inputs:

```math
s^{(t+1)} \leftarrow s^{(t)}_{\text{init}} + W_s \cdot \text{LayerNorm}(s^{(t)})
```

```math
z^{(t+1)} \leftarrow z^{(t)}_{\text{init}} + W_z \cdot \text{LayerNorm}(z^{(t)})
```

---

## Diffusion Module

The Diffusion Module predicts 3D atomic coordinates through a denoising diffusion process. It uses a U-Net-like architecture with atom-level and token-level processing.

**Total Parameters**: ~230M

### Architecture Components

#### 1. Atom Attention Encoder

Encodes atomic features and aggregates to token level.

**Configuration**:
- Depth: 3 layers
- Heads: 4
- Atoms per window (queries): 32
- Atoms per window (keys): 128
- Atom single dimension: 128
- Atom pair dimension: 16

**Output**: Token-level representation from atomic features

#### 2. Diffusion Token Transformer

Processes token-level representations with conditioning.

**Configuration**:
- Depth: 24 layers
- Heads: 16
- Token dimension: $2 \times \text{token\_s} = 768$
- Single condition dimension: 768
- Pair dimension: 128

##### Transformer Layer Architecture

Each transformer layer consists of:

###### Adaptive Layer Normalization (AdaLN)

Conditions the normalization on time and trunk features:

```math
\text{AdaLN}(a, s) = \sigma(W_{\text{scale}} \cdot \text{LN}(s)) \odot \text{LN}(a) + W_{\text{bias}} \cdot \text{LN}(s)
```

Where:
- $a$: token activations
- $s$: conditioning signal (time + trunk features)
- $\sigma$: sigmoid function

###### Attention with Pair Bias

```math
a_i \leftarrow a_i + \text{AttentionPairBias}(a, z, s)
```

Same formulation as in Pairformer but with AdaLN conditioning.

###### Conditioned Transition Block

```math
a_i \leftarrow a_i + \text{ConditionedTransition}(a_i, s_i)
```

```math
\text{ConditionedTransition}(a, s) = \sigma(W_p(s)) \odot W_b(\text{SwiGLU}(W_a(\text{AdaLN}(a, s))))
```

Where SwiGLU is:
```math
\text{SwiGLU}(x) = (x \cdot W_1) \odot \text{Swish}(x \cdot W_2)
```

- Expansion factor: 2
- Hidden dimension: $2 \times 768 = 1536$

#### 3. Atom Attention Decoder

Broadcasts token features to atoms and decodes atomic coordinates.

**Configuration**:
- Depth: 3 layers
- Heads: 4
- Atoms per window: 32/128

**Output**: Per-atom coordinate updates

### Conditioning

The diffusion module is conditioned on:

#### Single Conditioning

```math
s_{\text{cond}} = \text{Concat}(s_{\text{trunk}}, s_{\text{inputs}})
```

Processed through:
```math
s_{\text{diffusion}} = \text{Transition}(\text{Concat}(s_{\text{cond}}, \text{Fourier}(t)))
```

Where $\text{Fourier}(t)$ embeds the diffusion timestep:
- Fourier dimension: 256

#### Pairwise Conditioning

```math
z_{\text{diffusion}} = \text{Transition}(z_{\text{trunk}} + \text{RelPos})
```

- Relative position encoding dimension: 128
- Number of transition layers: 2

### Diffusion Process

#### Noise Schedule

The model uses a continuous-time diffusion process with the following schedule:

```math
\sigma(t) = \left( \sigma_{\max}^{1/\rho} + \frac{t}{T-1}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}) \right)^\rho
```

**Parameters**:
- $\sigma_{\min} = 0.0004$
- $\sigma_{\max} = 160.0$
- $\sigma_{\text{data}} = 16.0$
- $\rho = 7$
- Sampling steps: 200

#### Noise Distribution (Training)

Training noise levels are sampled from a log-normal distribution:

```math
\sigma \sim \sigma_{\text{data}} \cdot \exp(\mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2))
```

**Parameters**:
- $P_{\text{mean}} = -1.2$
- $P_{\text{std}} = 1.5$

#### Preconditioning Functions

The model uses EDM-style preconditioning:

**Skip connection weight**:
```math
c_{\text{skip}}(\sigma) = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}
```

**Output scaling**:
```math
c_{\text{out}}(\sigma) = \frac{\sigma \cdot \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}
```

**Input scaling**:
```math
c_{\text{in}}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}
```

**Time embedding**:
```math
c_{\text{noise}}(\sigma) = \frac{\log(\sigma / \sigma_{\text{data}})}{4}
```

#### Preconditioned Network

```math
\hat{x}_0 = c_{\text{skip}}(\sigma) \cdot x_t + c_{\text{out}}(\sigma) \cdot F_\theta(c_{\text{in}}(\sigma) \cdot x_t, c_{\text{noise}}(\sigma))
```

Where:
- $x_t$: noisy coordinates
- $\hat{x}_0$: predicted clean coordinates
- $F_\theta$: neural network (encoder + transformer + decoder)

#### Sampling Algorithm

**Second-order sampler** with stochastic noise injection:

For each timestep $t$ from $T$ to $0$:

1. **Add noise**:
   ```math
   \hat{t} = \sigma_t (1 + \gamma)
   ```
   ```math
   \epsilon \sim \mathcal{N}(0, (\hat{t}^2 - \sigma_t^2))
   ```
   ```math
   x_{\text{noisy}} = x_t + \epsilon
   ```

2. **Denoise**:
   ```math
   \hat{x}_0 = \text{Preconditioned}(x_{\text{noisy}}, \hat{t})
   ```

3. **Alignment** (optional):
   ```math
   x_{\text{noisy}} \leftarrow \text{WeightedRigidAlign}(x_{\text{noisy}}, \hat{x}_0)
   ```

4. **Update**:
   ```math
   x_{t-1} = x_{\text{noisy}} + s \cdot (\sigma_{t-1} - \hat{t}) \cdot \frac{x_{\text{noisy}} - \hat{x}_0}{\hat{t}}
   ```

**Parameters**:
- $\gamma_0 = 0.8$ (noise injection factor for $\sigma > \gamma_{\min}$)
- $\gamma_{\min} = 1.0$
- $s = 1.5$ (step scale)
- Noise scale: 1.0

### Loss Formulation

The diffusion loss combines MSE and smooth LDDT terms:

#### MSE Loss

```math
\mathcal{L}_{\text{MSE}} = \mathbb{E}_{\sigma, \epsilon} \left[ w(\sigma) \cdot \frac{\sum_{i} w_i m_i \|\hat{x}_{0,i} - x_{0,i}^*\|^2}{\sum_i w_i m_i} \right]
```

Where:
- $w(\sigma) = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma \cdot \sigma_{\text{data}})^2}$ (loss weighting)
- $w_i$: per-atom weights (higher for nucleotides and ligands)
- $m_i$: resolved atom mask
- $x_{0,i}^*$: aligned ground truth coordinates

**Atom type weights**:
- Protein/RNA/DNA: 1.0 / 6.0 / 6.0
- Ligands: 11.0

#### Smooth LDDT Loss

Auxiliary loss based on local distance difference test:

```math
\mathcal{L}_{\text{LDDT}} = \mathbb{E} \left[ \frac{\sum_{i,j} m_{ij} \cdot f(\Delta d_{ij})}{\sum_{i,j} m_{ij}} \right]
```

Where:
```math
\Delta d_{ij} = |d_{\text{pred},ij} - d_{\text{true},ij}|
```

```math
f(\Delta d) = \frac{1}{4}\sum_{t \in \{0.5, 1, 2, 4\}} \text{sigmoid}(t - \Delta d)
```

Distance mask:
```math
m_{ij} = \begin{cases}
1 & \text{if } d_{\text{true},ij} < 30Å \text{ (nucleotides)} \\
1 & \text{if } d_{\text{true},ij} < 15Å \text{ (others)} \\
0 & \text{otherwise}
\end{cases}
```

#### Total Diffusion Loss

```math
\mathcal{L}_{\text{diffusion}} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{LDDT}}
```

---

## Confidence Module

The Confidence Module estimates prediction quality without reference structures. It can optionally imitate the trunk architecture for improved performance.

**Total Parameters**: ~525M (with trunk imitation)

### Architecture Options

#### Standard Mode
- Pairformer stack (48 blocks, same as trunk)
- Confidence heads

#### Trunk Imitation Mode
- Full trunk replication:
  - Input Embedder
  - MSA Module (4 blocks)
  - Pairformer Module (48 blocks)
- Confidence heads

### Input Processing

The confidence module takes as input:

```math
s_{\text{conf}} = s_{\text{trunk}}
```

If using diffusion features:
```math
s_{\text{conf}} \leftarrow s_{\text{conf}} + W_{\text{diff}} \cdot \text{LayerNorm}(s_{\text{diffusion}})
```

Pair representation with predicted structure information:

```math
z_{\text{conf}} = z_{\text{trunk}} + W_{s,1}(s_{\text{inputs}}) \otimes W_{s,2}(s_{\text{inputs}}) + \text{DistBin}(d_{\text{pred}})
```

Where:
```math
\text{DistBin}(d) = \text{Embed}(\text{Discretize}(d, [2, 22]Å, 64 \text{ bins}))
```

Optional outer product:
```math
z_{\text{conf}} \leftarrow z_{\text{conf}} + W_o(W_{p,1}(s_{\text{inputs}}) \odot W_{p,2}(s_{\text{inputs}}))
```

### Confidence Heads

The model predicts four types of confidence metrics:

#### 1. pLDDT (per-token Local Distance Difference Test)

Predicts per-token accuracy on a 0-100 scale.

```math
\text{pLDDT}_{\text{logits}} = W_{\text{plddt}}(s) \in \mathbb{R}^{N \times 50}
```

**Binning**: 50 bins from 0 to 100

**Aggregation**:
```math
\text{pLDDT}_i = \sum_{b=1}^{50} P(b|s_i) \cdot \text{bin\_center}_b
```

**Complex-level metrics**:

pLDDT:
```math
\text{pLDDT}_{\text{complex}} = \frac{\sum_i m_i \cdot \text{pLDDT}_i}{\sum_i m_i}
```

Interface pLDDT (ipLDDT):
```math
\text{ipLDDT} = \frac{\sum_i w_i m_i \cdot \text{pLDDT}_i}{\sum_i w_i m_i}
```

Where interface weight:
```math
w_i = \begin{cases}
2 & \text{if token } i \text{ is ligand} \\
1 & \text{if token } i \text{ is interface residue} \\
0 & \text{otherwise}
\end{cases}
```

Interface residue: has contact ($d < 8Å$) with different chain

#### 2. PDE (Pairwise Distance Error)

Predicts error in pairwise distances.

```math
\text{PDE}_{\text{logits}} = W_{\text{pde}}(z + z^T) \in \mathbb{R}^{N \times N \times 64}
```

**Binning**: 64 bins from 0 to 32Å

**Aggregation**:
```math
\text{PDE}_{ij} = \sum_{b=1}^{64} P(b|z_{ij}) \cdot \text{bin\_center}_b
```

**Complex-level PDE**:

Averaged over predicted contacts:
```math
\text{PDE}_{\text{complex}} = \frac{\sum_{i,j} m_{ij} \cdot \text{PDE}_{ij}}{\sum_{i,j} m_{ij}}
```

Where contact mask from distogram:
```math
m_{ij} = P(\text{contact}|z_{ij}) \cdot \text{pad\_mask}_{ij}
```

**Interface PDE (iPDE)**:
```math
\text{iPDE} = \frac{\sum_{i,j} m_{ij} \cdot I(\text{chain}_i \neq \text{chain}_j) \cdot \text{PDE}_{ij}}{\sum_{i,j} m_{ij} \cdot I(\text{chain}_i \neq \text{chain}_j)}
```

#### 3. PAE (Predicted Aligned Error)

Predicts alignment error between residue pairs.

```math
\text{PAE}_{\text{logits}} = W_{\text{pae}}(z) \in \mathbb{R}^{N \times N \times 64}
```

**Binning**: 64 bins from 0 to 32Å

**Aggregation**:
```math
\text{PAE}_{ij} = \sum_{b=1}^{64} P(b|z_{ij}) \cdot \text{bin\_center}_b
```

**PTM (Predicted TM-score)**:

```math
\text{PTM} = \frac{1}{N_{\text{res}}} \sum_{ij} w_{ij} \cdot s(\text{PAE}_{ij})
```

Where:
```math
s(e) = \frac{1}{1 + (e/d_0)^2}, \quad d_0 = 1.24 \sqrt[3]{N_{\text{res}} - 15} - 1.8
```

**Interface PTM (iPTM)**:

```math
\text{iPTM} = \frac{\sum_{ij} w_{ij} \cdot I(\text{chain}_i \neq \text{chain}_j) \cdot s(\text{PAE}_{ij})}{\sum_{ij} w_{ij} \cdot I(\text{chain}_i \neq \text{chain}_j)}
```

Also computed for:
- Ligand iPTM (ligand-protein interfaces)
- Protein iPTM (protein-protein interfaces)
- Per-chain-pair iPTM

#### 4. Resolved Mask

Predicts whether each token will be resolved in the structure.

```math
\text{Resolved}_{\text{logits}} = W_{\text{resolved}}(s) \in \mathbb{R}^{N \times 2}
```

Binary classification (resolved vs unresolved)

### Confidence Loss

The confidence loss combines all four metrics:

```math
\mathcal{L}_{\text{confidence}} = \mathcal{L}_{\text{pLDDT}} + \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{resolved}} + \alpha_{\text{PAE}} \cdot \mathcal{L}_{\text{PAE}}
```

#### pLDDT Loss

```math
\mathcal{L}_{\text{pLDDT}} = -\frac{1}{N} \sum_i m_i \sum_b y_{ib}^{\text{pLDDT}} \log P(b|s_i)
```

Where $y_{ib}^{\text{pLDDT}}$ is computed from true LDDT:

1. Compute true LDDT for each token:
   ```math
   \text{LDDT}_i^{\text{true}} = \frac{\sum_j m_{ij} \sum_{t \in \{0.5,1,2,4\}} \mathbb{I}(|\Delta d_{ij}| < t)}{\sum_j m_{ij} \cdot 4}
   ```

2. Convert to bin distribution (soft binning with Gaussian)

#### PDE Loss

```math
\mathcal{L}_{\text{PDE}} = -\frac{1}{N^2} \sum_{ij} m_{ij} \sum_b y_{ijb}^{\text{PDE}} \log P(b|z_{ij})
```

Where $y_{ijb}^{\text{PDE}}$ computed from:
```math
\text{PDE}_{ij}^{\text{true}} = |d_{\text{pred},ij} - d_{\text{true},ij}|
```

#### PAE Loss

```math
\mathcal{L}_{\text{PAE}} = -\frac{1}{N^2} \sum_{ij} m_{ij} \sum_b y_{ijb}^{\text{PAE}} \log P(b|z_{ij})
```

Where $y_{ijb}^{\text{PAE}}$ computed from alignment error after optimal superposition.

#### Resolved Loss

```math
\mathcal{L}_{\text{resolved}} = -\frac{1}{N} \sum_i m_i [r_i \log P(\text{resolved}|s_i) + (1-r_i) \log P(\text{unresolved}|s_i)]
```

Where $r_i \in \{0, 1\}$ is the true resolved status.

**Configuration**:
- $\alpha_{\text{PAE}} = 1.0$

---

## Validation Module

The Validation Module provides an auxiliary distogram prediction head for training regularization and validation.

### Distogram Head

Predicts binned distance distribution between token representative atoms.

```math
\text{Distogram}_{\text{logits}} = W_{\text{disto}}(z + z^T) \in \mathbb{R}^{N \times N \times 64}
```

**Configuration**:
- Number of bins: 64
- Distance range: 2Å to 22Å
- Binning: Linear spacing

**Target computation**:

For each pair $(i,j)$:
1. Compute distance: $d_{ij} = \|x_i^{\text{repr}} - x_j^{\text{repr}}\|$
2. Assign to bin: $b_{ij} = \text{bin}(d_{ij})$
3. One-hot encode: $y_{ij} \in \{0,1\}^{64}$

### Distogram Loss

Cross-entropy loss over predicted distance bins:

```math
\mathcal{L}_{\text{distogram}} = -\frac{1}{N^2} \sum_{ij} m_{ij} \sum_b y_{ijb} \log P(b|z_{ij})
```

Where:
- $m_{ij}$: pair mask (excludes self-pairs and padding)
- $y_{ijb}$: one-hot target from true distances

**Loss weight**: 0.03 (relative to diffusion loss)

---

## Hidden Dimensions and Parameter Counts

### Representation Dimensions

| Representation | Dimension | Description |
|---|---|---|
| `atom_s` | 128 | Atom single representation |
| `atom_p` / `atom_z` | 16 | Atom pair representation |
| `token_s` | 384 | Token single representation |
| `token_z` / `token_p` | 128 | Token pair representation |
| `msa_s` | 64 | MSA representation |
| `fourier_dim` | 256 | Fourier time embedding |

### Transition Expansion Factors

| Module | Input Dim | Hidden Dim | Expansion |
|---|---|---|---|
| MSA Transition | 64 | 256 | 4x |
| Pair Transition (MSA) | 128 | 512 | 4x |
| Pair Transition (Pairformer) | 128 | 512 | 4x |
| Single Transition (Pairformer) | 384 | 1536 | 4x |
| Diffusion Token | 768 | 1536 | 2x |

### Attention Configurations

| Module | Heads | Head Dim | Total Dim |
|---|---|---|---|
| Pair-Weighted Averaging | 8 | 32 | 256 |
| Triangle Attention (MSA) | 4 | 32 | 128 |
| Pairformer Attention | 16 | 24 | 384 |
| Atom Encoder/Decoder | 4 | 32 | 128 |
| Token Transformer | 16 | 48 | 768 |

### Parameter Breakdown

#### Trunk Module: ~520M parameters

**Input Embedder** (~15M):
- Atom Attention Encoder: ~10M
  - 3 layers × 4 heads
  - Atom dim: 128/16

**MSA Module** (~25M):
- 4 blocks × ~6M per block
- Components per block:
  - Pair-Weighted Averaging: ~1.5M
  - Outer Product Mean: ~0.5M
  - Triangle operations: ~3M
  - Transitions: ~1M

**Pairformer Module** (~480M):
- 48 blocks × ~10M per block
- Components per block:
  - Attention with pair bias: ~3M
    - Q/K/V projections: 384×384 × 3 = ~0.5M
    - Output projection: ~0.5M
    - Pair bias: 128→16 = ~2K
  - Triangle operations: ~3M
    - Tri mult (2×): ~1.5M
    - Tri attn (2×): ~1.5M
  - Transitions: ~4M
    - Single: 384→1536→384 = ~2M
    - Pair: 128→512→128 = ~2M

#### Diffusion Module: ~230M parameters

**Atom Encoder** (~10M):
- 3 layers × ~3M

**Token Transformer** (~200M):
- 24 layers × ~8M per layer
- Per layer:
  - Attention: 768×768 × 4 (Q/K/V/O) = ~2.5M
  - Pair bias: ~16K
  - Conditioned transition: ~5M
    - SwiGLU: 768→1536 × 2 = ~2.5M
    - Projections: ~2.5M

**Atom Decoder** (~10M):
- 3 layers × ~3M

**Conditioning Networks** (~10M):
- Single/Pair conditioners: ~10M

#### Confidence Module: ~525M parameters

With trunk imitation:
- Full trunk replication: ~520M
- Confidence heads: ~5M
  - pLDDT head: 384→50 = ~20K
  - PDE head: 128→64 = ~8K
  - PAE head: 128→64 = ~8K
  - Resolved head: 384→2 = ~1K

#### Distogram Head: <1M parameters
- Linear projection: 128→64 = ~8K

### Total Model Parameters

| Module | Parameters |
|---|---|
| Trunk | ~520M |
| Diffusion | ~230M |
| Confidence (with trunk) | ~525M |
| Distogram | <1M |
| **Total** | **~1.275B** |

Without confidence trunk imitation: ~750M total

---

## Training Configuration

### Optimizer

**Adam optimizer** with specific hyperparameters:

```yaml
Optimizer: Adam
beta_1: 0.9
beta_2: 0.95
epsilon: 1e-8
```

### Learning Rate Schedule

**AlphaFold3-style schedule** with warmup and step decay:

```yaml
Schedule: AF3
base_lr: 0.0
max_lr: 0.0018
warmup_steps: 1000
start_decay_after: 50000 steps
decay_every: 50000 steps
decay_factor: 0.95
```

**Formulation**:

```python
if step < warmup_steps:
    lr = max_lr * step / warmup_steps
elif step < start_decay_after:
    lr = max_lr
else:
    decay_epochs = (step - start_decay_after) // decay_every
    lr = max_lr * (decay_factor ** decay_epochs)
```

### Loss Weights

```yaml
diffusion_loss_weight: 4.0
distogram_loss_weight: 0.03
confidence_loss_weight: 0.003
```

**Total loss**:
```math
\mathcal{L}_{\text{total}} = 4.0 \cdot \mathcal{L}_{\text{diffusion}} + 0.03 \cdot \mathcal{L}_{\text{distogram}} + 0.003 \cdot \mathcal{L}_{\text{confidence}}
```

### Exponential Moving Average (EMA)

```yaml
ema: true
ema_decay: 0.999
```

Model weights are updated with EMA for inference:

```math
\theta_{\text{ema}}^{(t+1)} = 0.999 \cdot \theta_{\text{ema}}^{(t)} + 0.001 \cdot \theta^{(t+1)}
```

### Training Hyperparameters

```yaml
# Data
max_tokens: 512
max_atoms: 4608
max_seqs: 2048
batch_size: 1
accumulate_grad_batches: 128
num_workers: 4

# Diffusion
recycling_steps: 3
sampling_steps: 200
diffusion_multiplicity: 16
diffusion_samples: 1

# Augmentation
coordinate_augmentation: true
alignment_reverse_diff: true

# Regularization
gradient_clip_val: 10.0
dropout_msa: 0.15
dropout_pair: 0.25
dropout_pairformer: 0.25

# Precision
precision: 32 (FP32)

# Cropping
min_neighborhood: 0
max_neighborhood: 40

# Filtering
min_chains: 1
max_chains: 300
max_resolution: 4.0Å
date_cutoff: "2021-09-30"
```

### Special Training Features

#### Recycling

Number of trunk recycling iterations:
- Training: 3
- Validation: 3

#### Diffusion Training

- Multiplicity: 16 (number of noisy samples per structure)
- Samples: 1 (number of denoising trajectories)
- Steps: 200

#### Symmetry Correction

Applied to ground truth for symmetric complexes:
```yaml
symmetry_correction: true
```

#### Pocket Conditioning

For binding pocket predictions:
```yaml
binder_pocket_conditioned_prop: 0.3
binder_pocket_cutoff: 6.0Å
binder_pocket_sampling_geometric_p: 0.3
```

### Validation Configuration

```yaml
recycling_steps: 3
sampling_steps: 200
diffusion_samples: 5
symmetry_correction: true
run_confidence_sequentially: true
```

---

## Performance Benchmarks

### CASP15 Performance

Boltz-1 demonstrates competitive performance approaching AlphaFold3 on the CASP15 benchmark.

**Metrics** (approximate from paper):

| Target Type | Metric | Boltz-1 | AlphaFold3 |
|---|---|---|---|
| Protein Monomers | LDDT-Cα | ~0.85 | ~0.87 |
| Protein Complexes | Interface LDDT | ~0.73 | ~0.76 |
| Protein-Nucleic Acid | LDDT | ~0.65 | ~0.68 |
| Protein-Ligand | LDDT | ~0.61 | ~0.67 |

### Test Set Performance

**Protein targets**:
- Average pLDDT: ~85
- High confidence (pLDDT > 90): ~70% of residues

**Protein-ligand complexes**:
- Ligand RMSD < 2Å: ~60% of predictions
- Interface pLDDT: ~75

**Multimer complexes**:
- Interface contact accuracy: ~80%
- iPTM: ~0.70

### Inference Speed

**Timing on modern GPU** (approximate):

| System Size | Tokens | Atoms | Time (GPU) |
|---|---|---|---|
| Small Protein | 100 | 800 | ~10s |
| Medium Complex | 300 | 2500 | ~45s |
| Large System | 500 | 4500 | ~120s |

**Configuration**: Single A100 GPU, 200 sampling steps, 5 samples

**Factors affecting speed**:
- Number of tokens (quadratic scaling for attention)
- Number of atoms (linear in encoder/decoder)
- Sampling steps (linear)
- Number of samples (linear)
- MSA depth (linear)

### Model Accuracy by Component

**Trunk Module**:
- Distogram accuracy (< 8Å): ~85%
- Contact prediction (< 8Å): ~90%

**Diffusion Module**:
- Cα RMSD (well-folded proteins): ~1.5Å
- All-atom RMSD: ~2.5Å
- Side-chain accuracy (χ1 within 20°): ~75%

**Confidence Module**:
- pLDDT correlation with true LDDT: ~0.85
- Ranking correlation (multiple models): ~0.90
- PAE accuracy: ~0.80

---

## Implementation Details

### Initialization Schemes

#### Weight Initialization

**Standard layers**:
- Linear weights: Lecun normal
  ```python
  std = sqrt(1.0 / fan_in)
  nn.init.normal_(weight, mean=0, std=std)
  ```

**Gating projections**:
- Zero initialization for residual connections
  ```python
  nn.init.zeros_(weight)
  ```

**Final layers** (before residuals):
- Scaled initialization
  ```python
  std = sqrt(1.0 / (3 * fan_in))
  nn.init.normal_(weight, mean=0, std=std)
  ```

**Layer norm**:
- Weight: ones
- Bias: zeros
  ```python
  nn.init.ones_(layer_norm.weight)
  nn.init.zeros_(layer_norm.bias)
  ```

### Activation Functions

| Location | Activation |
|---|---|
| Transitions (Trunk) | ReLU |
| Diffusion Transformer | SwiGLU |
| Gating | Sigmoid |
| Output heads | None (logits) |

**SwiGLU Definition**:
```python
def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)
```

### Memory Optimization Techniques

#### Activation Checkpointing

Enabled for:
- MSA blocks
- Pairformer blocks
- Diffusion transformer layers

**Configuration**:
```yaml
activation_checkpointing: true
offload_to_cpu: false
```

Saves ~60% of activation memory at ~20% compute cost.

#### Chunking Strategies

**Chunk sizes** (inference):

| Operation | Small Systems | Large Systems |
|---|---|---|
| Pair-Weighted Averaging | No chunking | Head-wise |
| Outer Product Mean | No chunking | 4-chunk |
| Triangle Attention | 512 tokens | 128 tokens |
| Transitions | No chunking | 64 tokens |

**Threshold**: 256 tokens

**Chunking implementation**:
```python
if num_tokens > chunk_size_threshold:
    # Process in chunks
    for i in range(0, num_tokens, chunk_size):
        chunk = input[i:i+chunk_size]
        output[i:i+chunk_size] = process(chunk)
else:
    # Process all at once
    output = process(input)
```

#### Mixed Precision

**Default**: FP32 throughout

**Selective FP32 casts**:
- Alignment operations
- SVD computations
- Determinant calculations

```python
with torch.autocast("cuda", enabled=False):
    result = align_coords(x.float(), y.float())
```

### Numerical Stability

**Masking**: Use large negative bias instead of -inf
```python
bias = (1 - mask) * -1e9  # Instead of -float('inf')
```

**Division**: Add small epsilon
```python
normalized = x / (count + 1e-7)
```

**Clamping**:
```python
num_mask = mask.sum().clamp(min=1)
```

### Padding and Masking

**Padding scheme**:
- Pad all tensors to `max_tokens` / `max_atoms`
- Use masks to exclude padding from computation

**Mask types**:
1. `token_pad_mask`: Valid tokens (B, N)
2. `atom_pad_mask`: Valid atoms (B, M)
3. `msa_mask`: Valid MSA sequences (B, S, N)
4. `atom_resolved_mask`: Resolved atoms in ground truth

**Mask application**:
```python
# In attention
attn_weights = softmax(scores + (1 - mask) * -1e9)

# In pooling
mean = (x * mask).sum() / mask.sum().clamp(min=1)

# In loss
loss = (error * mask).sum() / (mask.sum() + 1e-7)
```

### Batching Strategy

**Training**:
- Batch size: 1 structure
- Gradient accumulation: 128 steps
- Effective batch size: 128 structures

**Diffusion multiplicity**:
- Each structure generates 16 noisy samples
- Processed together for efficiency

**Validation**:
- Sequential processing of confidence samples
- Batch size: 1
- Generate 5 structure samples per input

### Data Pipeline

**Preprocessing**:
1. Parse mmCIF files
2. Generate/load MSAs
3. Tokenize sequences
4. Compute features
5. Apply cropping
6. Pad to max size

**Data augmentation**:
- Random coordinate augmentation (rotation + translation)
- MSA subsampling
- Cropping with random neighborhood size

**Feature caching**:
- MSAs cached on disk
- Features computed on-the-fly
- Symmetry information precomputed

### Model Compilation

**Optional compilation** (PyTorch 2.0+):
```yaml
compile_pairformer: false  # Can enable for speed
```

When enabled:
- Pairformer module compiled with torch.compile
- ~10-20% speedup
- Longer first-iteration compile time

### Inference Optimizations

**Model caching**:
```yaml
use_inference_model_cache: true
```

Cache invariant computations across diffusion steps:
- Pairwise conditioning (z)
- Trunk outputs

**Sequential confidence**:
```yaml
run_confidence_sequentially: true
```

Process confidence samples one at a time to reduce memory.

**Kernel fusion**:
- Triangle operations use optimized kernels when available
- Requires `cuequivariance` library

---

## References

1. **Boltz-1 Paper**: Wohlwend et al., 2024. "Boltz-1: Democratizing Biomolecular Interaction Modeling." bioRxiv. doi:10.1101/2024.11.19.624167

2. **AlphaFold3**: Abramson et al., 2024. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." Nature.

3. **EDM Preconditioning**: Karras et al., 2022. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS.

4. **AlphaFold2**: Jumper et al., 2021. "Highly accurate protein structure prediction with AlphaFold." Nature.

---

## Appendix: Mathematical Notation

### Symbols

| Symbol | Description |
|---|---|
| $N$ | Number of tokens |
| $M$ | Number of atoms |
| $S$ | Number of MSA sequences |
| $B$ | Batch size |
| $d$ | Dimension |
| $h$ | Number of attention heads |
| $s_i$ | Single representation for token $i$ |
| $z_{ij}$ | Pair representation for tokens $i,j$ |
| $m_{si}$ | MSA representation for sequence $s$, position $i$ |
| $x_i$ | 3D coordinates of atom $i$ |
| $\sigma$ | Noise level |
| $t$ | Timestep |
| $\theta$ | Model parameters |
| $\odot$ | Element-wise multiplication |
| $\otimes$ | Outer product |
| $\oplus$ | Concatenation |

### Functions

| Function | Description |
|---|---|
| $\text{LN}(\cdot)$ | Layer normalization |
| $\sigma(\cdot)$ | Sigmoid function |
| $\text{softmax}(\cdot)$ | Softmax function |
| $\text{ReLU}(\cdot)$ | Rectified linear unit |
| $\text{SwiGLU}(\cdot)$ | Swish-gated linear unit |
| $\mathcal{N}(\mu, \sigma^2)$ | Normal distribution |

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Model Version**: Boltz-1  
**Repository**: https://github.com/jwohlwend/boltz

