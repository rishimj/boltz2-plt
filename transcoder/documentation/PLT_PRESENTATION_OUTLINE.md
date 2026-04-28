# PLT for Boltz: Presentation Outline

This is a slide-ready presentation outline for explaining the work on training a PLT-style transcoder for Boltz.

Use this as:
- a talk script,
- a slide deck outline,
- or source material for PowerPoint / Google Slides / LaTeX Beamer.

---

## Slide 1: Title

**Training a PLT-Style Transcoder for Boltz2**

Subtitle:
- Interpreting Pairformer representations in a biomolecular structure model

Speaker notes:
- The goal of this project is to open up Boltz2’s internal representations and learn sparse, interpretable features from them.
- The main challenge is that Boltz is not a plain language model or vision transformer. It has coupled single-token and pairwise representations, so a standard transcoder does not fit directly.

---

## Slide 2: What Is Boltz?

**Boltz is a biomolecular foundation model for structure and affinity prediction**

Key points:
- Inputs: proteins, DNA, RNA, ligands, templates, structural constraints, MSAs
- Outputs: 3D structure, confidence metrics, and optionally binding affinity
- Core internal state:
  - single/token representation `s` of size 384
  - pair representation `z` of size 128
- Main computation happens in the Pairformer trunk

Suggested visual:
- Input biomolecules -> features -> trunk -> diffusion structure head -> confidence / affinity

Speaker notes:
- The central idea in Boltz is that it maintains two synchronized representations:
- `s`: per-residue state
- `z`: per-residue-pair state
- Those two streams are repeatedly updated together and eventually drive structure generation.

Source anchors:
- `src/boltz/model/models/boltz2.py`
- `src/boltz/model/layers/pairformer.py`
- `src/boltz/model/modules/trunkv2.py`

---

## Slide 3: Boltz2 Architecture At A High Level

**High-level pipeline**

1. Parse biomolecular input
2. Build token and pair features
3. Add MSA information
4. Run Pairformer blocks over `s` and `z`
5. Decode structure with diffusion
6. Produce confidence and optional affinity outputs

Suggested figure text:

```text
Input
  -> InputEmbedder
  -> Initial single/pair states
  -> MSA module
  -> Pairformer trunk
  -> Distogram + diffusion
  -> Confidence / affinity heads
```

Speaker notes:
- The interpretability work focuses on the trunk, not the final structure decoder.
- That is where Boltz builds most of its high-level structural representation.

---

## Slide 4: Boltz Internals Relevant To Interpretability

**Why the Pairformer trunk matters**

Inside each Pairformer layer:
- `transition_z` updates pair representation `z`
- attention updates single representation `s` using pair bias
- `transition_s` updates single representation `s`

Code-level summary from `PairformerLayer`:

```text
z = z + triangle updates + transition_z(z)
s = s + pair-biased attention(s, z)
s = s + transition_s(s)
```

Why this is the right hook point:
- clean module boundaries,
- strong nonlinear mixing,
- directly interpretable input/output tensors,
- repeated across many layers.

Speaker notes:
- The transition modules are MLP-like blocks inside each Pairformer layer.
- They are good mechanistic analysis targets because they are expressive but structurally localized.

---

## Slide 5: Why Train A Transcoder Here?

**Goal: learn sparse, interpretable features from Boltz activations**

We want features that answer questions like:
- which residues activate a structural concept?
- which latent features explain pairwise geometric relationships?
- how do internal concepts evolve across Pairformer layers?

Why sparsity helps:
- fewer active features per token,
- easier feature attribution,
- encourages specialization,
- better basis for mechanistic probing.

Speaker notes:
- Instead of treating Boltz as a black box, the transcoder gives us a dictionary of sparse features.
- Those features can then be analyzed against biological annotations, geometry, or intervention experiments.

---

## Slide 6: Background: Standard Sparse Transcoder

**Standard transcoder setup**

A normal sparse transcoder maps one representation to another:

$$
x \xrightarrow{\phi} z \xrightarrow{g} \hat{y}
$$

Typical equations:

$$
\tilde{x} = \operatorname{LN}(x) - b_{pre}
$$

$$
h = W_{enc}\tilde{x} + b_{enc}
$$

$$
z = \operatorname{TopK}(\operatorname{ReLU}(h), k)
$$

$$
\hat{y} = zW_{dec} + b_{dec}
$$

Standard loss:

$$
\mathcal{L}_{std} = \operatorname{MSE}(\hat{y}, y) + \lambda \Omega(z)
$$

If it is an autoencoder, then `y = x`.

Speaker notes:
- This works well when input and target live on the same axis or at least share compatible geometry.
- That assumption breaks in Boltz.

---

## Slide 7: Why A Normal Transcoder Does Not Work On Boltz

**Boltz violates the assumptions of a normal transcoder**

### Problem 1: Geometry mismatch

Single stream:

$$
s \in \mathbb{R}^{B \times N \times H}
$$

Pair stream:

$$
y \in \mathbb{R}^{B \times N^2 \times P}
$$

A normal transcoder expects:

$$
N \rightarrow N
$$

But Boltz requires:

$$
N \rightarrow N^2
$$

### Problem 2: One input is not enough

Each Pairformer layer has both:
- pre-transition single state `s_1`
- post-transition single state `s_2`

Both carry signal about the pair representation.

### Problem 3: Reconstruction alone is insufficient

Even good token-level reconstruction may fail to capture pairwise structural semantics.

### Problem 4: Sparse feature collapse

TopK models can accumulate dead neurons unless training explicitly addresses this.

Speaker notes:
- This is the core motivation for the extension.
- The problem is not just “train a bigger autoencoder.”
- The problem is that the target representation has different geometry and different semantics.

---

## Slide 8: Our Extension For Boltz

**PLT-style universal transcoder adapted to Boltz**

Inputs per batch:
- `s_1`: input single activations
- `s_2`: output single activations
- `y_1`: input pair activations
- `y_2`: output pair activations

Architecture:
- shared encoder from single space to sparse latent space,
- TopK sparse latent representation,
- decoder head 1 predicts pair-input side,
- decoder head 2 predicts pair-output side,
- consistency loss across pathways,
- auxiliary dead-neuron recovery path,
- token-to-pair expansion bridge.

Conceptually:

```text
s1 -> encoder -> sparse z1 -> decoder1/decoder2 -> y1_hat, y2_hat
s2 -> encoder -> sparse z2 -> decoder1/decoder2 -> y1_hat, y2_hat
```

Speaker notes:
- The important design choice is that both `s1` and `s2` share the same latent basis.
- That pressures the latent space to represent stable concepts rather than layer-local noise.

---

## Slide 9: Full Model Equations

**Encoder and sparse latent map**

For each token vector `s`:

$$
\mu(s) = \frac{1}{H}\sum_{j=1}^{H}s_j
$$

$$
\sigma(s) = \sqrt{\frac{1}{H}\sum_{j=1}^{H}(s_j - \mu(s))^2}
$$

$$
\operatorname{LN}(s) = \frac{s - \mu(s)}{\sigma(s) + \epsilon}
$$

$$
\tilde{s} = \operatorname{LN}(s) - b_{pre}
$$

$$
h = W_{enc}\tilde{s} + b_{enc}
$$

$$
z = \operatorname{TopK}(\operatorname{ReLU}(h), k)
$$

Compact latent function:

$$
f_{latent}(s)
=
\operatorname{TopK}
\left(
\operatorname{ReLU}
\left(
W_{enc}(\operatorname{LN}(s)-b_{pre}) + b_{enc}
\right),
k
\right)
$$

Speaker notes:
- This is the sparse feature extractor.
- Each token activates only `k` latent features, which makes the representation easier to interpret.

---

## Slide 10: Decoder Equations

**Dual decoder heads**

For pathway `i \in \{1,2\}`:

$$
z_i = f_{latent}(s_i)
$$

Token-level decoder outputs:

$$
\hat{y}^{token}_{1,i} = z_i W_{dec,1} + b_{pre,1}
$$

$$
\hat{y}^{token}_{2,i} = z_i W_{dec,2} + b_{pre,2}
$$

Then apply the implementation’s per-token scaling from the encoder normalization statistics.

Token-to-pair bridge:

$$
\hat{y}_{m,i} = \mathcal{E}(\hat{y}^{token}_{m,i}), \quad m \in \{1,2\}
$$

where `\mathcal{E}` expands token-indexed predictions into pair-indexed rows.

Speaker notes:
- This expansion operator is the practical bridge that makes training against pair targets possible.
- It is not the final ideal solution, but it allows a sparse token latent space to supervise against pair-space tensors.

---

## Slide 11: Loss Function

**Normalized reconstruction, consistency, and auxiliary residual fitting**

Normalized MSE:

$$
\operatorname{NMSE}(a,b)=\frac{\operatorname{MSE}(a,b)}{\operatorname{Var}(b)+\epsilon}
$$

### Reconstruction loss

$$
\mathcal{L}_{recon}=
\operatorname{NMSE}(\hat{y}_{1,1},y_1)+
\operatorname{NMSE}(\hat{y}_{2,1},y_2)+
\operatorname{NMSE}(\hat{y}_{1,2},y_1)+
\operatorname{NMSE}(\hat{y}_{2,2},y_2)
$$

### Consistency loss

$$
\mathcal{L}_{cons}=
\frac{\operatorname{MSE}(\hat{y}^{token}_{1,1},\hat{y}^{token}_{1,2})}{\operatorname{Var}(y_1)+\epsilon}
+
\frac{\operatorname{MSE}(\hat{y}^{token}_{2,1},\hat{y}^{token}_{2,2})}{\operatorname{Var}(y_2)+\epsilon}
$$

### Auxiliary residual loss

Residuals:

$$
r_{1,i}=y_1-\hat{y}_{1,i}, \quad r_{2,i}=y_2-\hat{y}_{2,i}
$$

Auxiliary predictions from dead-neuron activations:

$$
\hat{r}_{1,i}, \hat{r}_{2,i}
$$

Loss:

$$
\mathcal{L}_{aux}
=
\alpha
\sum_{i\in\{1,2\}}
\left[
\operatorname{NMSE}(\hat{r}_{1,i},r_{1,i})+
\operatorname{NMSE}(\hat{r}_{2,i},r_{2,i})
\right]
$$

### Total loss

$$
\mathcal{L}_{total} = \mathcal{L}_{recon} + \mathcal{L}_{cons} + \mathcal{L}_{aux}
$$

Speaker notes:
- `NMSE` matters because pair tensors can have heterogeneous scale across proteins and layers.
- The consistency term is what pushes both single-state pathways to decode the same pair concepts.

---

## Slide 12: Direct Comparison With A Standard Transcoder

**Standard sparse transcoder**

$$
z=\phi(x), \quad \hat{x}=g(z)
$$

$$
\mathcal{L}_{std}=\operatorname{MSE}(\hat{x},x)+\lambda\Omega(z)
$$

**Boltz extension**

$$
z_1=\phi(s_1), \quad z_2=\phi(s_2)
$$

$$
\hat{y}_{1,1}=\mathcal{E}(g_1(z_1)), \quad
\hat{y}_{2,1}=\mathcal{E}(g_2(z_1))
$$

$$
\hat{y}_{1,2}=\mathcal{E}(g_1(z_2)), \quad
\hat{y}_{2,2}=\mathcal{E}(g_2(z_2))
$$

$$
\mathcal{L}_{boltz}
=
\sum \text{reconstruction terms}
+
\sum \text{consistency terms}
+
\mathcal{L}_{aux}
$$

Comparison table:

| Aspect | Standard | Boltz extension |
|---|---|---|
| Input streams | 1 | 2 (`s_1`, `s_2`) |
| Decoder heads | 1 | 2 (`y_1`, `y_2`) |
| Target axis | `N` | `N^2` via expansion |
| Objective | Reconstruction | Reconstruction + consistency + aux recovery |
| Scale handling | Often raw MSE | NMSE |
| Dead neuron handling | Optional | Built in |

Speaker notes:
- This is the slide that most clearly justifies the novelty.
- The project is not just “run an SAE on Boltz.”
- It is a geometry-aware and pathway-aware transcoder extension.

---

## Slide 13: Training Pipeline In Practice

**How the system is trained**

1. Parse FASTA or Boltz inputs
2. Tokenize and featurize with Boltz2 tooling
3. Run Boltz2 forward pass
4. Hook `transition_s` and `transition_z` in selected Pairformer layers
5. Collect:
   - `input_s`
   - `output_s`
   - `input_z`
   - `output_z`
6. Train one layer-specific transcoder, or multiple online layer trainers

Important implementation pieces:
- `collection_scripts/collect_multi_layer.py`
- `universal_transcoder/universal_model.py`
- `universal_transcoder/train_online_multi_layer.py`

Speaker notes:
- This can be framed as a streamed interpretability pipeline.
- We do not need to pre-store huge corpora if we train online from collected batches.

---

## Slide 14: Why This Is PLT-Style

**Connection to PLT**

The current model is PLT-compatible because it includes:
- sparse TopK latent activation,
- per-layer training,
- unit-norm decoder columns,
- projected decoder gradients,
- dead-neuron tracking,
- auxiliary dead-feature recovery.

What makes it “PLT-style” rather than a generic sparse autoencoder:
- it is layer-specific,
- it is designed for mechanistic feature recovery,
- and it is structurally ready for broader per-layer interpretability workflows.

Speaker notes:
- If someone asks “is this exactly textbook PLT?”, the clean answer is:
- it is a PLT-compatible, Boltz-adapted implementation.

---

## Slide 15: Limitations And Current Gaps

**Current limitations**

1. Token-to-pair expansion is a coarse bridge
2. Pair-aware decoding is still implicit, not explicit
3. Some documentation in the repo is historical and not fully synced with current code
4. Interpretable features still require downstream biological probing

Important honest framing:
- low loss does not automatically imply semantic interpretability,
- but sparse latent structure creates the conditions for interpretability.

Speaker notes:
- This slide is useful because it makes the project look technically mature rather than overstated.

---

## Slide 16: Next Extensions

**Where this can go next**

### Pair-aware decoder

Instead of pure expansion:

$$
\hat{y}_{ij} = g(z_i, z_j, e_{ij})
$$

### Symmetry-aware objectives

$$
\mathcal{L}_{sym}=\operatorname{MSE}(\hat{y}_{ij}, \hat{y}_{ji})
$$

### Feature decorrelation

$$
\mathcal{L}_{decor}=\left\|\frac{1}{T}Z^\top Z-I\right\|_F^2
$$

### Cross-layer dictionary alignment

Train layer-specific transcoders with explicit cross-layer comparison.

Speaker notes:
- This shows that the current work is a serious first step, not an endpoint.

---

## Slide 17: Takeaway

**Main message**

- Boltz uses coupled single and pair representations, so a normal transcoder does not fit directly.
- We built a PLT-style extension that maps sparse token latents into pair-space supervision.
- The model uses dual pathways, dual decoders, consistency loss, NMSE scaling, and dead-neuron recovery.
- This turns Boltz internal activations into a usable interpretability object.

Short closing line:
- The contribution is a sparse, mechanistically motivated bridge from Boltz single activations to Boltz pair semantics.

---

## Optional Appendix Slide: Minimal Equation Summary

$$
z_i = f_{latent}(s_i)
$$

$$
f_{latent}(s)
=
\operatorname{TopK}
\left(
\operatorname{ReLU}
\left(
W_{enc}(\operatorname{LN}(s)-b_{pre}) + b_{enc}
\right),
k
\right)
$$

$$
\hat{y}^{token}_{1,i} = z_i W_{dec,1} + b_{pre,1}
$$

$$
\hat{y}^{token}_{2,i} = z_i W_{dec,2} + b_{pre,2}
$$

$$
\hat{y}_{m,i} = \mathcal{E}(\hat{y}^{token}_{m,i})
$$

$$
\mathcal{L}_{total} = \mathcal{L}_{recon} + \mathcal{L}_{cons} + \mathcal{L}_{aux}
$$

---

## Suggested Speaking Strategy

If you need this to be very explanatory, spend the most time on these four slides:
- Slide 4: what Pairformer is doing
- Slide 7: why a normal transcoder fails
- Slide 8: what your extension changes
- Slide 12: direct equation-level comparison

That sequence gives the audience:
- context,
- problem,
- method,
- justification.

---

## References For The Presentation

- `src/boltz/model/models/boltz2.py`
- `src/boltz/model/layers/pairformer.py`
- `src/boltz/model/modules/trunkv2.py`
- `transcoder/universal_transcoder/universal_model.py`
- `transcoder/universal_transcoder/train_online_multi_layer.py`
- `transcoder/collection_scripts/collect_multi_layer.py`
- `transcoder/documentation/BOLTZ_TRANSCODER_ARCHITECTURE_MATH.md`

