# PLT for Boltz: Deep Reader Guide

This document is meant to help you deeply understand the project before you make a presentation.

It is intentionally more explanatory than a slide deck. The goal is:
- understand what Boltz is doing internally,
- understand what a transcoder or PLT is trying to do,
- understand exactly why a normal transcoder does not fit Boltz directly,
- understand the concrete extension implemented in this repository,
- understand the math well enough to explain it cleanly.

You should read this as a study document first and a presentation source second.

---

## 1. What This Project Is Actually About

At the highest level, your project is an interpretability project on top of the Boltz2 model.

Boltz2 is a biomolecular foundation model. It takes biological molecules such as proteins and ligands, builds internal representations, and predicts:
- 3D structure,
- confidence scores,
- and optionally binding affinity.

Your project is not trying to improve Boltz2’s primary prediction objective directly.

Instead, your project is trying to answer a mechanistic question:

**What kinds of internal features does Boltz2 learn inside its Pairformer trunk?**

More concretely:
- when Boltz2 processes a protein or protein-ligand complex,
- it builds internal hidden states,
- and we want to factor those hidden states into sparse, interpretable features.

That is where the transcoder comes in.

The central idea is:
- collect internal activations from Boltz2,
- train a sparse model on those activations,
- and use the sparse latent units as a more interpretable feature basis.

This is conceptually similar to sparse autoencoder or transcoder work in language-model interpretability, but Boltz introduces a structural complication:

Boltz does not just have one sequence representation.

It has:
- a **single/token representation** `s`,
- and a **pair representation** `z`.

Those two spaces interact constantly. That interaction is the main reason a standard transcoder is not enough.

---

## 2. What Boltz Is, In Plain Language

Boltz is a family of models for biomolecular interaction prediction.

From the repo and docs, the main user-facing capabilities are:
- predicting biomolecular structures,
- handling proteins, nucleic acids, ligands, and complexes,
- estimating confidence metrics,
- and, in Boltz2, predicting affinity-related outputs.

At the code level, the main inference pipeline is organized around:
- input parsing,
- featurization,
- a trunk that iteratively updates internal representations,
- a structure module that turns those representations into coordinates,
- and downstream heads for confidence and affinity.

The most important entrypoint is:
- [main.py](/usr/scratch/rmanimaran8/boltz/src/boltz/main.py)

The most important model file is:
- [boltz2.py](/usr/scratch/rmanimaran8/boltz/src/boltz/model/models/boltz2.py)

The most important “internal computation” files for your project are:
- [trunkv2.py](/usr/scratch/rmanimaran8/boltz/src/boltz/model/modules/trunkv2.py)
- [pairformer.py](/usr/scratch/rmanimaran8/boltz/src/boltz/model/layers/pairformer.py)

If you want a compact mental model of Boltz2, think:

```text
Input molecules
  -> feature construction
  -> single representation s
  -> pair representation z
  -> repeated Pairformer updates of s and z
  -> distogram / diffusion structure generation
  -> confidence and affinity heads
```

That is the simplest faithful summary.

---

## 3. The Two Representation Spaces In Boltz

This is the most important architectural idea to understand.

Boltz carries around two different but coupled internal state spaces:

### 3.1 Single representation

The single representation is usually written as `s`.

It is a per-token or per-residue representation.

Shape:

$$
s \in \mathbb{R}^{B \times N \times H}
$$

Where:
- `B` = batch size,
- `N` = number of residues or tokens,
- `H` = single representation width.

In the current setup used in your docs and code:

$$
H = 384
$$

Intuition:
- each residue gets a hidden vector,
- that vector can encode sequence identity, local structural context, evolutionary information, and higher-level learned concepts.

You can think of `s` as “what the model knows about each residue individually.”

### 3.2 Pair representation

The pair representation is usually written as `z` in the Boltz code, and often called `y` in the transcoder math document when used as the target.

Shape:

$$
z \in \mathbb{R}^{B \times N \times N \times P}
$$

or flattened for training:

$$
y \in \mathbb{R}^{B \times N^2 \times P}
$$

Where:
- `P` = pair representation width,
- in your current setup `P = 128`.

Intuition:
- for every residue pair `(i, j)`, the model stores a learned relational vector,
- this can encode distance tendencies, contact information, geometry, chain interaction structure, interface information, and constraints.

You can think of `z` as “what the model knows about relationships between residues.”

### 3.3 Why this matters

A lot of transformer interpretability assumes one main hidden state type:
- token hidden states in a language model,
- patch hidden states in a vision transformer,
- or one vector per position.

Boltz is different because it has:
- token states,
- pair states,
- and the model’s main reasoning depends on both.

So if you want to interpret Boltz, you cannot ignore the pair representation.

That is exactly the problem your transcoder project is solving.

---

## 4. How Boltz2 Builds Its Internal States

Now we move from high-level overview into the real forward path.

The cleanest place to understand this is the `forward` method in [boltz2.py](/usr/scratch/rmanimaran8/boltz/src/boltz/model/models/boltz2.py).

### 4.1 Input embedding

The model first computes:

```python
s_inputs = self.input_embedder(feats)
```

This is the first true token-level representation.

The `InputEmbedder` in [trunkv2.py](/usr/scratch/rmanimaran8/boltz/src/boltz/model/modules/trunkv2.py) combines:
- atom-level information,
- residue identity,
- MSA-derived profile features,
- optional conditioning such as modified flags, cyclic features, or molecule-type features.

So `s_inputs` is not just a sequence embedding.
It is already a biologically informed token representation built from molecular and evolutionary context.

### 4.2 Single initialization

After that:

```python
s_init = self.s_init(s_inputs)
```

This creates the initialized token state used by the trunk.

### 4.3 Pair initialization

The pair state is initialized from the token input by combining token features in both row and column directions:

```python
z_init = (
    self.z_init_1(s_inputs)[:, :, None]
    + self.z_init_2(s_inputs)[:, None, :]
)
```

Then Boltz adds more relational information:
- relative position encoding,
- bond features,
- contact conditioning,
- optionally bond type features.

Conceptually, this means:
- `z_init[i, j]` starts as a learned function of token `i` and token `j`,
- then receives structural priors about position, bond structure, and constraints.

This is important for your project because it shows that pair space is not secondary. It is foundational from the very start.

### 4.4 Recycling

Boltz then iteratively refines `s` and `z` across recycling steps:

```python
s = s_init + self.s_recycle(self.s_norm(s))
z = z_init + self.z_recycle(self.z_norm(z))
```

Recycling means the model does not just compute one pass of hidden states.
It takes the current estimate and feeds it back through the trunk for additional refinement.

Interpretability implication:
- the final hidden states are the result of iterative geometric reasoning,
- not just one shallow pass over the input.

---

## 5. What The MSA Module Does

The MSA module is important because it enriches pair space using evolutionary information.

In [trunkv2.py](/usr/scratch/rmanimaran8/boltz/src/boltz/model/modules/trunkv2.py), the model computes:

```python
z = z + msa_module(z, s_inputs, feats, ...)
```

This means the MSA module outputs an update to the pair representation.

At a conceptual level:
- the model receives multiple aligned homologous sequences,
- projects them into an MSA representation,
- mixes MSA information across rows,
- and uses that information to refine the pair representation.

This is a common pattern in structure models:
- sequence-family information is often most useful when translated into residue-residue relation signals.

Inside `MSALayer`, the main steps are:
- pair-weighted averaging from pair space into MSA space,
- MSA transition MLP,
- outer product mean from MSA space back into pair space,
- pairwise refinement through a pair-only Pairformer variant.

The key idea is:

**MSA information is injected into Boltz primarily by shaping the relational geometry encoded in pair space.**

That is part of why pair space is so important to interpret.

---

## 6. What The Pairformer Does

The Pairformer trunk is the heart of the model.

The key file is:
- [pairformer.py](/usr/scratch/rmanimaran8/boltz/src/boltz/model/layers/pairformer.py)

Inside each `PairformerLayer`, the forward pass looks approximately like this:

### 6.1 Pair updates first

The pair representation `z` is updated through:
- triangle multiplication outgoing,
- triangle multiplication incoming,
- triangle attention starting node,
- triangle attention ending node,
- then an MLP-style transition block `transition_z`.

Then:

$$
z \leftarrow z + \text{pair updates} + \text{transition}_z(z)
$$

This part is where residue-residue relational structure is refined.

### 6.2 Single update next

The single representation `s` is updated using attention that is biased by pair information:

```python
s = s + self.attention(s=s_normed, z=z, ...)
s = s + self.transition_s(s)
```

So the model uses the refined pair structure to update token-level state, then applies an MLP transition on token state.

This alternating structure is central:
- pair state helps update token state,
- token state originally helped seed pair state,
- and both are refined layer after layer.

### 6.3 Why this matters for mechanistic analysis

The Pairformer transitions are good interpretability targets because:
- they are modular,
- they are repeated across layers,
- and they sit at a location where rich concepts are likely concentrated.

The transition blocks are also easy to hook:
- `transition_s` gives you token-level MLP input/output,
- `transition_z` gives you pair-level MLP input/output.

This is exactly what your collector scripts exploit.

---

## 7. Why The Transition Modules Are The Right Hook Point

In mechanistic interpretability, you generally want a place in the network that is:
- semantically rich,
- structurally clean,
- and stable enough to study across many inputs.

The transition modules in Pairformer satisfy those requirements.

### 7.1 Richness

By the time a Pairformer layer is reached, the model has already integrated:
- atom-level features,
- sequence identity,
- MSA context,
- relative positional information,
- and earlier trunk computations.

So the transition modules are not operating on raw inputs.
They are operating on already meaningful hidden states.

### 7.2 Clean interface

A transition block has a clear input and output.

That gives you:
- `input_s`, `output_s`,
- `input_z`, `output_z`.

Those are exactly the tensors you need for a controlled interpretability objective.

### 7.3 Modular and repeated

Every Pairformer layer has transition modules, so you can:
- analyze one layer,
- compare multiple layers,
- or scale to an online multi-layer analysis workflow.

That is why the transcoder project generalizes beyond a single pilot layer.

---

## 8. What A Transcoder Is, Conceptually

Before focusing on Boltz, it helps to understand the basic transcoder concept.

### 8.1 Autoencoder vs transcoder

A sparse autoencoder tries to reconstruct the same space:

$$
x \rightarrow z \rightarrow \hat{x}
$$

A transcoder is more general:

$$
x \rightarrow z \rightarrow \hat{y}
$$

You still learn a sparse latent basis `z`, but now the model does not need to reconstruct the exact input space. It can map from one internal representation to another.

### 8.2 Why sparsity matters

Sparse latent features are valuable because:
- only a few features are active at once,
- features are forced to specialize,
- attribution becomes easier,
- and the latent dictionary becomes more interpretable.

TopK sparsity is a particularly strong form of control:
- for each token, only the top `k` activations survive,
- all others are zeroed out.

So if `D = 2048` and `k = 16`, each token activates only 16 out of 2048 latent features.

That is a clean interpretability regime.

### 8.3 In standard settings

A normal sparse transcoder often looks like:

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

and a corresponding loss such as:

$$
\mathcal{L}_{std} = \operatorname{MSE}(\hat{y}, y) + \lambda \Omega(z)
$$

This is the baseline mental model you should keep in mind.

Your method is an extension of this baseline.

---

## 9. Why A Normal Transcoder Does Not Fit Boltz

This is the core intellectual justification for your project.

It is worth understanding deeply, because this is the main thing that turns your work from “apply a known method” into “adapt a method to a nontrivial architecture.”

There are several reasons.

### 9.1 Geometry mismatch: token space vs pair space

A normal transcoder usually assumes the target lives on the same kind of axis as the input.

For example:
- token hidden states to token hidden states,
- or token hidden states to logits indexed by the same positions.

But in Boltz:

Single stream:

$$
s \in \mathbb{R}^{B \times N \times H}
$$

Pair stream:

$$
y \in \mathbb{R}^{B \times N^2 \times P}
$$

This means the input and target do not just have different feature width.
They have different combinatorial geometry.

The input is indexed by residues.
The target is indexed by residue pairs.

That means we are not just doing:

$$
N \rightarrow N
$$

We are doing:

$$
N \rightarrow N^2
$$

That is a major mismatch.

### 9.2 One single-state snapshot is not enough

In each Pairformer transition, there are meaningful pre- and post-transition states:
- `s_1`: input to `transition_s`
- `s_2`: output from `transition_s`

Similarly on the pair side:
- `y_1`: input to `transition_z`
- `y_2`: output from `transition_z`

If you only use one single-state view, you lose information about the transformation itself.

Your project wants a latent basis that is meaningful across the transition, not only at one endpoint.

### 9.3 Token-only reconstruction is too weak

Suppose you trained a standard sparse autoencoder only on `s`.

You might reconstruct `s` well but still fail to explain:
- interface structure,
- residue-residue geometry,
- pairwise constraints,
- and the relational semantics Boltz stores in `z`.

That is because the biologically important structural reasoning is often expressed most directly in pair space, not only in token space.

### 9.4 Dead features are especially harmful here

TopK sparse models can leave some latent units permanently inactive.

That problem already exists in standard SAE training, but here it is more damaging because:
- the target geometry is more complex,
- the supervision is harder,
- and a reduced effective dictionary hurts interpretability quality.

This is why dead-neuron recovery is built into the design.

### 9.5 Boltz data collection is operationally expensive

Collecting Boltz activations is not trivial because:
- there is a full biomolecular preprocessing pipeline,
- MSA handling,
- structured tensor construction,
- and heavy model execution.

That pushes the project toward streamed or online training workflows rather than assuming extremely large, simple offline activation dumps.

---

## 10. The Core Idea Of The Boltz Extension

Now we arrive at the actual method.

The extension used in this repository can be summarized in one sentence:

**Learn a shared sparse token-latent basis that predicts both pre- and post-transition pair representations from both pre- and post-transition single representations.**

That sounds dense, so unpack it carefully.

### 10.1 Inputs to the transcoder training step

For each training example, you collect:

- `s_1`: input single activation
- `s_2`: output single activation
- `y_1`: input pair activation
- `y_2`: output pair activation

These come from hooks on `transition_s` and `transition_z`.

### 10.2 Shared encoder

Both `s_1` and `s_2` go through the same sparse encoder:

$$
z_1 = f_{latent}(s_1), \qquad z_2 = f_{latent}(s_2)
$$

This is important:
- the same latent basis must explain both pathways,
- so the dictionary is pushed toward transition-stable concepts.

### 10.3 Dual decoders

From each latent vector, the model predicts:
- pair-input side target,
- pair-output side target.

So there are two decoder heads:

$$
g_1(z) \rightarrow \hat{y}_1
$$

$$
g_2(z) \rightarrow \hat{y}_2
$$

This means a single sparse latent basis is being used to explain both sides of the pair transition.

### 10.4 Pathway consistency

Because both `s_1` and `s_2` are encoded through the same sparse basis, the model is encouraged to decode similar pair predictions from both pathways.

This is where the consistency loss comes in.

### 10.5 Token-to-pair bridge

The encoder acts on token-level inputs, but the supervision is pair-level.

So the implementation uses an expansion operator:

$$
\mathcal{E}: \mathbb{R}^{B \cdot N \times P} \rightarrow \mathbb{R}^{B \cdot N^2 \times P}
$$

This repeats token predictions across pair rows so they can be compared to pair-space targets.

This is not the final theoretically ideal geometry, but it is the practical bridge that makes training possible right now.

### 10.6 Dead-neuron recovery

The model tracks neurons that remain inactive for too long and uses an auxiliary path to fit residuals using dead-neuron subsets.

That keeps the effective dictionary alive and improves feature coverage.

---

## 11. The Concrete Universal Transcoder In This Repo

The main implementation is:
- [universal_model.py](/usr/scratch/rmanimaran8/boltz/transcoder/universal_transcoder/universal_model.py)

The training logic is in:
- [train_online_multi_layer.py](/usr/scratch/rmanimaran8/boltz/transcoder/universal_transcoder/train_online_multi_layer.py)
- [train_universal.py](/usr/scratch/rmanimaran8/boltz/transcoder/universal_transcoder/train_universal.py)

### 11.1 The model architecture

The universal transcoder uses:
- input dimension `d_model = 384`,
- latent dimension `d_hidden = 2048`,
- pair target width `d_pair = 128`,
- TopK sparsity with `k = 16`,
- auxiliary dead-neuron budget `auxk = 32`.

At a high level:

```text
single token vector s
  -> layer norm and centering
  -> linear encoder
  -> ReLU + TopK sparsity
  -> sparse latent z
  -> decoder_y1
  -> decoder_y2
```

The decoders are stored as parameters and constrained to unit norm by:
- explicit weight normalization,
- and projected gradients.

This mirrors PLT-style stability mechanics.

### 11.2 Why the latent space is shared

The same encoder is used for:
- `s_1`,
- `s_2`.

That means the same latent dimensions must explain:
- single-state information before the transition,
- and single-state information after the transition.

This is useful because it pressures the latent dictionary to represent persistent concepts rather than idiosyncratic activations of only one pathway.

### 11.3 Why there are two decoder heads

There are two pair targets:
- `y_1` = pair activation before `transition_z`
- `y_2` = pair activation after `transition_z`

If you only decoded one of them, you would lose information about how single-space concepts relate to both sides of the pair transition.

Using two decoder heads turns the model into a more faithful probe of the pair dynamics.

---

## 12. The Math: Encoder, Latent Function, Decoder

This section is the core math you should know well.

The notation from the architecture math document is:

- `B`: batch size
- `N`: number of tokens
- `H`: single representation size, usually 384
- `P`: pair representation size, usually 128
- `D`: latent dimension, usually 2048
- `k`: TopK active features

### 12.1 Flattening

For training, token-level single activations are flattened:

$$
s_1^{flat}, s_2^{flat} \in \mathbb{R}^{(B \cdot N) \times H}
$$

Pair activations are flattened over pair rows:

$$
y_1^{flat}, y_2^{flat} \in \mathbb{R}^{(B \cdot N^2) \times P}
$$

### 12.2 Layer normalization and centering

For a single flattened token vector `s`:

$$
\mu(s) = \frac{1}{H}\sum_{j=1}^{H} s_j
$$

$$
\sigma(s) = \sqrt{\frac{1}{H}\sum_{j=1}^{H}(s_j-\mu(s))^2}
$$

$$
\operatorname{LN}(s)=\frac{s-\mu(s)}{\sigma(s)+\epsilon}
$$

Then center:

$$
\tilde{s}=\operatorname{LN}(s)-b_{pre}
$$

This makes the encoder more stable and more comparable across examples.

### 12.3 Encoder and latent map

The pre-activations are:

$$
h = W_{enc}\tilde{s}+b_{enc}
$$

Then sparse TopK activation:

$$
z = \operatorname{TopK}(\operatorname{ReLU}(h), k)
$$

This defines the latent function:

$$
f_{latent}(s) = \operatorname{TopK}(\operatorname{ReLU}(W_{enc}(\operatorname{LN}(s)-b_{pre})+b_{enc}), k)
$$

This is one of the most important equations in the project.

Interpretation:
- normalize the token state,
- linearly project it into a big feature dictionary,
- apply nonlinearity,
- keep only the top `k` features.

So the latent vector is sparse and hopefully interpretable.

### 12.4 Decoder equations

For pathway `i \in \{1,2\}`:

$$
z_i = f_{latent}(s_i)
$$

Then the two decoder heads are:

$$
\hat{y}^{token}_{1,i}=z_iW_{dec,1}+b_{pre,1}
$$

$$
\hat{y}^{token}_{2,i}=z_iW_{dec,2}+b_{pre,2}
$$

In the implementation, these token-level outputs are also rescaled by the input normalization statistics.

Interpretation:
- the sparse latent basis is decoded into two pair-oriented views,
- one for pair-input side,
- one for pair-output side.

### 12.5 Token-to-pair expansion

Because the decoder operates on tokens but the target is pair-indexed, the model uses an expansion operator:

$$
\hat{y}_{m,i}=\mathcal{E}(\hat{y}^{token}_{m,i}), \quad m\in\{1,2\}
$$

Operationally, this repeats or broadcasts token predictions into pair rows.

The important thing to say clearly is:

**This is the compatibility layer that allows token-latent features to be supervised against pair-space targets.**

It is a pragmatic approximation, not the final ideal pair-aware decoder.

---

## 13. The Math: Loss Function

The loss has three main parts:
- reconstruction,
- consistency,
- auxiliary residual fitting.

### 13.1 Normalized MSE

The training code and math document use normalized MSE:

$$
\operatorname{NMSE}(a,b)=\frac{\operatorname{MSE}(a,b)}{\operatorname{Var}(b)+\epsilon}
$$

Why use NMSE instead of plain MSE?

Because in Boltz:
- different targets can have different scales,
- different proteins and layers can produce different target variances,
- and raw MSE can become misleading.

NMSE makes losses more comparable and better behaved.

### 13.2 Reconstruction loss

Each pathway predicts both pair targets.

That creates four reconstruction terms:

$$
\mathcal{L}_{recon}=
\operatorname{NMSE}(\hat{y}_{1,1},y_1)+
\operatorname{NMSE}(\hat{y}_{2,1},y_2)+
\operatorname{NMSE}(\hat{y}_{1,2},y_1)+
\operatorname{NMSE}(\hat{y}_{2,2},y_2)
$$

Interpretation:
- `s_1` should explain `y_1`,
- `s_1` should also explain `y_2`,
- `s_2` should explain `y_1`,
- `s_2` should also explain `y_2`.

That is much richer than a standard single reconstruction objective.

### 13.3 Consistency loss

The consistency loss compares the token-level decoder outputs from the two pathways:

$$
\mathcal{L}_{cons}=
\frac{\operatorname{MSE}(\hat{y}^{token}_{1,1},\hat{y}^{token}_{1,2})}{\operatorname{Var}(y_1)+\epsilon}
+
\frac{\operatorname{MSE}(\hat{y}^{token}_{2,1},\hat{y}^{token}_{2,2})}{\operatorname{Var}(y_2)+\epsilon}
$$

Interpretation:
- if a concept is robust, decoding from `s_1` and `s_2` should agree,
- so the latent basis is encouraged to represent stable structure rather than one-off local fluctuations.

This is one of the most important conceptual innovations in the method.

### 13.4 Auxiliary dead-neuron residual loss

Residuals are defined as:

$$
r_{1,i}=y_1-\hat{y}_{1,i}, \quad r_{2,i}=y_2-\hat{y}_{2,i}
$$

Dead-neuron subsets are used to predict these residuals:

$$
\hat{r}_{1,i},\hat{r}_{2,i}
$$

Then:

$$
\mathcal{L}_{aux}=\alpha\sum_{i\in\{1,2\}}\left[\operatorname{NMSE}(\hat{r}_{1,i},r_{1,i})+\operatorname{NMSE}(\hat{r}_{2,i},r_{2,i})\right]
$$

Interpretation:
- if some neurons are dead or underused,
- we give them a chance to model what the current active-path reconstruction missed.

This improves dictionary utilization.

### 13.5 Total loss

The full objective is:

$$
\mathcal{L}_{total}=\mathcal{L}_{recon}+\mathcal{L}_{cons}+\mathcal{L}_{aux}
$$

This equation is the shortest faithful summary of your training objective.

---

## 14. Direct Comparison: Standard Sparse Transcoder vs Boltz Extension

This comparison is worth internalizing because it clarifies the novelty very cleanly.

### 14.1 Standard sparse transcoder

Typical setup:

$$
z=\phi(x), \qquad \hat{x}=g(z)
$$

Typical loss:

$$
\mathcal{L}_{std}=\operatorname{MSE}(\hat{x},x)+\lambda\Omega(z)
$$

Key assumptions:
- one input stream,
- one output stream,
- roughly matching axis geometry,
- reconstruction-centric objective.

### 14.2 Boltz extension

Your setup:

$$
z_1=\phi(s_1), \qquad z_2=\phi(s_2)
$$

$$
\hat{y}_{1,1}=\mathcal{E}(g_1(z_1)), \qquad
\hat{y}_{2,1}=\mathcal{E}(g_2(z_1))
$$

$$
\hat{y}_{1,2}=\mathcal{E}(g_1(z_2)), \qquad
\hat{y}_{2,2}=\mathcal{E}(g_2(z_2))
$$

Loss:

$$
\mathcal{L}_{boltz}
=
\sum \text{reconstruction terms}
+
\sum \text{consistency terms}
+
\mathcal{L}_{aux}
$$

### 14.3 What actually changed

Compared to a standard transcoder, you introduced:

1. two input pathways instead of one,
2. two decoder heads instead of one,
3. pair-space supervision instead of same-axis token supervision,
4. an expansion bridge to handle `N -> N^2`,
5. a consistency objective across pathways,
6. explicit dead-neuron recovery.

That is the right way to explain the extension to an audience.

---

## 15. How Activation Collection Works In This Repo

The collection logic is in:
- [collect_multi_layer.py](/usr/scratch/rmanimaran8/boltz/transcoder/collection_scripts/collect_multi_layer.py)

The key idea is simple:
- register forward hooks on `transition_s` and `transition_z`,
- run the Boltz model,
- save the inputs and outputs of those submodules.

For each chosen layer, the collector stores:
- `input_s`
- `output_s`
- `input_z`
- `output_z`

For pair activations:
- the raw shape is `[B, N, N, P]`,
- and it is flattened to `[B, N^2, P]`.

This gives you directly the four tensors needed by the transcoder objective.

This part is conceptually important because it ties the theory to the actual project:

you are not inventing abstract symbols after the fact.

You are literally training on tensors collected from real Boltz Pairformer transition modules.

---

## 16. How Online Multi-Layer Training Works

The multi-layer trainer is in:
- [train_online_multi_layer.py](/usr/scratch/rmanimaran8/boltz/transcoder/universal_transcoder/train_online_multi_layer.py)

The purpose of this script is to:
- collect Boltz activations,
- immediately train layer-specific transcoders,
- and do so in a deterministic, reproducible way.

Important details:
- all RNG seeds are set,
- CUDA deterministic options are enabled,
- MSA subsampling can be disabled for reproducibility,
- separate layer trainers can be maintained for multiple layer indices.

Conceptually, this gives you a scalable workflow:
- pick multiple Pairformer layers,
- collect activations from each,
- train a separate transcoder per layer,
- compare the resulting dictionaries.

This is valuable because representation meaning may change with depth.

For example:
- early layers may encode more local or sequence-derived information,
- deeper layers may encode more refined global structure or interface semantics.

---

## 17. How To Think About What The Learned Features Mean

After training, the sparse latents are the objects you care about.

Let:

$$
Z \in \mathbb{R}^{(B \cdot N) \times D}
$$

Then each latent dimension `j` is a candidate interpretable feature.

### 17.1 Activation frequency

One basic metric is:

$$
f_j = \frac{1}{B\cdot N}\sum_{t=1}^{B\cdot N}\mathbf{1}[Z_{t,j} > 0]
$$

This tells you how often a feature turns on.

Interpretation:
- very frequent features may represent universal structural primitives,
- rarer features may represent selective motifs or interfaces.

### 17.2 Decoder vectors

Each feature has decoder contributions:

$$
v_{1,j}=W_{dec,1}[j,:], \qquad v_{2,j}=W_{dec,2}[j,:]
$$

These vectors tell you how that feature contributes to the predicted pair targets.

This means interpretability is not only about “when is a feature active?”

It is also about:
- what pair-space effect does that feature induce?

### 17.3 Practical probing questions

Once you have trained features, useful downstream probes include:
- correlation with amino acid class,
- correlation with secondary structure,
- correlation with solvent exposure,
- correlation with interfaces,
- comparison across layer depth,
- comparison of decoder head 1 vs decoder head 2.

This is where the project can become biologically interesting rather than only mathematically interesting.

---

## 18. Current Limitations

A strong presentation benefits from being honest about limitations.

### 18.1 Expansion operator is crude

The token-to-pair expansion bridge is the biggest current approximation.

It lets training work, but it does not explicitly model pairwise interaction as a function of both residues `i` and `j`.

In other words, it is useful, but not fully pair-aware.

### 18.2 Low loss is not the same as interpretability

A feature dictionary can reconstruct targets well without being semantically clean.

You still need downstream probing and validation.

### 18.3 Some repo docs are out of sync

Some markdown files describe older stages of the project, older layer counts, or earlier pilot variants.

The code should be treated as the source of truth.

### 18.4 Cross-layer interpretability is still a next step

Per-layer training is supported, but fully understanding how dictionaries relate across layers remains future work.

---

## 19. Natural Next Extensions

These are good future-work ideas and also useful for showing that you understand the limits of the current method.

### 19.1 Pair-aware decoder

A more expressive decoder would explicitly depend on both token features:

$$
\hat{y}_{ij}=g(z_i, z_j, e_{ij})
$$

This would better reflect actual pair geometry rather than relying on a broadcast expansion.

### 19.2 Symmetry-aware objectives

When appropriate, enforce:

$$
\mathcal{L}_{sym}=\operatorname{MSE}(\hat{y}_{ij}, \hat{y}_{ji})
$$

This could better match the symmetry structure of some pair targets.

### 19.3 Feature decorrelation

Encourage a more orthogonal or disentangled dictionary:

$$
\mathcal{L}_{decor}=\left\|\frac{1}{T}Z^\top Z-I\right\|_F^2
$$

### 19.4 Cross-layer alignment

Train or post-process dictionaries so that similar concepts can be compared across layers.

This would be especially useful if you want to tell a story about representation evolution through the trunk.

---

## 20. The Cleanest High-Level Story You Can Tell

If you want one concise but rigorous story to keep in your head, it is this:

Boltz2 reasons using both token-level and pair-level hidden states.
That makes standard token-only interpretability methods inadequate.
So we collect transition activations from the Pairformer trunk and train a PLT-style sparse transcoder that uses token-space inputs to explain pair-space targets.
The model uses:
- a shared sparse encoder,
- dual decoder heads,
- pathway consistency,
- normalized reconstruction losses,
- and dead-neuron recovery.

The result is a sparse latent basis that is better aligned with how Boltz actually represents structure.

That is the core project.

---

## 21. How To Study This Project Efficiently

If you want to move from “I have read the docs” to “I can confidently explain the project,” use this reading order:

1. [boltz2.py](/usr/scratch/rmanimaran8/boltz/src/boltz/model/models/boltz2.py)
Read the `forward` method first. Understand how `s_inputs`, `s_init`, `z_init`, recycling, MSA, Pairformer, diffusion, confidence, and affinity connect.

2. [pairformer.py](/usr/scratch/rmanimaran8/boltz/src/boltz/model/layers/pairformer.py)
Understand exactly how a Pairformer layer updates `z` and then `s`, and where `transition_s` / `transition_z` sit.

3. [trunkv2.py](/usr/scratch/rmanimaran8/boltz/src/boltz/model/modules/trunkv2.py)
Understand input embedding, template logic, MSA logic, and distogram output.

4. [collect_multi_layer.py](/usr/scratch/rmanimaran8/boltz/transcoder/collection_scripts/collect_multi_layer.py)
Understand how the activations are actually captured.

5. [universal_model.py](/usr/scratch/rmanimaran8/boltz/transcoder/universal_transcoder/universal_model.py)
Understand the sparse model itself.

6. [train_online_multi_layer.py](/usr/scratch/rmanimaran8/boltz/transcoder/universal_transcoder/train_online_multi_layer.py)
Understand how the loss is implemented in practice.

7. [BOLTZ_TRANSCODER_ARCHITECTURE_MATH.md](/usr/scratch/rmanimaran8/boltz/transcoder/documentation/BOLTZ_TRANSCODER_ARCHITECTURE_MATH.md)
Use this to formalize what you already understood from the code.

This order is better than reading only the docs, because it grounds the ideas in real execution flow.

---

## 22. Final Takeaway

The most important conceptual takeaway is:

**The project is not just “train a sparse autoencoder on Boltz.”**

It is:

**adapt sparse transcoder / PLT-style interpretability to a model whose internal reasoning is fundamentally split across token and pair geometries.**

That is why the method needs:
- shared encoder across multiple single-state pathways,
- dual decoder heads,
- pair-space supervision,
- geometry bridge from token outputs to pair targets,
- and consistency constraints.

That is the real intellectual center of the work.

