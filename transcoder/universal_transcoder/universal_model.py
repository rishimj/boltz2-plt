"""
Universal Transcoder Model - PLT-Compatible Architecture

Maps single representations (s) to two pair representations (y1, y2).
Based on PerLayerTranscoder (PLT) architecture for easy conversion to full PLT later.
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size 
# N: Number of tokens (residues)
# D: Latent Dim (d_hidden)
# H: Input dimension (d_model = 384 for single representation)
# P: Pair representation dimension (128)
# ──────────────────────────────────────────────────────────────────────────────

class UniversalTranscoder(nn.Module):
    def __init__(
        self,
        d_model: int = 384,          # Input dimension (single representation)
        d_hidden: int = 2048,        # Latent dimension
        d_pair: int = 128,           # Pair representation dimension
        k: int = 16,                 # Top-K activation
        auxk: int = 32,              # Auxiliary K for dead neurons
        batch_size: int = 10,        # Batch size for dead neuron tracking
        dead_steps_threshold: int = 10000,
    ):
        """
        Initializes the Universal Transcoder.
        
        Architecture:
        - Single encoder: Linear(d_model → d_hidden) with TopK activation
        - Two decoders: decoder_y1 and decoder_y2 (both d_hidden → d_pair)
        - Dead neuron tracking and AuxK resurrection mechanism
        
        Args:
            d_model: Input dimension (384 for single representation)
            d_hidden: Latent dimension (2048 default)
            d_pair: Pair representation dimension (128)
            k: Top-K activation sparsity
            auxk: Auxiliary K for dead neuron resurrection
            batch_size: Batch size for normalizing dead steps threshold
            dead_steps_threshold: Steps before neuron considered dead
        """
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.d_pair = d_pair
        self.k = k
        self.auxk = auxk
        self.batch_size = batch_size
        self.dead_steps_threshold = dead_steps_threshold / batch_size

        # --- Encoder ---
        self.encoder = nn.Linear(d_model, d_hidden)
        
        # --- Decoders (as Parameters for unit norm constraint) ---
        self.decoder_y1 = nn.Parameter(torch.empty(d_hidden, d_pair))
        self.decoder_y2 = nn.Parameter(torch.empty(d_hidden, d_pair))
        
        # --- Biases ---
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.b_pre_y1 = nn.Parameter(torch.zeros(d_pair))
        self.b_pre_y2 = nn.Parameter(torch.zeros(d_pair))
        
        # --- Initialize (matching PLT) ---
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        
        # Initialize decoders with Kaiming initialization
        # Decoders are (d_hidden, d_pair) = (2048, 128)
        nn.init.kaiming_uniform_(self.decoder_y1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.decoder_y2, a=math.sqrt(5))
        
        # Normalize to unit norm (matching PLT: dim=0 is hidden dimension)
        self.decoder_y1.data /= self.decoder_y1.data.norm(dim=0, keepdim=True)
        self.decoder_y2.data /= self.decoder_y2.data.norm(dim=0, keepdim=True)

        # Buffer for tracking dead neurons
        self.register_buffer("stats_last_nonzero", torch.zeros(d_hidden, dtype=torch.long))

    def topK_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Top-K activation function.
        
        Args:
            x: Input tensor [..., D]
            k: Number of top activations to keep
            
        Returns:
            Sparse tensor with only top-k values (ReLU applied)
        """
        topk = torch.topk(x, k=k, dim=-1, sorted=False)
        values = F.relu(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def LN(self, x: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Layer normalization.
        
        Args:
            x: Input tensor [..., H]
            eps: Epsilon for numerical stability
            
        Returns:
            Tuple of (normalized_x, mean, std)
        """
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass through Universal Transcoder.
        
        Args:
            x: Input single representation [B, N, H] or [B*N, H]
            
        Returns:
            Tuple of:
            - y1_recon: Reconstructed y1 [B*N, P]
            - y2_recon: Reconstructed y2 [B*N, P]
            - auxk_y1: AuxK reconstruction for y1 (or None)
            - auxk_y2: AuxK reconstruction for y2 (or None)
            - dead_mask: Boolean mask of dead neurons [D]
        """
        # Handle both [B, N, H] and [B*N, H] inputs
        if x.dim() == 3:
            B, N, H = x.shape
            x = x.reshape(B * N, H)
        else:
            assert x.dim() == 2
        
        # 1. Normalize & Center (matching PLT)
        x_norm, mu, std = self.LN(x)
        x_centered = x_norm - self.b_pre
        
        # 2. Encode
        pre_acts = self.encoder(x_centered) + self.b_enc  # [B*N, D]
        
        # 3. Activate (TopK sparsity)
        latents = self.topK_activation(pre_acts, k=self.k)  # [B*N, D]

        # 4. Decode to both y1 and y2
        y1_recon = (latents @ self.decoder_y1) + self.b_pre_y1  # [B*N, P]
        y2_recon = (latents @ self.decoder_y2) + self.b_pre_y2  # [B*N, P]
        
        # 5. Denormalize
        y1_recon = y1_recon * std + mu
        y2_recon = y2_recon * std + mu

        # --- Stats & AuxK (matching PLT) ---
        is_dead = (latents == 0).all(dim=0).long()  # [D]
        
        auxk_y1 = None
        auxk_y2 = None
        
        if self.stats_last_nonzero.sum() > self.dead_steps_threshold:
            dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
            num_dead = dead_mask.sum().item()
            
            if num_dead > 0:
                k_aux = min(self.d_model // 2, num_dead)
                
                # Compute auxiliary activations for dead neurons only
                aux_latents = torch.where(dead_mask[None, :], pre_acts, -torch.inf)
                aux_acts = self.topK_activation(aux_latents, k=k_aux)
                
                # Decode auxiliary activations
                auxk_y1 = (aux_acts @ self.decoder_y1) + self.b_pre_y1
                auxk_y2 = (aux_acts @ self.decoder_y2) + self.b_pre_y2
                
                # Denormalize
                auxk_y1 = auxk_y1 * std + mu
                auxk_y2 = auxk_y2 * std + mu
        
        if auxk_y1 is None:
            auxk_y1 = torch.zeros_like(y1_recon)
            auxk_y2 = torch.zeros_like(y2_recon)

        # Update dead neuron statistics
        with torch.no_grad():
            self.stats_last_nonzero *= is_dead
            self.stats_last_nonzero += 1
        
        dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
        
        return y1_recon, y2_recon, auxk_y1, auxk_y2, dead_mask

    @torch.no_grad()
    def norm_weights(self):
        """Normalizes decoder weights to unit norm (matching PLT)."""
        # MATCH PLT: dim=0 (Hidden dimension)
        self.decoder_y1.data /= self.decoder_y1.data.norm(dim=0, keepdim=True)
        self.decoder_y2.data /= self.decoder_y2.data.norm(dim=0, keepdim=True)

    @torch.no_grad()
    def norm_grad(self):
        """Projects gradients to maintain unit norm constraint (matching PLT)."""
        for param_name, param in [('decoder_y1', self.decoder_y1), ('decoder_y2', self.decoder_y2)]:
            if param.grad is not None:
                # Project gradient to be orthogonal to current weights
                dot_products = torch.sum(param.data * param.grad, dim=0, keepdim=True)
                param.grad.sub_(param.data * dot_products)
