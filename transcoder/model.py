"""Joint Transcoder for Pairformer Layer 48 MLP activations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointTranscoder(nn.Module):
    """
    Joint transcoder that processes both single (s) and pair (z) MLP activations.
    
    Architecture:
    - Two separate encoders: one for s (384->2048), one for z (128->2048)
    - Shared sparse latent space (2048 dimensions)
    - Two separate decoders: one for s (2048->384), one for z (2048->128)
    """
    
    def __init__(
        self,
        dim_s: int = 384,
        dim_z: int = 128,
        latent_dim: int = 2048,
        l1_coeff: float = 1e-4,
    ):
        super().__init__()
        
        self.dim_s = dim_s
        self.dim_z = dim_z
        self.latent_dim = latent_dim
        self.l1_coeff = l1_coeff
        
        # Encoder for single representation (s)
        self.encoder_s = nn.Linear(dim_s, latent_dim, bias=True)
        
        # Encoder for pair representation (z)
        self.encoder_z = nn.Linear(dim_z, latent_dim, bias=True)
        
        # Decoder for single representation (s)
        self.decoder_s = nn.Linear(latent_dim, dim_s, bias=True)
        
        # Decoder for pair representation (z)
        self.decoder_z = nn.Linear(latent_dim, dim_z, bias=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following standard practice for autoencoders."""
        # Initialize encoders with small weights
        nn.init.kaiming_uniform_(self.encoder_s.weight, a=0.01)
        nn.init.kaiming_uniform_(self.encoder_z.weight, a=0.01)
        nn.init.zeros_(self.encoder_s.bias)
        nn.init.zeros_(self.encoder_z.bias)
        
        # Initialize decoders (transpose of encoders is a good starting point)
        nn.init.kaiming_uniform_(self.decoder_s.weight, a=0.01)
        nn.init.kaiming_uniform_(self.decoder_z.weight, a=0.01)
        nn.init.zeros_(self.decoder_s.bias)
        nn.init.zeros_(self.decoder_z.bias)
    
    def encode(self, x_s, x_z):
        """
        Encode both s and z inputs into shared latent space.
        
        Args:
            x_s: Single representation activations [..., 384]
            x_z: Pair representation activations [..., 128]
        
        Returns:
            latent_s: Latent representation from s [..., 2048]
            latent_z: Latent representation from z [..., 2048]
        """
        latent_s = F.relu(self.encoder_s(x_s))
        latent_z = F.relu(self.encoder_z(x_z))
        return latent_s, latent_z
    
    def decode(self, latent_s, latent_z):
        """
        Decode latent representations back to original spaces.
        
        Args:
            latent_s: Latent representation from s [..., 2048]
            latent_z: Latent representation from z [..., 2048]
        
        Returns:
            recon_s: Reconstructed s [..., 384]
            recon_z: Reconstructed z [..., 128]
        """
        recon_s = self.decoder_s(latent_s)
        recon_z = self.decoder_z(latent_z)
        return recon_s, recon_z
    
    def forward(self, x_s, x_z):
        """
        Full forward pass through the transcoder.
        
        Args:
            x_s: Single representation activations [..., 384]
            x_z: Pair representation activations [..., 128]
        
        Returns:
            recon_s: Reconstructed s [..., 384]
            recon_z: Reconstructed z [..., 128]
            latent_s: Latent representation from s [..., 2048]
            latent_z: Latent representation from z [..., 2048]
        """
        latent_s, latent_z = self.encode(x_s, x_z)
        recon_s, recon_z = self.decode(latent_s, latent_z)
        return recon_s, recon_z, latent_s, latent_z
    
    def compute_loss(self, x_s, x_z, recon_s, recon_z, latent_s, latent_z):
        """
        Compute reconstruction loss with L1 sparsity penalty.
        
        Args:
            x_s: Original s activations
            x_z: Original z activations
            recon_s: Reconstructed s
            recon_z: Reconstructed z
            latent_s: Latent representation from s
            latent_z: Latent representation from z
        
        Returns:
            total_loss: Combined loss
            metrics: Dict of individual loss components
        """
        # Reconstruction losses (MSE)
        recon_loss_s = F.mse_loss(recon_s, x_s)
        recon_loss_z = F.mse_loss(recon_z, x_z)
        
        # Combined reconstruction loss (equal weighting)
        recon_loss = recon_loss_s + recon_loss_z
        
        # L1 sparsity penalty on latent activations
        l1_loss_s = torch.mean(torch.abs(latent_s))
        l1_loss_z = torch.mean(torch.abs(latent_z))
        l1_loss = l1_loss_s + l1_loss_z
        
        # Total loss
        total_loss = recon_loss + self.l1_coeff * l1_loss
        
        # L0 sparsity (fraction of non-zero activations)
        with torch.no_grad():
            l0_s = (latent_s > 1e-6).float().mean()
            l0_z = (latent_z > 1e-6).float().mean()
        
        metrics = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'recon_loss_s': recon_loss_s.item(),
            'recon_loss_z': recon_loss_z.item(),
            'l1_loss': l1_loss.item(),
            'l0_sparsity_s': l0_s.item(),
            'l0_sparsity_z': l0_z.item(),
        }
        
        return total_loss, metrics


if __name__ == "__main__":
    # Test the model
    print("Testing JointTranscoder...")
    
    model = JointTranscoder()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data
    batch_size = 16
    seq_len = 64
    x_s = torch.randn(batch_size, seq_len, 384)
    x_z = torch.randn(batch_size, seq_len * seq_len, 128)  # Flattened pairs
    
    # Forward pass
    recon_s, recon_z, latent_s, latent_z = model(x_s, x_z)
    
    print(f"Input shapes: s={x_s.shape}, z={x_z.shape}")
    print(f"Latent shapes: s={latent_s.shape}, z={latent_z.shape}")
    print(f"Recon shapes: s={recon_s.shape}, z={recon_z.shape}")
    
    # Compute loss
    loss, metrics = model.compute_loss(x_s, x_z, recon_s, recon_z, latent_s, latent_z)
    print(f"\nLoss metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n✓ Model test passed!")
