"""Train the joint transcoder on collected activations."""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from datetime import datetime

from model import JointTranscoder


class ActivationDataset(Dataset):
    """Dataset for loading saved activations."""
    
    def __init__(self, activation_dir, max_samples_per_file=None):
        """
        Initialize dataset from saved activation files.
        
        Args:
            activation_dir: Directory containing batch_*.npz files
            max_samples_per_file: Limit samples per file (for debugging)
        """
        self.activation_dir = Path(activation_dir)
        self.max_samples_per_file = max_samples_per_file
        
        # Find all batch files
        self.batch_files = sorted(self.activation_dir.glob("batch_*.npz"))
        print(f"Found {len(self.batch_files)} batch files")
        
        # Load and count samples
        self.samples = []
        total_s = 0
        total_z = 0
        
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            
            # Count samples in this file
            if 'input_s' in data:
                n_s = data['input_s'].shape[0]
                total_s += n_s
            if 'input_z' in data:
                n_z = data['input_z'].shape[0]
                total_z += n_z
            
            self.samples.append({
                'file': batch_file,
                'has_s': 'input_s' in data,
                'has_z': 'input_z' in data,
            })
        
        print(f"Total s samples: {total_s:,}")
        print(f"Total z samples: {total_z:,}")
        
        # Build index for efficient access
        self._build_index()
    
    def _build_index(self):
        """Build index mapping dataset index to (file, sample_idx)."""
        self.index = []
        
        for sample_info in self.samples:
            data = np.load(sample_info['file'])
            
            if sample_info['has_s'] and sample_info['has_z']:
                n_samples = min(data['input_s'].shape[0], data['input_z'].shape[0])
                if self.max_samples_per_file:
                    n_samples = min(n_samples, self.max_samples_per_file)
                
                for i in range(n_samples):
                    self.index.append({
                        'file': sample_info['file'],
                        'idx': i,
                    })
        
        print(f"Dataset size: {len(self.index):,} samples")
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        sample_info = self.index[idx]
        data = np.load(sample_info['file'])
        sample_idx = sample_info['idx']
        
        # Load activations
        input_s = torch.from_numpy(data['input_s'][sample_idx]).float()
        output_s = torch.from_numpy(data['output_s'][sample_idx]).float()
        input_z = torch.from_numpy(data['input_z'][sample_idx]).float()
        output_z = torch.from_numpy(data['output_z'][sample_idx]).float()
        
        return {
            'input_s': input_s,
            'output_s': output_s,
            'input_z': input_z,
            'output_z': output_z,
        }


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_metrics = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        input_s = batch['input_s'].to(device)
        input_z = batch['input_z'].to(device)
        
        # Forward pass
        recon_s, recon_z, latent_s, latent_z = model(input_s, input_z)
        
        # Compute loss
        loss, metrics = model.compute_loss(
            input_s, input_z, recon_s, recon_z, latent_s, latent_z
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'l0_s': f"{metrics['l0_sparsity_s']:.3f}",
            'l0_z': f"{metrics['l0_sparsity_z']:.3f}",
        })
    
    # Average metrics
    n_batches = len(dataloader)
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
    avg_metrics['loss'] = total_loss / n_batches
    
    return avg_metrics


def train_transcoder(
    activation_dir,
    checkpoint_dir,
    log_file,
    latent_dim=2048,
    l1_coeff=1e-4,
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=100,
    checkpoint_every=500,
    device='cuda'
):
    """
    Train the joint transcoder.
    
    Args:
        activation_dir: Directory with activation .npz files
        checkpoint_dir: Directory to save checkpoints
        log_file: Path to training log file
        latent_dim: Latent dimension size
        l1_coeff: L1 sparsity coefficient
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of training epochs
        checkpoint_every: Save checkpoint every N steps
        device: Device to train on
    """
    # Create output directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("Initializing dataset...")
    dataset = ActivationDataset(activation_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print("Initializing model...")
    model = JointTranscoder(
        dim_s=384,
        dim_z=128,
        latent_dim=latent_dim,
        l1_coeff=l1_coeff
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Log file: {log_file}")
    
    training_log = []
    global_step = 0
    
    for epoch in range(1, num_epochs + 1):
        metrics = train_epoch(model, dataloader, optimizer, device, epoch)
        
        # Log metrics
        log_entry = {
            'epoch': epoch,
            'step': global_step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        training_log.append(log_entry)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {metrics['loss']:.6f}")
        print(f"  Recon Loss S: {metrics['recon_loss_s']:.6f}")
        print(f"  Recon Loss Z: {metrics['recon_loss_z']:.6f}")
        print(f"  L1 Loss: {metrics['l1_loss']:.6f}")
        print(f"  L0 Sparsity S: {metrics['l0_sparsity_s']:.3f}")
        print(f"  L0 Sparsity Z: {metrics['l0_sparsity_z']:.3f}")
        
        # Save checkpoint
        if epoch % (checkpoint_every // len(dataloader)) == 0 or epoch == num_epochs:
            checkpoint_path = Path(checkpoint_dir) / f"step_{global_step:06d}.pt"
            torch.save({
                'epoch': epoch,
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")
        
        global_step += len(dataloader)
        
        # Save log
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
    
    # Save final model
    final_path = Path(checkpoint_dir).parent / "transcoder_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'dim_s': 384,
            'dim_z': 128,
            'latent_dim': latent_dim,
            'l1_coeff': l1_coeff,
        },
        'final_metrics': metrics,
    }, final_path)
    print(f"\n✓ Saved final model: {final_path}")
    
    return model, training_log


def main():
    parser = argparse.ArgumentParser(description='Train joint transcoder')
    parser.add_argument('--activations', type=str, required=True, help='Activation directory')
    parser.add_argument('--checkpoints', type=str, default='pilot_checkpoints', help='Checkpoint directory')
    parser.add_argument('--log', type=str, default='training_log.txt', help='Log file')
    parser.add_argument('--latent-dim', type=int, default=2048, help='Latent dimension')
    parser.add_argument('--l1-coeff', type=float, default=1e-4, help='L1 coefficient')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--checkpoint-every', type=int, default=500, help='Checkpoint frequency')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    train_transcoder(
        activation_dir=args.activations,
        checkpoint_dir=args.checkpoints,
        log_file=args.log,
        latent_dim=args.latent_dim,
        l1_coeff=args.l1_coeff,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
        device=args.device
    )


if __name__ == "__main__":
    main()
