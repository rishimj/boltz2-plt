"""Train transcoder with dynamic batching for variable sequence lengths."""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import json
from datetime import datetime

from model import JointTranscoder


class VariableLengthActivationDataset(Dataset):
    """Dataset for loading activations with variable sequence lengths."""
    
    def __init__(self, activation_dir):
        self.activation_dir = Path(activation_dir)
        self.files = sorted(self.activation_dir.glob("protein_*.npz"))
        print(f"Found {len(self.files)} activation files")
        
        # Analyze sequence lengths
        self.seq_lengths = []
        for f in self.files:
            data = np.load(f)
            if 'input_s' in data:
                self.seq_lengths.append(data['input_s'].shape[1])
        
        if self.seq_lengths:
            print(f"Sequence lengths: min={min(self.seq_lengths)}, max={max(self.seq_lengths)}, mean={np.mean(self.seq_lengths):.1f}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        
        # Load single sample (already shape [1, seq_len, dim])
        input_s = torch.from_numpy(data['input_s'][0]).float()  # [seq_len, 384]
        output_s = torch.from_numpy(data['output_s'][0]).float()
        input_z = torch.from_numpy(data['input_z'][0]).float()  # [seq_len², 128]
        output_z = torch.from_numpy(data['output_z'][0]).float()
        
        return {
            'input_s': input_s,
            'output_s': output_s,
            'input_z': input_z,
            'output_z': output_z,
            'seq_len': input_s.shape[0],
        }


def collate_variable_length(batch):
    """Collate function that pads sequences to max length in batch."""
    # Find max sequence length in this batch
    max_len = max(item['seq_len'] for item in batch)
    
    batch_size = len(batch)
    
    # Pre-allocate tensors
    input_s = torch.zeros(batch_size, max_len, 384)
    output_s = torch.zeros(batch_size, max_len, 384)
    input_z = torch.zeros(batch_size, max_len * max_len, 128)
    output_z = torch.zeros(batch_size, max_len * max_len, 128)
    masks_s = torch.zeros(batch_size, max_len, dtype=torch.bool)
    masks_z = torch.zeros(batch_size, max_len * max_len, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        
        # Copy s data
        input_s[i, :seq_len] = item['input_s']
        output_s[i, :seq_len] = item['output_s']
        masks_s[i, :seq_len] = True
        
        # Copy z data (seq_len²)
        z_len = seq_len * seq_len
        input_z[i, :z_len] = item['input_z']
        output_z[i, :z_len] = item['output_z']
        masks_z[i, :z_len] = True
    
    return {
        'input_s': input_s,
        'output_s': output_s,
        'input_z': input_z,
        'output_z': output_z,
        'mask_s': masks_s,
        'mask_z': masks_z,
    }


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch with masked loss."""
    model.train()
    
    total_loss = 0
    total_metrics = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        input_s = batch['input_s'].to(device)
        input_z = batch['input_z'].to(device)
        mask_s = batch['mask_s'].to(device)
        mask_z = batch['mask_z'].to(device)
        
        # Forward pass
        recon_s, recon_z, latent_s, latent_z = model(input_s, input_z)
        
        # Compute masked loss
        loss, metrics = compute_masked_loss(
            model, input_s, input_z, recon_s, recon_z,
            latent_s, latent_z, mask_s, mask_z
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


def compute_masked_loss(model, input_s, input_z, recon_s, recon_z, latent_s, latent_z, mask_s, mask_z):
    """Compute loss only on valid (non-padded) positions."""
    import torch.nn.functional as F
    
    # Masked reconstruction losses
    diff_s = (recon_s - input_s) ** 2
    diff_z = (recon_z - input_z) ** 2
    
    # Apply masks and average only over valid positions
    recon_loss_s = (diff_s * mask_s.unsqueeze(-1)).sum() / mask_s.sum()
    recon_loss_z = (diff_z * mask_z.unsqueeze(-1)).sum() / mask_z.sum()
    
    recon_loss = recon_loss_s + recon_loss_z
    
    # L1 sparsity on valid positions
    l1_loss_s = (torch.abs(latent_s) * mask_s.unsqueeze(-1)).sum() / mask_s.sum()
    l1_loss_z = (torch.abs(latent_z) * mask_z.unsqueeze(-1)).sum() / mask_z.sum()
    l1_loss = l1_loss_s + l1_loss_z
    
    total_loss = recon_loss + model.l1_coeff * l1_loss
    
    # L0 sparsity
    with torch.no_grad():
        l0_s = ((latent_s > 1e-6).float() * mask_s.unsqueeze(-1)).sum() / mask_s.sum()
        l0_z = ((latent_z > 1e-6).float() * mask_z.unsqueeze(-1)).sum() / mask_z.sum()
    
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


def train_transcoder(
    activation_dir,
    checkpoint_dir,
    log_file,
    latent_dim=2048,
    l1_coeff=1e-4,
    learning_rate=1e-3,
    batch_size=4,  # Smaller batch for variable lengths
    num_epochs=100,
    device='cuda'
):
    """Train the joint transcoder on variable-length activations."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("Initializing dataset...")
    dataset = VariableLengthActivationDataset(activation_dir)
    
    if len(dataset) == 0:
        print("ERROR: No activation files found!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_variable_length,
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
    
    for epoch in range(1, num_epochs + 1):
        metrics = train_epoch(model, dataloader, optimizer, device, epoch)
        
        # Log metrics
        log_entry = {
            'epoch': epoch,
            'timestamp': str(datetime.now()),
            **metrics
        }
        training_log.append(log_entry)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Recon S: {metrics['recon_loss_s']:.4f}, Z: {metrics['recon_loss_z']:.4f}")
        print(f"  L1: {metrics['l1_loss']:.4f}")
        print(f"  L0 Sparsity: S={metrics['l0_sparsity_s']:.3f}, Z={metrics['l0_sparsity_z']:.3f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = Path(checkpoint_dir) / f"epoch_{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save log
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
    
    # Save final model
    final_path = Path(checkpoint_dir).parent / "transcoder_real_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Training complete! Final model saved to: {final_path}")
    print(f"✓ Training log saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_dir", default="real_activations")
    parser.add_argument("--checkpoint_dir", default="real_model")
    parser.add_argument("--log_file", default="training_log_real.json")
    parser.add_argument("--latent_dim", type=int, default=2048)
    parser.add_argument("--l1_coeff", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    train_transcoder(
        activation_dir=args.activation_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        latent_dim=args.latent_dim,
        l1_coeff=args.l1_coeff,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device
    )
