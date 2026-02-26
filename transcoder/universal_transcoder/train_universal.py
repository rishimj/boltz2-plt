"""
Train the Universal Transcoder with dual-pass loss function.

Training Strategy:
1. Forward pass 1: s1 (input_s) → predict y1, y2
2. Forward pass 2: s2 (output_s) → predict y1, y2
3. Combined loss:
   - Reconstruction: MSE between predictions and true y1, y2 (4 terms)
   - Consistency: MSE between predictions from s1 and s2 (2 terms)
   - AuxK: Auxiliary loss for dead neuron resurrection
"""
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from datetime import datetime
import time

from universal_model import UniversalTranscoder


class UniversalActivationDataset(Dataset):
    """Dataset for loading saved activations for Universal Transcoder."""
    
    def __init__(self, activation_dir):
        """
        Initialize dataset from saved activation files.
        
        Args:
            activation_dir: Directory containing batch_*.npz files
        """
        self.activation_dir = Path(activation_dir)
        
        # Find all batch files
        self.batch_files = sorted(self.activation_dir.glob("batch_*.npz"))
        print(f"Found {len(self.batch_files)} batch files")
        
        # Build index mapping dataset index to (file, sample_idx)
        self.index = []
        total_samples = 0
        
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            n_samples = data['input_s'].shape[0]
            total_samples += n_samples
            
            for i in range(n_samples):
                self.index.append({
                    'file': batch_file,
                    'idx': i,
                })
        
        print(f"Total samples: {total_samples}")
        print(f"Dataset size: {len(self.index)} samples")
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        sample_info = self.index[idx]
        data = np.load(sample_info['file'])
        sample_idx = sample_info['idx']
        
        # Load activations
        # s1 = input_s, s2 = output_s (single representations)
        # y1 = input_z, y2 = output_z (pair representations, already flattened)
        s1 = torch.from_numpy(data['input_s'][sample_idx]).float()   # [N, 384]
        s2 = torch.from_numpy(data['output_s'][sample_idx]).float()  # [N, 384]
        y1 = torch.from_numpy(data['input_z'][sample_idx]).float()   # [N², 128]
        y2 = torch.from_numpy(data['output_z'][sample_idx]).float()  # [N², 128]
        
        return {
            's1': s1,  # input_s
            's2': s2,  # output_s
            'y1': y1,  # input_z (target)
            'y2': y2,  # output_z (target)
        }


def train_universal_transcoder(args):
    """Main training function."""
    
    print("=" * 80)
    print("UNIVERSAL TRANSCODER TRAINING")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Training steps: {args.num_steps}")
    print(f"  d_model: {args.d_model}")
    print(f"  d_hidden: {args.d_hidden}")
    print(f"  d_pair: {args.d_pair}")
    print(f"  k: {args.k}")
    print(f"  auxk: {args.auxk}")
    print("=" * 80)
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Load dataset
    dataset = UniversalActivationDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Initialize model
    model = UniversalTranscoder(
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        d_pair=args.d_pair,
        k=args.k,
        auxk=args.auxk,
        batch_size=args.batch_size,
        dead_steps_threshold=args.dead_steps_threshold,
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print()
    
    # Optimizer (matching PLT)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,  # Match PLT
    )
    
    # Training loop
    print("Starting training...")
    print()
    
    start_time = time.time()
    step = 0
    epoch = 0
    
    # Create data iterator
    data_iter = iter(dataloader)
    
    # Metrics tracking
    metrics_history = []
    
    with tqdm(total=args.num_steps, desc="Training") as pbar:
        while step < args.num_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart iterator
                epoch += 1
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Move batch to device
            s1 = batch['s1'].to(device)  # [B, N, 384]
            s2 = batch['s2'].to(device)  # [B, N, 384]
            y1_true = batch['y1'].to(device)  # [B, N², 128]
            y2_true = batch['y2'].to(device)  # [B, N², 128]
            
            B, N, _ = s1.shape
            
            # Flatten single representations: [B, N, 384] → [B*N, 384]
            s1_flat = s1.reshape(B * N, -1)
            s2_flat = s2.reshape(B * N, -1)
            
            # Flatten pair representations: [B, N², 128] → [B*N², 128]
            y1_true_flat = y1_true.reshape(B * y1_true.shape[1], -1)
            y2_true_flat = y2_true.reshape(B * y2_true.shape[1], -1)
            
            # === DUAL-PASS FORWARD ===
            
            # Pass 1: s1 → y1, y2
            y1_pred1, y2_pred1, aux_y1_1, aux_y2_1, dead_mask = model(s1_flat)
            
            # Pass 2: s2 → y1, y2
            y1_pred2, y2_pred2, aux_y1_2, aux_y2_2, _ = model(s2_flat)
            
            # === COMBINED LOSS ===
            
            # 1. Reconstruction Loss (4 terms)
            # We need to match dimensions - predictions are [B*N, 128] but targets are [B*N², 128]
            # Solution: Expand predictions to match target size or use first N² predictions
            
            # For simplicity, let's reshape to match the pair structure
            # Each token position should predict all N² pairs
            N_sq = y1_true.shape[1]  # N²
            
            # We have B*N predictions of size 128 each
            # We need B*N² targets of size 128 each
            # Strategy: Repeat each prediction N times to match N² targets
            # Or: Take only first predictions to match target size
            
            # Let's use a simpler approach: average pool the tokens
            # Actually, the issue is that we're predicting pair representations from single representations
            # Let's just use the predictions as-is and repeat to match target dimensions
            
            # Better approach: Broadcast predictions across pair dimension
            y1_pred1_expanded = y1_pred1.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
            y2_pred1_expanded = y2_pred1.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
            y1_pred2_expanded = y1_pred2.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
            y2_pred2_expanded = y2_pred2.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
            
            loss_recon_y1_from_s1 = F.mse_loss(y1_pred1_expanded, y1_true_flat)
            loss_recon_y2_from_s1 = F.mse_loss(y2_pred1_expanded, y2_true_flat)
            loss_recon_y1_from_s2 = F.mse_loss(y1_pred2_expanded, y1_true_flat)
            loss_recon_y2_from_s2 = F.mse_loss(y2_pred2_expanded, y2_true_flat)
            
            loss_reconstruction = (
                loss_recon_y1_from_s1 + 
                loss_recon_y2_from_s1 + 
                loss_recon_y1_from_s2 + 
                loss_recon_y2_from_s2
            )
            
            # 2. Consistency Loss (2 terms)
            # Predictions from s1 and s2 should agree
            loss_consistency_y1 = F.mse_loss(y1_pred1, y1_pred2)
            loss_consistency_y2 = F.mse_loss(y2_pred1, y2_pred2)
            
            loss_consistency = loss_consistency_y1 + loss_consistency_y2
            
            # 3. AuxK Loss (for dead neuron resurrection, matching PLT)
            loss_auxk = 0.0
            auxk_coef = 1.0 / 32.0  # Match PLT
            
            if aux_y1_1 is not None and dead_mask.sum() > 0:
                # Residual from pass 1
                residual_y1_1 = (y1_true_flat - y1_pred1_expanded).detach()
                residual_y2_1 = (y2_true_flat - y2_pred1_expanded).detach()
                
                aux_y1_1_expanded = aux_y1_1.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
                aux_y2_1_expanded = aux_y2_1.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
                
                loss_auxk += F.mse_loss(aux_y1_1_expanded, residual_y1_1) * auxk_coef
                loss_auxk += F.mse_loss(aux_y2_1_expanded, residual_y2_1) * auxk_coef
                
                # Residual from pass 2
                residual_y1_2 = (y1_true_flat - y1_pred2_expanded).detach()
                residual_y2_2 = (y2_true_flat - y2_pred2_expanded).detach()
                
                aux_y1_2_expanded = aux_y1_2.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
                aux_y2_2_expanded = aux_y2_2.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
                
                loss_auxk += F.mse_loss(aux_y1_2_expanded, residual_y1_2) * auxk_coef
                loss_auxk += F.mse_loss(aux_y2_2_expanded, residual_y2_2) * auxk_coef
            
            # Total Loss
            loss_total = loss_reconstruction + loss_consistency + loss_auxk
            
            # Backward pass
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            # Normalize weights (matching PLT)
            model.norm_weights()
            
            # Log metrics
            step += 1
            
            metrics = {
                'step': step,
                'epoch': epoch,
                'loss_total': loss_total.item(),
                'loss_reconstruction': loss_reconstruction.item(),
                'loss_consistency': loss_consistency.item(),
                'loss_auxk': loss_auxk if isinstance(loss_auxk, float) else loss_auxk.item(),
                'dead_neurons': dead_mask.sum().item(),
            }
            metrics_history.append(metrics)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{loss_total.item():.4f}",
                'recon': f"{loss_reconstruction.item():.4f}",
                'cons': f"{loss_consistency.item():.4f}",
                'dead': dead_mask.sum().item(),
            })
            
            # Log every N steps
            if step % args.log_every == 0:
                print()
                print(f"Step {step}/{args.num_steps} (Epoch {epoch})")
                print(f"  Total Loss: {loss_total.item():.6f}")
                print(f"  Reconstruction Loss: {loss_reconstruction.item():.6f}")
                print(f"    - y1 from s1: {loss_recon_y1_from_s1.item():.6f}")
                print(f"    - y2 from s1: {loss_recon_y2_from_s1.item():.6f}")
                print(f"    - y1 from s2: {loss_recon_y1_from_s2.item():.6f}")
                print(f"    - y2 from s2: {loss_recon_y2_from_s2.item():.6f}")
                print(f"  Consistency Loss: {loss_consistency.item():.6f}")
                print(f"    - y1 consistency: {loss_consistency_y1.item():.6f}")
                print(f"    - y2 consistency: {loss_consistency_y2.item():.6f}")
                if isinstance(loss_auxk, torch.Tensor):
                    print(f"  AuxK Loss: {loss_auxk.item():.6f}")
                print(f"  Dead Neurons: {dead_mask.sum().item()} / {args.d_hidden}")
                print()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Time per step: {training_time/args.num_steps:.3f} seconds")
    print(f"Final total loss: {metrics_history[-1]['loss_total']:.6f}")
    print(f"Final reconstruction loss: {metrics_history[-1]['loss_reconstruction']:.6f}")
    print(f"Final consistency loss: {metrics_history[-1]['loss_consistency']:.6f}")
    print(f"Final dead neurons: {metrics_history[-1]['dead_neurons']} / {args.d_hidden}")
    print("=" * 80)
    print()
    
    # Save checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / 'universal_transcoder_final.pt'
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
        'training_time': training_time,
        'final_metrics': metrics_history[-1],
        'hyperparameters': vars(args),
        'metrics_history': metrics_history,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved to: {checkpoint_path}")
    
    # Save metrics to JSON
    metrics_path = checkpoint_dir / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'hyperparameters': vars(args),
            'training_time_seconds': training_time,
            'final_metrics': metrics_history[-1],
            'all_metrics': metrics_history,
        }, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")
    print()
    
    return model, metrics_history, training_time


def main():
    parser = argparse.ArgumentParser(description="Train Universal Transcoder")
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing training data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log every N steps')
    
    # Model
    parser.add_argument('--d_model', type=int, default=384,
                        help='Input dimension (single representation)')
    parser.add_argument('--d_hidden', type=int, default=2048,
                        help='Latent dimension')
    parser.add_argument('--d_pair', type=int, default=128,
                        help='Pair representation dimension')
    parser.add_argument('--k', type=int, default=16,
                        help='Top-K activation')
    parser.add_argument('--auxk', type=int, default=32,
                        help='Auxiliary K for dead neurons')
    parser.add_argument('--dead_steps_threshold', type=int, default=10000,
                        help='Steps before neuron considered dead')
    
    args = parser.parse_args()
    
    # Run training
    train_universal_transcoder(args)


if __name__ == '__main__':
    main()
