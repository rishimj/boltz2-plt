"""
Evaluation script for Universal Transcoder.

Computes comprehensive metrics:
- Reconstruction quality (MSE, NMSE, R²)
- Sparsity metrics (L0, L1, dead neurons)
- Latent space analysis (feature activation patterns)
- Consistency metrics (dual-pass agreement)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from universal_model import UniversalTranscoder
from train_universal import UniversalActivationDataset


def compute_r2_score(predictions, targets):
    """Compute R² score (coefficient of determination)."""
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return r2.item()


def evaluate_model(model, dataloader, device):
    """
    Comprehensive evaluation of Universal Transcoder.
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Accumulators
    total_samples = 0
    
    # Reconstruction metrics
    mse_y1_from_s1_total = 0.0
    mse_y2_from_s1_total = 0.0
    mse_y1_from_s2_total = 0.0
    mse_y2_from_s2_total = 0.0
    
    # Consistency metrics
    consistency_y1_total = 0.0
    consistency_y2_total = 0.0
    
    # Sparsity metrics
    l0_total = 0.0
    l1_total = 0.0
    
    # Latent activation patterns
    all_latents_s1 = []
    all_latents_s2 = []
    
    # Dead neuron tracking
    neuron_activation_count = torch.zeros(model.d_hidden, device=device)
    
    # R² scores
    r2_y1_from_s1_total = 0.0
    r2_y2_from_s1_total = 0.0
    r2_y1_from_s2_total = 0.0
    r2_y2_from_s2_total = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            s1 = batch['s1'].to(device)  # [B, N, 384]
            s2 = batch['s2'].to(device)  # [B, N, 384]
            y1_true = batch['y1'].to(device)  # [B, N², 128]
            y2_true = batch['y2'].to(device)  # [B, N², 128]
            
            B, N, _ = s1.shape
            
            # Flatten
            s1_flat = s1.reshape(B * N, -1)
            s2_flat = s2.reshape(B * N, -1)
            y1_true_flat = y1_true.reshape(B * y1_true.shape[1], -1)
            y2_true_flat = y2_true.reshape(B * y2_true.shape[1], -1)
            
            # Forward pass 1: s1 → y1, y2
            y1_pred1, y2_pred1, aux_y1_1, aux_y2_1, dead_mask1 = model(s1_flat)
            
            # Forward pass 2: s2 → y1, y2
            y1_pred2, y2_pred2, aux_y1_2, aux_y2_2, dead_mask2 = model(s2_flat)
            
            # Get latents for analysis (before TopK)
            with torch.no_grad():
                # Re-run encoder to get pre-TopK activations
                x_norm1, mu1, std1 = model.LN(s1_flat)
                x_centered1 = x_norm1 - model.b_pre
                pre_acts1 = model.encoder(x_centered1) + model.b_enc
                
                x_norm2, mu2, std2 = model.LN(s2_flat)
                x_centered2 = x_norm2 - model.b_pre
                pre_acts2 = model.encoder(x_centered2) + model.b_enc
                
                all_latents_s1.append(pre_acts1.cpu())
                all_latents_s2.append(pre_acts2.cpu())
                
                # Track which neurons are active
                latents1 = model.topK_activation(pre_acts1, k=model.k)
                latents2 = model.topK_activation(pre_acts2, k=model.k)
                neuron_activation_count += (latents1 > 0).sum(dim=0)
                neuron_activation_count += (latents2 > 0).sum(dim=0)
            
            # Expand predictions to match target dimensions
            N_sq = y1_true.shape[1]
            y1_pred1_exp = y1_pred1.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
            y2_pred1_exp = y2_pred1.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
            y1_pred2_exp = y1_pred2.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
            y2_pred2_exp = y2_pred2.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
            
            # Reconstruction metrics
            mse_y1_from_s1 = F.mse_loss(y1_pred1_exp, y1_true_flat)
            mse_y2_from_s1 = F.mse_loss(y2_pred1_exp, y2_true_flat)
            mse_y1_from_s2 = F.mse_loss(y1_pred2_exp, y1_true_flat)
            mse_y2_from_s2 = F.mse_loss(y2_pred2_exp, y2_true_flat)
            
            mse_y1_from_s1_total += mse_y1_from_s1.item() * B
            mse_y2_from_s1_total += mse_y2_from_s1.item() * B
            mse_y1_from_s2_total += mse_y1_from_s2.item() * B
            mse_y2_from_s2_total += mse_y2_from_s2.item() * B
            
            # R² scores
            r2_y1_from_s1_total += compute_r2_score(y1_pred1_exp, y1_true_flat) * B
            r2_y2_from_s1_total += compute_r2_score(y2_pred1_exp, y2_true_flat) * B
            r2_y1_from_s2_total += compute_r2_score(y1_pred2_exp, y1_true_flat) * B
            r2_y2_from_s2_total += compute_r2_score(y2_pred2_exp, y2_true_flat) * B
            
            # Consistency metrics
            consistency_y1 = F.mse_loss(y1_pred1, y1_pred2)
            consistency_y2 = F.mse_loss(y2_pred1, y2_pred2)
            
            consistency_y1_total += consistency_y1.item() * B
            consistency_y2_total += consistency_y2.item() * B
            
            # Sparsity metrics (on sparse latents)
            latents1_sparse = model.topK_activation(pre_acts1, k=model.k)
            latents2_sparse = model.topK_activation(pre_acts2, k=model.k)
            
            l0_batch = ((latents1_sparse > 1e-6).float().mean() + (latents2_sparse > 1e-6).float().mean()) / 2
            l1_batch = (latents1_sparse.abs().mean() + latents2_sparse.abs().mean()) / 2
            
            l0_total += l0_batch.item() * B
            l1_total += l1_batch.item() * B
            
            total_samples += B
    
    # Compute averages
    metrics = {
        # Reconstruction MSE
        'mse_y1_from_s1': mse_y1_from_s1_total / total_samples,
        'mse_y2_from_s1': mse_y2_from_s1_total / total_samples,
        'mse_y1_from_s2': mse_y1_from_s2_total / total_samples,
        'mse_y2_from_s2': mse_y2_from_s2_total / total_samples,
        'mse_reconstruction_avg': (mse_y1_from_s1_total + mse_y2_from_s1_total + 
                                   mse_y1_from_s2_total + mse_y2_from_s2_total) / (4 * total_samples),
        
        # R² scores
        'r2_y1_from_s1': r2_y1_from_s1_total / total_samples,
        'r2_y2_from_s1': r2_y2_from_s1_total / total_samples,
        'r2_y1_from_s2': r2_y1_from_s2_total / total_samples,
        'r2_y2_from_s2': r2_y2_from_s2_total / total_samples,
        'r2_avg': (r2_y1_from_s1_total + r2_y2_from_s1_total +
                   r2_y1_from_s2_total + r2_y2_from_s2_total) / (4 * total_samples),
        
        # Consistency
        'consistency_y1': consistency_y1_total / total_samples,
        'consistency_y2': consistency_y2_total / total_samples,
        'consistency_avg': (consistency_y1_total + consistency_y2_total) / (2 * total_samples),
        
        # Sparsity
        'l0_sparsity': l0_total / total_samples,
        'l0_sparsity_percent': (l0_total / total_samples) * 100,
        'l1_activation': l1_total / total_samples,
        
        # Dead neurons
        'dead_neurons': (neuron_activation_count == 0).sum().item(),
        'dead_neurons_percent': ((neuron_activation_count == 0).sum().item() / model.d_hidden) * 100,
        'active_neurons': (neuron_activation_count > 0).sum().item(),
        
        # Total samples
        'num_samples': total_samples,
    }
    
    # Latent space analysis
    all_latents_s1 = torch.cat(all_latents_s1, dim=0)  # [total_tokens, d_hidden]
    all_latents_s2 = torch.cat(all_latents_s2, dim=0)
    
    latent_stats = {
        'latent_mean': all_latents_s1.mean().item(),
        'latent_std': all_latents_s1.std().item(),
        'latent_max': all_latents_s1.max().item(),
        'latent_min': all_latents_s1.min().item(),
        'neuron_activation_count': neuron_activation_count.cpu().numpy().tolist(),
    }
    
    return metrics, latent_stats


def print_evaluation_report(metrics, latent_stats, output_path=None):
    """Print a formatted evaluation report."""
    
    report = []
    report.append("=" * 80)
    report.append("UNIVERSAL TRANSCODER EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append("1. RECONSTRUCTION QUALITY")
    report.append("-" * 80)
    report.append(f"  MSE (Mean Squared Error):")
    report.append(f"    y1 from s1: {metrics['mse_y1_from_s1']:.6f}")
    report.append(f"    y2 from s1: {metrics['mse_y2_from_s1']:.6f}")
    report.append(f"    y1 from s2: {metrics['mse_y1_from_s2']:.6f}")
    report.append(f"    y2 from s2: {metrics['mse_y2_from_s2']:.6f}")
    report.append(f"    Average:    {metrics['mse_reconstruction_avg']:.6f}")
    report.append("")
    report.append(f"  R² Score (Variance Explained):")
    report.append(f"    y1 from s1: {metrics['r2_y1_from_s1']:.4f}")
    report.append(f"    y2 from s1: {metrics['r2_y2_from_s1']:.4f}")
    report.append(f"    y1 from s2: {metrics['r2_y1_from_s2']:.4f}")
    report.append(f"    y2 from s2: {metrics['r2_y2_from_s2']:.4f}")
    report.append(f"    Average:    {metrics['r2_avg']:.4f}")
    report.append("")
    
    report.append("2. CONSISTENCY (Dual-Pass Agreement)")
    report.append("-" * 80)
    report.append(f"  y1 consistency: {metrics['consistency_y1']:.6f}")
    report.append(f"  y2 consistency: {metrics['consistency_y2']:.6f}")
    report.append(f"  Average:        {metrics['consistency_avg']:.6f}")
    report.append("")
    
    report.append("3. SPARSITY METRICS")
    report.append("-" * 80)
    report.append(f"  L0 Sparsity (fraction active): {metrics['l0_sparsity']:.6f} ({metrics['l0_sparsity_percent']:.2f}%)")
    report.append(f"  Expected for k=16/2048:        0.007813 (0.78%)")
    report.append(f"  L1 Activation (magnitude):     {metrics['l1_activation']:.6f}")
    report.append("")
    
    report.append("4. DEAD NEURONS")
    report.append("-" * 80)
    report.append(f"  Dead neurons:   {metrics['dead_neurons']} / 2048 ({metrics['dead_neurons_percent']:.2f}%)")
    report.append(f"  Active neurons: {metrics['active_neurons']} / 2048")
    report.append("")
    
    report.append("5. LATENT SPACE STATISTICS")
    report.append("-" * 80)
    report.append(f"  Mean activation:  {latent_stats['latent_mean']:.6f}")
    report.append(f"  Std deviation:    {latent_stats['latent_std']:.6f}")
    report.append(f"  Max activation:   {latent_stats['latent_max']:.6f}")
    report.append(f"  Min activation:   {latent_stats['latent_min']:.6f}")
    report.append("")
    
    report.append("6. EVALUATION SUMMARY")
    report.append("-" * 80)
    report.append(f"  Total samples evaluated: {metrics['num_samples']}")
    report.append("")
    
    report.append("=" * 80)
    
    full_report = "\n".join(report)
    print(full_report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(full_report)
        print(f"\nReport saved to: {output_path}")
    
    return full_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Universal Transcoder")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing evaluation data')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load model
    hparams = checkpoint['hyperparameters']
    model = UniversalTranscoder(
        d_model=hparams['d_model'],
        d_hidden=hparams['d_hidden'],
        d_pair=hparams['d_pair'],
        k=hparams['k'],
        auxk=hparams['auxk'],
        batch_size=hparams['batch_size'],
        dead_steps_threshold=hparams['dead_steps_threshold'],
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Load data
    dataset = UniversalActivationDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Evaluate
    print("Running evaluation...")
    metrics, latent_stats = evaluate_model(model, dataloader, device)
    
    # Print report
    report_path = output_dir / 'evaluation_report.txt'
    print_evaluation_report(metrics, latent_stats, report_path)
    
    # Save metrics as JSON
    json_path = output_dir / 'evaluation_metrics.json'
    with open(json_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'latent_stats': latent_stats,
            'checkpoint': str(args.checkpoint),
            'data_dir': str(args.data_dir),
        }, f, indent=2)
    print(f"Metrics saved to: {json_path}")


if __name__ == '__main__':
    main()
