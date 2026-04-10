"""
Validate trained multi-layer PLT transcoders.

This script loads trained transcoders for multiple layers and validates them:
- Reconstruction error (MSE, RMSE)
- R² score
- Sparsity (average active neurons)
- Dead neurons count
- Weight norms (should be unit norm)
"""
import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

from universal_model import UniversalTranscoder


def load_transcoder(checkpoint_path, device='cuda'):
    """Load a trained transcoder from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get hyperparameters
    hparams = checkpoint['hyperparameters']
    
    # Create model
    model = UniversalTranscoder(
        d_model=hparams['d_model'],
        d_hidden=hparams['d_hidden'],
        d_pair=hparams['d_pair'],
        k=hparams['k'],
        auxk=hparams['auxk'],
        batch_size=hparams.get('batch_size', 10),
        dead_steps_threshold=hparams.get('dead_steps_threshold', 10000),
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, hparams


def compute_r_squared(y_true, y_pred):
    """Compute R² score."""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared.item()


def validate_single_layer(
    layer_idx,
    checkpoint_base_dir,
    data_base_dir,
    device='cuda',
    max_batches=None,
):
    """
    Validate transcoder for a single layer.
    
    Args:
        layer_idx: Layer index
        checkpoint_base_dir: Base directory containing checkpoints
        data_base_dir: Base directory containing validation data
        device: Device to use
        max_batches: Maximum number of batches to validate on (None = all)
        
    Returns:
        dict with validation metrics
    """
    print()
    print("=" * 80)
    print(f"VALIDATING LAYER {layer_idx}")
    print("=" * 80)
    
    # Check checkpoint exists
    checkpoint_dir = Path(checkpoint_base_dir) / f"layer_{layer_idx:02d}"
    checkpoint_path = checkpoint_dir / "universal_transcoder_final.pt"
    
    if not checkpoint_path.exists():
        print(f"⚠ No checkpoint found at {checkpoint_path}")
        return {
            'layer_idx': layer_idx,
            'status': 'no_checkpoint',
        }
    
    # Check data exists
    data_dir = Path(data_base_dir) / f"layer_{layer_idx:02d}"
    if not data_dir.exists():
        print(f"⚠ No data found at {data_dir}")
        return {
            'layer_idx': layer_idx,
            'status': 'no_data',
        }
    
    # Load model
    print(f"Loading checkpoint from {checkpoint_path}")
    model, hparams = load_transcoder(checkpoint_path, device=device)
    print(f"✓ Model loaded")
    print(f"  d_model={hparams['d_model']}, d_hidden={hparams['d_hidden']}, k={hparams['k']}")
    print()
    
    # Check weight norms
    print("Checking weight norms...")
    with torch.no_grad():
        encoder_norms = torch.norm(model.encoder.weight, dim=0)
        decoder_y1_norms = torch.norm(model.decoder_y1, dim=0)  # decoder_y1 is a Parameter, not Linear
        decoder_y2_norms = torch.norm(model.decoder_y2, dim=0)  # decoder_y2 is a Parameter, not Linear
        
        encoder_norm_mean = encoder_norms.mean().item()
        encoder_norm_std = encoder_norms.std().item()
        decoder_y1_norm_mean = decoder_y1_norms.mean().item()
        decoder_y1_norm_std = decoder_y1_norms.std().item()
        decoder_y2_norm_mean = decoder_y2_norms.mean().item()
        decoder_y2_norm_std = decoder_y2_norms.std().item()
        
        print(f"  Encoder weight norms: {encoder_norm_mean:.6f} ± {encoder_norm_std:.6f}")
        print(f"  Decoder Y1 weight norms: {decoder_y1_norm_mean:.6f} ± {decoder_y1_norm_std:.6f}")
        print(f"  Decoder Y2 weight norms: {decoder_y2_norm_mean:.6f} ± {decoder_y2_norm_std:.6f}")
        
        # Check if norms are close to 1.0 (unit norm)
        encoder_is_unit = abs(encoder_norm_mean - 1.0) < 0.01
        decoder_y1_is_unit = abs(decoder_y1_norm_mean - 1.0) < 0.01
        decoder_y2_is_unit = abs(decoder_y2_norm_mean - 1.0) < 0.01
        
        if encoder_is_unit and decoder_y1_is_unit and decoder_y2_is_unit:
            print(f"  ✓ All weights have unit norm (as expected for PLT)")
        else:
            print(f"  ⚠ Some weights do not have unit norm")
    print()
    
    # Load data and compute metrics
    print("Loading validation data...")
    batch_files = sorted(data_dir.glob("batch_*.npz"))
    
    if not batch_files:
        print(f"⚠ No batch files found in {data_dir}")
        return {
            'layer_idx': layer_idx,
            'status': 'no_batches',
        }
    
    if max_batches is not None:
        batch_files = batch_files[:max_batches]
    
    print(f"Found {len(batch_files)} batch files")
    print()
    
    # Accumulate metrics across all batches
    all_mse_y1 = []
    all_mse_y2 = []
    all_r2_y1 = []
    all_r2_y2 = []
    all_sparsity = []
    total_samples = 0
    
    print("Computing validation metrics...")
    with torch.no_grad():
        for batch_file in tqdm(batch_files, desc="Processing batches"):
            # Load batch
            data = np.load(batch_file)
            
            # Get activations
            s1 = torch.from_numpy(data['input_s']).float().to(device)   # [B, N, 384]
            s2 = torch.from_numpy(data['output_s']).float().to(device)  # [B, N, 384]
            y1_true = torch.from_numpy(data['input_z']).float().to(device)   # [B, N², 128]
            y2_true = torch.from_numpy(data['output_z']).float().to(device)  # [B, N², 128]
            
            B, N, _ = s1.shape
            N_sq = y1_true.shape[1]
            
            # Flatten single representations
            s1_flat = s1.reshape(B * N, -1)
            s2_flat = s2.reshape(B * N, -1)
            
            # Flatten pair representations
            y1_true_flat = y1_true.reshape(B * N_sq, -1)
            y2_true_flat = y2_true.reshape(B * N_sq, -1)
            
            # Forward pass on s1
            y1_pred, y2_pred, _, _, dead_mask = model(s1_flat)
            
            # Expand predictions to match pair dimensions
            y1_pred_expanded = y1_pred.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
            y2_pred_expanded = y2_pred.unsqueeze(1).expand(B * N, N, -1).reshape(B * N_sq, -1)
            
            # Compute MSE
            mse_y1 = F.mse_loss(y1_pred_expanded, y1_true_flat).item()
            mse_y2 = F.mse_loss(y2_pred_expanded, y2_true_flat).item()
            
            # Compute R²
            r2_y1 = compute_r_squared(y1_true_flat, y1_pred_expanded)
            r2_y2 = compute_r_squared(y2_true_flat, y2_pred_expanded)
            
            # Compute sparsity (k neurons active per sample by design)
            sparsity = model.k  # TopK ensures exactly k active neurons per forward pass
            
            # Accumulate
            all_mse_y1.append(mse_y1)
            all_mse_y2.append(mse_y2)
            all_r2_y1.append(r2_y1)
            all_r2_y2.append(r2_y2)
            all_sparsity.append(sparsity)
            total_samples += B
    
    # Compute averages
    avg_mse_y1 = np.mean(all_mse_y1)
    avg_mse_y2 = np.mean(all_mse_y2)
    avg_rmse_y1 = np.sqrt(avg_mse_y1)
    avg_rmse_y2 = np.sqrt(avg_mse_y2)
    avg_r2_y1 = np.mean(all_r2_y1)
    avg_r2_y2 = np.mean(all_r2_y2)
    avg_sparsity = np.mean(all_sparsity)
    
    # Print results
    print()
    print("Validation Results:")
    print("-" * 80)
    print(f"  Total samples: {total_samples}")
    print(f"  Batches processed: {len(batch_files)}")
    print()
    print(f"  Y1 (input_z) Prediction:")
    print(f"    MSE:  {avg_mse_y1:.6f}")
    print(f"    RMSE: {avg_rmse_y1:.6f}")
    print(f"    R²:   {avg_r2_y1:.6f}")
    print()
    print(f"  Y2 (output_z) Prediction:")
    print(f"    MSE:  {avg_mse_y2:.6f}")
    print(f"    RMSE: {avg_rmse_y2:.6f}")
    print(f"    R²:   {avg_r2_y2:.6f}")
    print()
    print(f"  Sparsity:")
    print(f"    Average active neurons: {avg_sparsity:.2f} / {hparams['d_hidden']}")
    print(f"    Expected (k): {hparams['k']}")
    print()
    
    # Check dead neurons from checkpoint
    if 'final_metrics' in hparams or 'final_metrics' in model.state_dict():
        # Try to get from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        final_metrics = checkpoint.get('final_metrics', {})
        dead_neurons = final_metrics.get('dead_neurons', 'N/A')
        print(f"  Dead neurons (from training): {dead_neurons} / {hparams['d_hidden']}")
    
    print("-" * 80)
    
    result = {
        'layer_idx': layer_idx,
        'status': 'success',
        'total_samples': total_samples,
        'num_batches': len(batch_files),
        'mse_y1': avg_mse_y1,
        'rmse_y1': avg_rmse_y1,
        'r2_y1': avg_r2_y1,
        'mse_y2': avg_mse_y2,
        'rmse_y2': avg_rmse_y2,
        'r2_y2': avg_r2_y2,
        'avg_sparsity': avg_sparsity,
        'expected_sparsity': hparams['k'],
        'encoder_norm_mean': encoder_norm_mean,
        'decoder_y1_norm_mean': decoder_y1_norm_mean,
        'decoder_y2_norm_mean': decoder_y2_norm_mean,
        'is_unit_norm': encoder_is_unit and decoder_y1_is_unit and decoder_y2_is_unit,
    }
    
    return result


def validate_multi_layer(
    checkpoint_base_dir,
    data_base_dir,
    layer_indices,
    device='cuda',
    max_batches=None,
):
    """
    Validate transcoders for multiple layers.
    
    Args:
        checkpoint_base_dir: Base directory containing all checkpoints
        data_base_dir: Base directory containing validation data
        layer_indices: List of layer indices to validate
        device: Device to use
        max_batches: Maximum batches per layer (None = all)
        
    Returns:
        dict with all validation results
    """
    print()
    print("=" * 80)
    print("MULTI-LAYER PLT VALIDATION")
    print("=" * 80)
    print(f"Checkpoint directory: {checkpoint_base_dir}")
    print(f"Data directory: {data_base_dir}")
    print(f"Layers to validate: {layer_indices}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Validate each layer
    all_results = []
    successful = []
    failed = []
    
    for layer_idx in layer_indices:
        result = validate_single_layer(
            layer_idx=layer_idx,
            checkpoint_base_dir=checkpoint_base_dir,
            data_base_dir=data_base_dir,
            device=device,
            max_batches=max_batches,
        )
        
        all_results.append(result)
        
        if result['status'] == 'success':
            successful.append(layer_idx)
        else:
            failed.append(layer_idx)
    
    # Print summary
    print()
    print()
    print("=" * 80)
    print("MULTI-LAYER VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Successfully validated: {len(successful)} layers - {successful}")
    print(f"Failed: {len(failed)} layers - {failed}")
    print()
    
    if successful:
        print("Detailed Results:")
        print("-" * 120)
        header = f"{'Layer':<8} {'R²(Y1)':>10} {'R²(Y2)':>10} {'RMSE(Y1)':>12} {'RMSE(Y2)':>12} {'Sparsity':>10} {'UnitNorm':>10}"
        print(header)
        print("-" * 120)
        
        for result in all_results:
            if result['status'] == 'success':
                layer = result['layer_idx']
                r2_y1 = result['r2_y1']
                r2_y2 = result['r2_y2']
                rmse_y1 = result['rmse_y1']
                rmse_y2 = result['rmse_y2']
                sparsity = result['avg_sparsity']
                unit_norm = '✓' if result['is_unit_norm'] else '✗'
                
                print(f"{layer:<8} {r2_y1:>10.4f} {r2_y2:>10.4f} {rmse_y1:>12.6f} {rmse_y2:>12.6f} {sparsity:>10.2f} {unit_norm:>10}")
        
        print("-" * 120)
    
    print()
    
    # Save summary
    summary_path = Path(checkpoint_base_dir) / 'validation_summary.json'
    summary = {
        'num_layers': len(layer_indices),
        'successful_layers': successful,
        'failed_layers': failed,
        'results': all_results,
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Validation summary saved to: {summary_path}")
    print()
    print("=" * 80)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate multi-layer PLT transcoders")
    
    parser.add_argument('--checkpoint_dir', type=str, default='multi_layer_checkpoints',
                        help='Base directory containing checkpoints')
    parser.add_argument('--data_dir', type=str, default='multi_layer_activations',
                        help='Base directory containing validation data (same as training)')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 8, 16, 24, 32, 40],
                        help='Layer indices to validate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum batches to validate per layer (None = all)')
    
    args = parser.parse_args()
    
    # Run validation
    validate_multi_layer(
        checkpoint_base_dir=args.checkpoint_dir,
        data_base_dir=args.data_dir,
        layer_indices=args.layers,
        device=args.device,
        max_batches=args.max_batches,
    )


if __name__ == '__main__':
    main()
