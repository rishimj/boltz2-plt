"""
Train separate Piecewise Linear Transcoders for multiple pairformer layers.

This script orchestrates training of independent transcoders for layers 0, 8, 16, 24, 32, 40.
Each layer gets its own transcoder with separate feature dictionary.
"""
import os
import sys
import argparse
from pathlib import Path
import json
import time
from datetime import datetime
import torch

# Import the existing training function
from train_universal import train_universal_transcoder


class Args:
    """Container for training arguments."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def train_single_layer(
    layer_idx,
    data_base_dir,
    checkpoint_base_dir,
    batch_size=10,
    num_steps=100,
    lr=1e-3,
    log_every=10,
    d_model=384,
    d_hidden=2048,
    d_pair=128,
    k=16,
    auxk=32,
    dead_steps_threshold=10000,
):
    """
    Train a transcoder for a single layer.
    
    Args:
        layer_idx: Layer index to train transcoder for
        data_base_dir: Base directory containing multi-layer activations
        checkpoint_base_dir: Base directory to save checkpoints
        ... (other training hyperparameters)
        
    Returns:
        dict with training results
    """
    print()
    print("=" * 80)
    print(f"TRAINING TRANSCODER FOR LAYER {layer_idx}")
    print("=" * 80)
    
    # Setup directories
    layer_data_dir = Path(data_base_dir) / f"layer_{layer_idx:02d}"
    layer_checkpoint_dir = Path(checkpoint_base_dir) / f"layer_{layer_idx:02d}"
    
    # Check if data exists
    if not layer_data_dir.exists():
        print(f"⚠ No data found for layer {layer_idx} at {layer_data_dir}")
        print(f"Skipping layer {layer_idx}")
        return {
            'layer_idx': layer_idx,
            'status': 'skipped',
            'reason': 'no_data',
        }
    
    # Count data files
    npz_files = list(layer_data_dir.glob("batch_*.npz"))
    if len(npz_files) == 0:
        print(f"⚠ No batch files found for layer {layer_idx}")
        print(f"Skipping layer {layer_idx}")
        return {
            'layer_idx': layer_idx,
            'status': 'skipped',
            'reason': 'no_batches',
        }
    
    print(f"Found {len(npz_files)} batch files for layer {layer_idx}")
    print()
    
    # Create args object
    args = Args(
        data_dir=str(layer_data_dir),
        checkpoint_dir=str(layer_checkpoint_dir),
        batch_size=batch_size,
        num_steps=num_steps,
        lr=lr,
        log_every=log_every,
        d_model=d_model,
        d_hidden=d_hidden,
        d_pair=d_pair,
        k=k,
        auxk=auxk,
        dead_steps_threshold=dead_steps_threshold,
    )
    
    # Train
    start_time = time.time()
    try:
        model, metrics_history, training_time = train_universal_transcoder(args)
        
        # Get final metrics
        final_metrics = metrics_history[-1]
        
        result = {
            'layer_idx': layer_idx,
            'status': 'success',
            'training_time_seconds': training_time,
            'final_loss_total': final_metrics['loss_total'],
            'final_loss_reconstruction': final_metrics['loss_reconstruction'],
            'final_loss_consistency': final_metrics['loss_consistency'],
            'final_loss_auxk': final_metrics['loss_auxk'],
            'final_dead_neurons': final_metrics['dead_neurons'],
            'num_steps': num_steps,
            'checkpoint_dir': str(layer_checkpoint_dir),
        }
        
        print()
        print(f"✓ Layer {layer_idx} training complete!")
        print(f"  Time: {training_time:.2f}s ({training_time/60:.2f} min)")
        print(f"  Final loss: {final_metrics['loss_total']:.6f}")
        print(f"  Dead neurons: {final_metrics['dead_neurons']} / {d_hidden}")
        
        return result
        
    except Exception as e:
        print()
        print(f"❌ Error training layer {layer_idx}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'layer_idx': layer_idx,
            'status': 'failed',
            'error': str(e),
        }


def train_multi_layer(
    data_base_dir,
    checkpoint_base_dir,
    layer_indices,
    batch_size=10,
    num_steps=100,
    lr=1e-3,
    log_every=10,
    d_model=384,
    d_hidden=2048,
    d_pair=128,
    k=16,
    auxk=32,
    dead_steps_threshold=10000,
):
    """
    Train transcoders for multiple layers.
    
    Args:
        data_base_dir: Base directory with multi-layer activations
        checkpoint_base_dir: Base directory to save all checkpoints
        layer_indices: List of layer indices to train
        ... (other hyperparameters)
        
    Returns:
        dict with all training results
    """
    print()
    print("=" * 80)
    print("MULTI-LAYER PLT TRAINING")
    print("=" * 80)
    print(f"Data directory: {data_base_dir}")
    print(f"Checkpoint directory: {checkpoint_base_dir}")
    print(f"Layers to train: {layer_indices}")
    print(f"Training steps per layer: {num_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Model: d_model={d_model}, d_hidden={d_hidden}, k={k}, auxk={auxk}")
    print("=" * 80)
    
    # Create checkpoint base directory
    os.makedirs(checkpoint_base_dir, exist_ok=True)
    
    # Train each layer sequentially
    all_results = []
    successful_layers = []
    failed_layers = []
    skipped_layers = []
    
    overall_start = time.time()
    
    for layer_idx in layer_indices:
        result = train_single_layer(
            layer_idx=layer_idx,
            data_base_dir=data_base_dir,
            checkpoint_base_dir=checkpoint_base_dir,
            batch_size=batch_size,
            num_steps=num_steps,
            lr=lr,
            log_every=log_every,
            d_model=d_model,
            d_hidden=d_hidden,
            d_pair=d_pair,
            k=k,
            auxk=auxk,
            dead_steps_threshold=dead_steps_threshold,
        )
        
        all_results.append(result)
        
        if result['status'] == 'success':
            successful_layers.append(layer_idx)
        elif result['status'] == 'failed':
            failed_layers.append(layer_idx)
        elif result['status'] == 'skipped':
            skipped_layers.append(layer_idx)
    
    overall_end = time.time()
    overall_time = overall_end - overall_start
    
    # Print summary
    print()
    print()
    print("=" * 80)
    print("MULTI-LAYER TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total training time: {overall_time:.2f}s ({overall_time/60:.2f} min)")
    print()
    print(f"Successful: {len(successful_layers)} layers - {successful_layers}")
    print(f"Failed: {len(failed_layers)} layers - {failed_layers}")
    print(f"Skipped: {len(skipped_layers)} layers - {skipped_layers}")
    print()
    
    # Print detailed results for successful layers
    if successful_layers:
        print("Detailed Results (Successful Layers):")
        print("-" * 80)
        print(f"{'Layer':<8} {'Loss':>12} {'Recon':>12} {'Consist':>12} {'Dead':>8} {'Time(s)':>10}")
        print("-" * 80)
        
        for result in all_results:
            if result['status'] == 'success':
                layer_idx = result['layer_idx']
                loss = result['final_loss_total']
                recon = result['final_loss_reconstruction']
                consist = result['final_loss_consistency']
                dead = result['final_dead_neurons']
                train_time = result['training_time_seconds']
                
                print(f"{layer_idx:<8} {loss:>12.6f} {recon:>12.6f} {consist:>12.6f} {dead:>8} {train_time:>10.2f}")
        
        print("-" * 80)
    
    print()
    print("Checkpoint locations:")
    for result in all_results:
        if result['status'] == 'success':
            print(f"  Layer {result['layer_idx']:2d}: {result['checkpoint_dir']}")
    
    print()
    print("=" * 80)
    
    # Save summary
    summary_path = Path(checkpoint_base_dir) / 'multi_layer_training_summary.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'overall_time_seconds': overall_time,
        'num_layers_trained': len(layer_indices),
        'num_successful': len(successful_layers),
        'num_failed': len(failed_layers),
        'num_skipped': len(skipped_layers),
        'successful_layers': successful_layers,
        'failed_layers': failed_layers,
        'skipped_layers': skipped_layers,
        'hyperparameters': {
            'batch_size': batch_size,
            'num_steps': num_steps,
            'lr': lr,
            'd_model': d_model,
            'd_hidden': d_hidden,
            'd_pair': d_pair,
            'k': k,
            'auxk': auxk,
        },
        'results': all_results,
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Training summary saved to: {summary_path}")
    print()
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train PLTs for multiple pairformer layers")
    
    # Data and output
    parser.add_argument('--data_dir', type=str, default='multi_layer_activations',
                        help='Base directory containing multi-layer activations')
    parser.add_argument('--checkpoint_dir', type=str, default='multi_layer_checkpoints',
                        help='Base directory to save all checkpoints')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 8, 16, 24, 32, 40],
                        help='Layer indices to train (default: 0 8 16 24 32 40)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of training steps per layer')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log every N steps')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=384,
                        help='Input dimension (single representation)')
    parser.add_argument('--d_hidden', type=int, default=2048,
                        help='Latent dimension (number of features)')
    parser.add_argument('--d_pair', type=int, default=128,
                        help='Pair representation dimension')
    parser.add_argument('--k', type=int, default=16,
                        help='Top-K activation')
    parser.add_argument('--auxk', type=int, default=32,
                        help='Auxiliary K for dead neurons')
    parser.add_argument('--dead_steps_threshold', type=int, default=10000,
                        help='Steps before neuron considered dead')
    
    args = parser.parse_args()
    
    # Run multi-layer training
    train_multi_layer(
        data_base_dir=args.data_dir,
        checkpoint_base_dir=args.checkpoint_dir,
        layer_indices=args.layers,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        log_every=args.log_every,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        d_pair=args.d_pair,
        k=args.k,
        auxk=args.auxk,
        dead_steps_threshold=args.dead_steps_threshold,
    )


if __name__ == '__main__':
    main()
