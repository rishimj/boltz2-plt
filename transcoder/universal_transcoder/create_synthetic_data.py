"""
Create synthetic activations for testing Universal Transcoder training.

Generates synthetic data matching the expected format:
- input_s: Single representation input [batch, num_tokens, 384]
- output_s: Single representation output [batch, num_tokens, 384]
- input_z: Pair representation input [batch, num_tokens², 128] (flattened)
- output_z: Pair representation output [batch, num_tokens², 128] (flattened)
"""
import numpy as np
from pathlib import Path
import argparse


def create_synthetic_batch(
    output_dir: Path,
    batch_idx: int = 0,
    num_tokens: int = 117,  # From test protein
    token_s_dim: int = 384, # Single representation dimension
    token_z_dim: int = 128, # Pair representation dimension
    num_samples: int = 10,  # Samples per batch
):
    """
    Create a synthetic activation batch file.
    
    Args:
        output_dir: Directory to save to
        batch_idx: Batch index
        num_tokens: Number of tokens (residues)
        token_s_dim: Dimension of single representation
        token_z_dim: Dimension of pair representation
        num_samples: Number of samples in batch
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic activations with realistic structure
    # Use normal distribution with mean=0, std=1 (typical for normalized activations)
    input_s_list = []
    output_s_list = []
    input_z_list = []
    output_z_list = []
    
    for i in range(num_samples):
        # Single representation: [num_tokens, token_s_dim]
        input_s = np.random.randn(num_tokens, token_s_dim).astype(np.float32)
        output_s = np.random.randn(num_tokens, token_s_dim).astype(np.float32)
        
        # Pair representation: [num_tokens², token_z_dim] (already flattened)
        # This matches the expected format from the existing codebase
        num_pairs = num_tokens * num_tokens
        input_z = np.random.randn(num_pairs, token_z_dim).astype(np.float32)
        output_z = np.random.randn(num_pairs, token_z_dim).astype(np.float32)
        
        input_s_list.append(input_s)
        output_s_list.append(output_s)
        input_z_list.append(input_z)
        output_z_list.append(output_z)
    
    # Stack into arrays
    data = {
        'input_s': np.stack(input_s_list, axis=0),    # [num_samples, num_tokens, 384]
        'output_s': np.stack(output_s_list, axis=0),  # [num_samples, num_tokens, 384]
        'input_z': np.stack(input_z_list, axis=0),    # [num_samples, num_tokens², 128]
        'output_z': np.stack(output_z_list, axis=0),  # [num_samples, num_tokens², 128]
    }
    
    output_path = output_dir / f"batch_{batch_idx:05d}.npz"
    np.savez(output_path, **data)
    
    print(f"Created synthetic batch at {output_path}")
    print(f"  input_s shape: {data['input_s'].shape}")
    print(f"  output_s shape: {data['output_s'].shape}")
    print(f"  input_z shape: {data['input_z'].shape}")
    print(f"  output_z shape: {data['output_z'].shape}")
    
    return output_path


def main():
    """Generate multiple batches of synthetic data."""
    parser = argparse.ArgumentParser(description="Generate synthetic activation data")
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for synthetic data')
    parser.add_argument('--num_batches', type=int, default=10,
                        help='Number of batches to generate')
    parser.add_argument('--samples_per_batch', type=int, default=10,
                        help='Number of samples per batch')
    parser.add_argument('--num_tokens', type=int, default=117,
                        help='Number of tokens (residues) per sample')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print(f"Generating {args.num_batches} batches of synthetic data...")
    print(f"Samples per batch: {args.samples_per_batch}")
    print(f"Tokens per sample: {args.num_tokens}")
    print(f"Output directory: {output_dir}")
    print()
    
    for batch_idx in range(args.num_batches):
        create_synthetic_batch(
            output_dir=output_dir,
            batch_idx=batch_idx,
            num_tokens=args.num_tokens,
            num_samples=args.samples_per_batch,
        )
    
    print()
    print(f"✓ Successfully generated {args.num_batches} batches")
    print(f"✓ Total samples: {args.num_batches * args.samples_per_batch}")
    print(f"✓ Saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
