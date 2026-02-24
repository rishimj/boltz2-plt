"""
Create synthetic activations for testing transcoder training.
"""
import numpy as np
from pathlib import Path

def create_synthetic_batch(
    output_dir: Path,
    batch_idx: int = 0,
    num_tokens: int = 117,  # From test protein
    token_s_dim: int = 384,
    token_z_dim: int = 128,
    num_samples: int = 10,
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
        
        # Pair representation: [num_tokens, num_tokens, token_z_dim]
        input_z = np.random.randn(num_tokens, num_tokens, token_z_dim).astype(np.float32)
        output_z = np.random.randn(num_tokens, num_tokens, token_z_dim).astype(np.float32)
        
        input_s_list.append(input_s)
        output_s_list.append(output_s)
        input_z_list.append(input_z)
        output_z_list.append(output_z)
    
    # Stack into arrays
    data = {
        'input_s': np.stack(input_s_list, axis=0),    # [num_samples, num_tokens, 384]
        'output_s': np.stack(output_s_list, axis=0),
        'input_z': np.stack(input_z_list, axis=0),    # [num_samples, num_tokens, num_tokens, 128]
        'output_z': np.stack(output_z_list, axis=0),
    }
    
    output_path = output_dir / f"batch_{batch_idx:05d}.npz"
    np.savez(output_path, **data)
    
    print(f"Created synthetic batch at {output_path}")
    print(f"  input_s shape: {data['input_s'].shape}")
    print(f"  output_s shape: {data['output_s'].shape}")
    print(f"  input_z shape: {data['input_z'].shape}")
    print(f"  output_z shape: {data['output_z'].shape}")
    
    return output_path

if __name__ == "__main__":
    output_dir = Path("pilot_activations_synthetic")
    
    # Create 5 batches for testing
    for i in range(5):
        create_synthetic_batch(output_dir, batch_idx=i, num_samples=20)
    
    print(f"\n✓ Created 5 synthetic activation batches in {output_dir}")
    print("You can now test transcoder training with: python train.py --activations pilot_activations_synthetic --output pilot_model --epochs 5")
