"""
Convert individual protein files to batched format for training.
"""
import numpy as np
from pathlib import Path

def main():
    data_dir = Path("real_activations")
    
    # Load all protein files
    protein_files = sorted(data_dir.glob("protein_*.npz"))
    print(f"Found {len(protein_files)} protein files")
    
    # Load each and check shapes
    proteins = []
    for pfile in protein_files:
        data = np.load(pfile)
        print(f"\n{pfile.name}:")
        for key in ['input_s', 'output_s', 'input_z', 'output_z']:
            print(f"  {key}: {data[key].shape}")
        proteins.append(data)
    
    # Since proteins have different sizes, save each as separate batch
    # (batch dimension is already there from collection)
    for i, (pfile, data) in enumerate(zip(protein_files, proteins)):
        output_file = data_dir / f"batch_{i:05d}.npz"
        
        # Data already has batch dimension from collection
        np.savez_compressed(
            output_file,
            input_s=data['input_s'],
            output_s=data['output_s'],
            input_z=data['input_z'],
            output_z=data['output_z'],
        )
        print(f"\n✓ Saved: {output_file.name}")
    
    print(f"\nTotal batches created: {len(proteins)}")

if __name__ == '__main__':
    main()
