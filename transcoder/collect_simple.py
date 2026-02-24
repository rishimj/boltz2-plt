"""
Collect real activations from Boltz2 layer 48 by running Boltz CLI.
This hooks into Boltz's own inference pipeline to avoid featurization issues.
"""
import argparse
import sys
from pathlib import Path
import torch
import numpy as np

# Global storage
activations = {}

def register_hooks(model, layer_idx=47):
    """Register hooks on the pairformer layer"""
    try:
        layer = model.pairformer_module.layers[layer_idx]
        
        def hook_s(module, input, output):
            activations['input_s'] = input[0].detach().cpu()
            activations['output_s'] = output.detach().cpu()
        
        def hook_z(module, input, output):
            activations['input_z'] = input[0].detach().cpu()
            activations['output_z'] = output.detach().cpu()
        
        layer.transition_s.register_forward_hook(hook_s)
        layer.transition_z.register_forward_hook(hook_z)
        
        print(f"✓ Registered hooks on layer {layer_idx}")
        return True
    except Exception as e:
        print(f"✗ Failed to register hooks: {e}")
        return False

def save_activations(output_path):
    """Save collected activations to npz file"""
    if len(activations) == 4:
        data = {k: v.numpy() for k, v in activations.items()}
        np.savez(output_path, **data)
        print(f"\n✓ Saved activations to {output_path}")
        for k, v in data.items():
            print(f"  {k}: {v.shape}")
        return True
    else:
        print(f"\n✗ Only captured {len(activations)}/4 activations")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True, help='Input fasta file')
    parser.add_argument('--output', required=True, help='Output npz path')
    parser.add_argument('--checkpoint', default='../boltz2_checkpoint.ckpt')
    parser.add_argument('--layer', type=int, default=47)
    args = parser.parse_args()
    
    # Monkey-patch Boltz2 to add hooks after initialization
    from boltz.model.models.boltz2 import Boltz2
    original_init = Boltz2.__init__
    
    def patched_init(self, *init_args, **init_kwargs):
        result = original_init(self, *init_args, **init_kwargs)
        register_hooks(self, args.layer)
        return result
    
    Boltz2.__init__ = patched_init
    
    # Now import and run Boltz CLI
    from boltz.main import predict
    
    # Set up CLI arguments
    sys.argv = [
        'boltz', 'predict', args.fasta,
        '--checkpoint', args.checkpoint,
        '--out_dir', 'temp_inference',
        '--num_workers', '0',
        '--override',
    ]
    
    print("Running Boltz inference to collect activations...")
    print(f"Input: {args.fasta}")
    print(f"Layer: {args.layer}")
    print(f"Output: {args.output}\n")
    
    try:
        predict()
        save_activations(args.output)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        if activations:
            print(f"\nPartial activations: {list(activations.keys())}")

if __name__ == '__main__':
    main()
