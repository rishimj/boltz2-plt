"""
Collect activations from layer 48 by hooking into Boltz CLI inference.
"""
import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# Global storage for activations
activations = {}
layer_idx = None

def hook_transition_s(module, input, output):
    """Hook for transition_s MLP"""
    activations['input_s'] = input[0].detach().cpu()
    activations['output_s'] = output.detach().cpu()

def hook_transition_z(module, input, output):
    """Hook for transition_z MLP"""
    activations['input_z'] = input[0].detach().cpu()
    activations['output_z'] = output.detach().cpu()

def install_hooks(model, layer_idx):
    """Install hooks on the pairformer layer"""
    try:
        target_layer = model.pairformer_module.layers[layer_idx]
        target_layer.transition_s.register_forward_hook(hook_transition_s)
        target_layer.transition_z.register_forward_hook(hook_transition_z)
        print(f"✓ Installed hooks on layer {layer_idx}")
        return True
    except:
        # Try with 'net' wrapper
        try:
            target_layer = model.net.pairformer_module.layers[layer_idx]
            target_layer.transition_s.register_forward_hook(hook_transition_s)
            target_layer.transition_z.register_forward_hook(hook_transition_z)
            print(f"✓ Installed hooks on layer {layer_idx} (via net wrapper)")
            return True
        except Exception as e:
            print(f"✗ Failed to install hooks: {e}")
            return False

def monkey_patch_model_init():
    """Monkey patch the model initialization to install hooks after loading"""
    from boltz.model import Boltz2
    
    original_init = Boltz2.__init__
    
    def new_init(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        # Install hooks after model is initialized
        if layer_idx is not None:
            install_hooks(self, layer_idx)
        return result
    
    Boltz2.__init__ = new_init

def save_activations(output_path):
    """Save collected activations"""
    if len(activations) == 4:  # All 4 expected
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract numpy arrays
        data = {
            k: v.numpy() for k, v in activations.items()
        }
        
        np.savz(output_path, **data)
        print(f"\n✓ Saved activations to {output_path}")
        print(f"  input_s: {data['input_s'].shape}")
        print(f"  output_s: {data['output_s'].shape}")
        print(f"  input_z: {data['input_z'].shape}")
        print(f"  output_z: {data['output_z'].shape}")
        return True
    else:
        print(f"\n✗ Incomplete activations captured: {list(activations.keys())}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=47, help='Layer index (0-63)')
    parser.add_argument('--output', type=str, required=True, help='Output npz path')
    parser.add_argument('--fasta', type=str, required=True, help='Input fasta file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--out-dir', type=str, default='boltz_inference_temp', help='Temporary inference output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    global layer_idx
    layer_idx = args.layer
    
    print(f"Setting up activation collection for layer {layer_idx}...")
    
    # Monkey patch the model initialization
    monkey_patch_model_init()
    
    # Import and run Boltz CLI after patching
    from boltz.main import predict
    
    # Prepare arguments for Boltz CLI
    sys.argv = [
        'boltz',
        'predict',
        args.fasta,
        '--checkpoint', args.checkpoint,
        '--out_dir', args.out_dir,
        '--device', args.device,
        '--accelerator', args.device,
        '--num_workers', '0',
    ]
    
    print(f"\nRunning Boltz inference (output to {args.out_dir})...")
    print("This will capture activations during the forward pass.\n")
    
    try:
        # Run prediction - hooks will capture activations
        predict()
        
        # Save the captured activations
        save_activations(args.output)
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save partial activations
        if activations:
            print(f"\nPartial activations captured: {list(activations.keys())}")
            save_activations(args.output)

if __name__ == '__main__':
    main()
