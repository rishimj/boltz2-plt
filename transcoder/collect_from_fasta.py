"""
Collect activations by running Boltz prediction and hooking layer 48.
Simplified approach using Boltz's CLI predict function directly.
"""
import sys
import argparse
from pathlib import Path
import torch
import numpy as np

# Global storage for activations
ACTIVATIONS = {}
LAYER_IDX = 47

def hook_transition_s(module, input, output):
    """Hook for transition_s MLP"""
    ACTIVATIONS['input_s'] = input[0].detach().cpu()
    ACTIVATIONS['output_s'] = output.detach().cpu()
    print(f"  Captured transition_s: input {input[0].shape} → output {output.shape}")

def hook_transition_z(module, input, output):
    """Hook for transition_z MLP"""
    ACTIVATIONS['input_z'] = input[0].detach().cpu()
    ACTIVATIONS['output_z'] = output.detach().cpu()
    print(f"  Captured transition_z: input {input[0].shape} → output {output.shape}")

def install_hooks_on_model(model):
    """Install hooks after model is created"""
    try:
        layer = model.pairformer_module.layers[LAYER_IDX]
        layer.transition_s.register_forward_hook(hook_transition_s)
        layer.transition_z.register_forward_hook(hook_transition_z)
        print(f"✓ Installed hooks on layer {LAYER_IDX}")
    except Exception as e:
        print(f"✗ Failed to install hooks: {e}")

def save_activations(output_path):
    """Save collected activations"""
    if len(ACTIVATIONS) == 4:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {k: v.numpy() for k, v in ACTIVATIONS.items()}
        np.savez(output_path, **data)
        
        print(f"\n✓ Saved activations to {output_path}")
        for k, v in data.items():
            print(f"  {k}: {v.shape}")
        return True
    else:
        print(f"\n✗ Only captured {len(ACTIVATIONS)}/4 activations: {list(ACTIVATIONS.keys())}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--layer', type=int, default=47)
    parser.add_argument('--out_dir', default='temp_boltz_inference')
    args = parser.parse_args()
    
    global LAYER_IDX
    LAYER_IDX = args.layer
    
    print(f"=== Collecting Activations from Layer {args.layer} ===")
    print(f"Input FASTA: {args.fasta}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}\n")
    
    # Monkey-patch Boltz2 to install hooks after creation
    from boltz.model.models.boltz2 import Boltz2
    original_init = Boltz2.__init__
    
    def patched_init(self, *init_args, **init_kwargs):
        result = original_init(self, *init_args, **init_kwargs)
        install_hooks_on_model(self)
        return result
    
    Boltz2.__init__ = patched_init
    
    # Import Boltz CLI
    from boltz import main as boltz_main
    
    # Set cache directory to avoid permission issues
    import os
    boltz_root = Path(__file__).parent.parent
    os.environ['BOLTZ_CACHE'] = str(boltz_root / ".boltz_cache")
    
    # Prepare arguments for Boltz CLI
    sys.argv = [
        'boltz',
        args.fasta,
        '--checkpoint', args.checkpoint,
        '--out_dir', args.out_dir,
        '--cache', str(boltz_root / ".boltz_cache"),
        '--num_workers', '0',
        '--override',
    ]
    
    print("Running Boltz inference...\n")
    
    try:
        boltz_main.predict()
        print("\n✓ Inference complete")
        save_activations(args.output)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        if ACTIVATIONS:
            print(f"\nPartial activations: {list(ACTIVATIONS.keys())}")
            save_activations(args.output)

if __name__ == '__main__':
    main()
