"""Collect activations from example proteins - FIXED VERSION."""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import torch
from datetime import datetime

# Add boltz to path
boltz_root = Path(__file__).parent.parent
sys.path.insert(0, str(boltz_root / "src"))

from boltz.model.models.boltz2 import Boltz2


class ActivationCollector:
    """Collects activations from pairformer layer 48 MLP."""
    
    def __init__(self, model, layer_idx=47, device='cuda'):
        self.model = model
        self.layer_idx = layer_idx
        self.device = device
        
        # Storage for activations
        self.activations = {
            'input_s': None,
            'output_s': None,
            'input_z': None,
            'output_z': None,
        }
        
        # Get the target layer
        self.target_layer = model.pairformer_module.layers[layer_idx]
        
        # Register hooks
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on transition_s and transition_z."""
        
        # Hook for transition_s (single representation MLP)
        def hook_s_input(module, input, output):
            x = input[0].detach().cpu()
            self.activations['input_s'] = x
        
        def hook_s_output(module, input, output):
            x = output.detach().cpu()
            self.activations['output_s'] = x
        
        # Hook for transition_z (pair representation MLP)
        def hook_z_input(module, input, output):
            x = input[0].detach().cpu()
            self.activations['input_z'] = x
        
        def hook_z_output(module, input, output):
            x = output.detach().cpu()
            self.activations['output_z'] = x
        
        # Register hooks
        if hasattr(self.target_layer, 'transition_s'):
            h1 = self.target_layer.transition_s.register_forward_hook(hook_s_input)
            h2 = self.target_layer.transition_s.register_forward_hook(hook_s_output)
            self.hooks.extend([h1, h2])
            print(f"✓ Registered hooks on transition_s")
        
        if hasattr(self.target_layer, 'transition_z'):
            h3 = self.target_layer.transition_z.register_forward_hook(hook_z_input)
            h4 = self.target_layer.transition_z.register_forward_hook(hook_z_output)
            self.hooks.extend([h3, h4])
            print(f"✓ Registered hooks on transition_z")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_activations(self):
        """Clear stored activations."""
        for key in self.activations:
            self.activations[key] = None
    
    def save_activations(self, output_path, protein_name):
        """Save collected activations to npz file."""
        data = {
            'protein_name': protein_name,
            'timestamp': str(datetime.now()),
        }
        
        if self.activations['input_s'] is not None:
            input_s = self.activations['input_s'].numpy()
            output_s = self.activations['output_s'].numpy()
            data['input_s'] = input_s  # Shape: (1, seq_len, 384)
            data['output_s'] = output_s
            print(f"  s shapes: {input_s.shape}")
        
        if self.activations['input_z'] is not None:
            input_z = self.activations['input_z'].numpy()
            output_z = self.activations['output_z'].numpy()
            # Flatten pairs: (1, N, N, 128) -> (1, N*N, 128)
            B, N1, N2, D = input_z.shape
            data['input_z'] = input_z.reshape(B, N1 * N2, D)
            data['output_z'] = output_z.reshape(B, N1 * N2, D)
            print(f"  z shapes: {data['input_z'].shape}")
        
        np.savez_compressed(output_path, **data)
        print(f"✓ Saved to {output_path}")
        
        self.clear_activations()


def collect_from_preprocessed(
    checkpoint_path,
    predictions_dir,
    output_dir,
    layer_idx=47,
    device='cuda'
):
    """
    Collect activations from already-preprocessed Boltz predictions.
    
    This loads the featurized data directly and runs only the trunk forward pass.
    """
    print(f"[{datetime.now()}] Loading checkpoint from {checkpoint_path}...")
    
    # Try using load_from_checkpoint to avoid manual initialization
    try:
        model = Boltz2.load_from_checkpoint(
            checkpoint_path,
            map_location='cpu',
            strict=False
        )
        print("✓ Loaded model using load_from_checkpoint")
    except Exception as e:
        print(f"load_from_checkpoint failed: {e}")
        print("Trying manual loading...")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        import inspect
        valid_params = set(inspect.signature(Boltz2.__init__).parameters.keys()) - {'self'}
        hparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in valid_params}
        
        print(f"Creating model (this may take 1-2 minutes)...")
        model = Boltz2(**hparams)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("✓ Loaded model manually")
    
    model = model.to(device)
    model.eval()
    
    print(f"Initializing activation collector for layer {layer_idx}...")
    collector = ActivationCollector(model, layer_idx=layer_idx, device=device)
    
    # Find all prediction directories
    predictions_dir = Path(predictions_dir)
    pred_dirs = sorted([d for d in predictions_dir.iterdir() if d.is_dir()])
    
    print(f"\nFound {len(pred_dirs)} prediction directories")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each prediction
    for idx, pred_dir in enumerate(tqdm(pred_dirs, desc="Collecting activations")):
        try:
            # Load the featurized batch from predictions directory
            # Boltz saves intermediate features in lightning_logs/
            processed_dir = pred_dir / "processed"
            
            if not processed_dir.exists():
                print(f"  Skipping {pred_dir.name} - no processed dir")
                continue
            
            # Load structure and features
            # This is the already-featurized data Boltz uses
            struct_files = list((processed_dir / "structures").glob("*.npz"))
            
            if not struct_files:
                print(f"  Skipping {pred_dir.name} - no structure files")
                continue
            
            struct_file = struct_files[0]
            print(f"\n  Processing {pred_dir.name} from {struct_file.name}...")
            
            # Load the processed structure
            from boltz.data.types import StructureV2
            structure = StructureV2.load(str(struct_file))
            
            # Load MSA if available
            msa_files = list((processed_dir / "msa").glob("*.npz"))
            msa = None
            if msa_files:
                from boltz.data.types import MSA
                msa = MSA.load(str(msa_files[0]))
            
            # Use Boltz's data loading pipeline
            # Load the saved features if they exist
            features_file = processed_dir / "features.npz"
            if features_file.exists():
                print(f"  Loading pre-computed features from {features_file}")
                loaded_feats = np.load(features_file)
                feats = {k: torch.from_numpy(v) for k, v in loaded_feats.items()}
            else:
                # Need to recompute features - use the full data pipeline
                from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
                from boltz.data.feature.featurizerv2 import Boltz2Featurizer
                from boltz.data.types import Input, Manifest
                
                # Load manifest
                manifest_file = processed_dir / "manifest.json"
                if not manifest_file.exists():
                    print(f"  Skipping {pred_dir.name} - no manifest")
                    continue
                manifest = Manifest.load(manifest_file)
                
                if not manifest.records or len(manifest.records) == 0:
                    print(f"  Skipping {pred_dir.name} - empty manifest")
                    continue
                
                record = manifest.records[0]
                
                # Create Input
                input_data = Input(
                    structure=structure,
                    msa={'0': msa} if msa else {},
                    record=record,
                )
                
                # Tokenize
                tokenizer = Boltz2Tokenizer()
                tokens = tokenizer.tokenize(input_data)
                
                # Load molecules (amino acids, nucleotides, etc.)
                from boltz.data.mol import load_canonicals
                mol_dir = boltz_root / ".boltz_cache" / "mols"
                molecules = load_canonicals(str(mol_dir))
                print(f"  Loaded {len(molecules)} canonical molecules")
                
                # Featurize
                featurizer = Boltz2Featurizer()
                import numpy as np
                feats = featurizer.process(
                    data=tokens,
                    random=np.random.default_rng(42),
                    molecules=molecules,
                    training=False,
                    max_seqs=128,
                )
                feats = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                        for k, v in feats.items()}
            
            # Convert to tensor batch
            feats_batch = {}
            for k, v in feats.items():
                if isinstance(v, np.ndarray):
                    feats_batch[k] = torch.from_numpy(v).unsqueeze(0).to(device)
                elif isinstance(v, torch.Tensor):
                    feats_batch[k] = v.unsqueeze(0).to(device)
                else:
                    feats_batch[k] = v
            
            # Run forward pass (trunk only, no structure prediction)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    output = model(
                        feats=feats_batch,
                        recycling_steps=0,
                        run_trunk_and_structure=True,
                        skip_run_structure=True  # Skip diffusion, only run trunk
                    )
            
            # Save activations
            output_path = Path(output_dir) / f"protein_{idx:03d}_{pred_dir.name}.npz"
            collector.save_activations(output_path, pred_dir.name)
            
        except Exception as e:
            print(f"  ERROR processing {pred_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    collector.remove_hooks()
    print(f"\n[{datetime.now()}] Collection complete!")
    print(f"Saved {idx+1} activation files to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../boltz2_checkpoint.ckpt")
    parser.add_argument("--predictions_dir", default="example_predictions")
    parser.add_argument("--output_dir", default="real_activations")
    parser.add_argument("--layer_idx", type=int, default=47)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    collect_from_preprocessed(
        checkpoint_path=args.checkpoint,
        predictions_dir=args.predictions_dir,
        output_dir=args.output_dir,
        layer_idx=args.layer_idx,
        device=args.device
    )
