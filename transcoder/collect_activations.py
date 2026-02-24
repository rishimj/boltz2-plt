"""Collect activations from Pairformer Layer 48 MLPs."""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import torch

# Add boltz to path
boltz_root = Path(__file__).parent.parent
sys.path.insert(0, str(boltz_root / "src"))

from boltz.model.models.boltz2 import Boltz2
from boltz.data.types import StructureV2, MSA, Record, Manifest, ChainInfo
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer


class ActivationCollector:
    """Collects activations from pairformer layer 48 MLP."""
    
    def __init__(self, model, layer_idx=47, device='cuda'):
        """
        Initialize activation collector.
        
        Args:
            model: Boltz model
            layer_idx: Index of pairformer layer (47 for layer 48)
            device: Device to run on
        """
        self.model = model
        self.layer_idx = layer_idx
        self.device = device
        
        # Storage for activations
        self.activations = {
            'input_s': [],
            'output_s': [],
            'input_z': [],
            'output_z': [],
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
            # Input to transition is after LayerNorm, before MLP
            # The Transition module applies LayerNorm first
            x = input[0].detach().cpu()
            self.activations['input_s'].append(x)
            print(f"[DEBUG] Captured s input: {x.shape}")
        
        def hook_s_output(module, input, output):
            x = output.detach().cpu()
            self.activations['output_s'].append(x)
            print(f"[DEBUG] Captured s output: {x.shape}")
        
        # Hook for transition_z (pair representation MLP)
        def hook_z_input(module, input, output):
            x = input[0].detach().cpu()
            self.activations['input_z'].append(x)
            print(f"[DEBUG] Captured z input: {x.shape}")
        
        def hook_z_output(module, input, output):
            x = output.detach().cpu()
            self.activations['output_z'].append(x)
            print(f"[DEBUG] Captured z output: {x.shape}")
        
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
            self.activations[key] = []
    
    def save_batch(self, output_dir, batch_idx):
        """
        Save collected activations to npz file.
        
        Args:
            output_dir: Directory to save activations
            batch_idx: Batch index for filename
        """
        output_path = Path(output_dir) / f"batch_{batch_idx:05d}.npz"
        
        # Convert lists to arrays and flatten appropriately
        data = {}
        
        if self.activations['input_s']:
            # Single: [B, N, 384] -> keep as is
            input_s = torch.cat(self.activations['input_s'], dim=0).numpy()
            output_s = torch.cat(self.activations['output_s'], dim=0).numpy()
            data['input_s'] = input_s
            data['output_s'] = output_s
            print(f"  s shapes: input={input_s.shape}, output={output_s.shape}")
        
        if self.activations['input_z']:
            # Pair: [B, N, N, 128] -> flatten to [B, N*N, 128]
            input_z_list = []
            output_z_list = []
            for inp, out in zip(self.activations['input_z'], self.activations['output_z']):
                B, N1, N2, D = inp.shape
                input_z_list.append(inp.reshape(B, N1 * N2, D))
                output_z_list.append(out.reshape(B, N1 * N2, D))
            
            input_z = torch.cat(input_z_list, dim=0).numpy()
            output_z = torch.cat(output_z_list, dim=0).numpy()
            data['input_z'] = input_z
            data['output_z'] = output_z
            print(f"  z shapes: input={input_z.shape}, output={output_z.shape}")
        
        np.savez_compressed(output_path, **data)
        print(f"✓ Saved batch {batch_idx} to {output_path}")
        
        # Clear activations after saving
        self.clear_activations()


def collect_activations(
    checkpoint_path,
    structures_dir,
    msa_dir,
    output_dir,
    max_structures=100,
    layer_idx=47,
    device='cuda',
    recycling_steps=0
):
    """
    Collect activations from Boltz model.
    
    Args:
        checkpoint_path: Path to Boltz2 model checkpoint
        structures_dir: Path to directory with .npz structure files
        msa_dir: Path to directory with MSA files
        output_dir: Directory to save activations
        max_structures: Maximum number of structures to process
        layer_idx: Pairformer layer index (47 for layer 48)
        device: Device to run on
        recycling_steps: Number of recycling steps (0 = final iteration only)
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract hyperparameters and filter to valid __init__ args
    all_hparams = checkpoint['hyper_parameters']
    
    # Get valid parameter names from Boltz2.__init__
    import inspect
    valid_params = set(inspect.signature(Boltz2.__init__).parameters.keys()) - {'self'}
    
    # Filter hyperparameters
    hparams = {k: v for k, v in all_hparams.items() if k in valid_params}
    
    print(f"Creating model with {len(hparams)} valid hyperparameters...")
    model = Boltz2(**hparams)
    
    # Load weights directly
    print("Loading weights from checkpoint...")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model = model.to(device)
    model.eval()
    model.eval()
    
    print(f"Initializing activation collector for layer {layer_idx}...")
    collector = ActivationCollector(model, layer_idx=layer_idx, device=device)
    
    print("Initializing tokenizer and featurizer...")
    tokenizer = Boltz2Tokenizer()
    featurizer = Boltz2Featurizer()
    
    # Load canonical molecules - try local cache first
    from boltz.data.mol import load_canonicals
    moldir = Path("/usr/scratch/rmanimaran8/boltz/.boltz_cache/mols")
    if not moldir.exists():
        moldir = Path.home() / ".boltz_cache" / "mols"
    if not moldir.exists():
        raise ValueError(f"Molecules directory not found. Expected at {moldir}")
    molecules = load_canonicals(str(moldir))
    print(f"Loaded {len(molecules)} canonical molecules from {moldir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of structure files
    structures_path = Path(structures_dir)
    structure_files = sorted(structures_path.glob("*.npz"))
    
    if not structure_files:
        raise ValueError(f"No .npz files found in {structures_dir}")
    
    print(f"Found {len(structure_files)} structure files")
    print(f"Will process up to {max_structures} structures")
    print(f"Saving activations to: {output_dir}")
    
    # Process structures
    batch_count = 0
    structure_count = 0
    
    with torch.no_grad():
        for structure_file in tqdm(structure_files[:max_structures], desc="Processing structures"):
            if structure_count >= max_structures:
                break
            
            try:
                print(f"\nProcessing {structure_file.name}")
                
                # Load structure
                structure = StructureV2.load(structure_file)
                
                # Load MSAs if available
                msa = {}
                msa_path = Path(msa_dir)
                if msa_path.exists():
                    for msa_file in msa_path.glob(f"{structure_file.stem}_*.npz"):
                        chain_id = msa_file.stem.split('_')[-1]  # Get chain ID from filename
                        msa_data = MSA.load(msa_file)
                        msa[chain_id] = msa_data
                
                # Create minimal Record
                from boltz.data.types import Record, StructureInfo, ChainInfo, InterfaceInfo
                
                # Build minimal chain info - just enough to satisfy tokenizer
                chains = []
                for i, chain_rec in enumerate(structure.chains):
                    chains.append(ChainInfo(
                        chain_id=i,
                        chain_name=chain_rec["name"].decode() if isinstance(chain_rec["name"], bytes) else str(chain_rec["name"]),
                        mol_type=int(chain_rec["mol_type"]),
                        cluster_id=i,
                        msa_id=i if str(i) in msa else -1,
                        num_residues=int(chain_rec["res_num"]),
                        valid=True,
                    ))
                
                record = Record(
                    id=structure_file.stem,
                    structure=StructureInfo(),
                    chains=chains,
                    interfaces=[],
                    affinity=None,
                )
                
                # Create input
                from boltz.data.types import Input
                input_data = Input(structure=structure, msa=msa, record=record)
                
                # Tokenize
                print("  Tokenizing...")
                tokenized = tokenizer.tokenize(input_data)
                
                # Featurize
                print("  Featurizing...")
                random_generator = np.random.default_rng(42)
                features = featurizer.process(
                    tokenized,
                    molecules=molecules,
                    random=random_generator,
                    training=False,
                    max_atoms=None,
                    max_tokens=None,
                    max_seqs=128,  # Reasonable default
                    pad_to_max_seqs=False,
                    compute_frames=True,
                )
                
                # Move to device and add batch dimension
                feats = {}
                for key, value in features.items():
                    if isinstance(value, torch.Tensor):
                        feats[key] = value.unsqueeze(0).to(device)
                    elif isinstance(value, np.ndarray):
                        feats[key] = torch.from_numpy(value).unsqueeze(0).to(device)
                    else:
                        feats[key] = value
                
                print(f"  Running inference (recycling_steps={recycling_steps})...")
                # Run model forward pass (activations collected via hooks)
                try:
                    output = model(feats, recycling_steps=recycling_steps)
                    print("  ✓ Inference complete")
                except Exception as e:
                    print(f"  Warning: Model forward pass failed: {e}")
                    print("  Continuing to check if activations were collected...")
                
                structure_count += 1
                
                # Save collected activations if we have any
                if len(collector.activations['input_s']) > 0 or len(collector.activations['input_z']) > 0:
                    print(f"  Saving activations (batch {batch_count})...")
                    collector.save_batch(output_dir, batch_count)
                    batch_count += 1
                else:
                    print("  Warning: No activations were collected!")
                
            except Exception as e:
                print(f"Error processing {structure_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    collector.remove_hooks()
    print(f"\n✓ Collection complete! Processed {structure_count} structures")
    print(f"✓ Saved {batch_count} activation batches")


def main():
    parser = argparse.ArgumentParser(description='Collect activations from Pairformer Layer 48')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Boltz2 checkpoint')
    parser.add_argument('--structures', type=str, required=True, help='Path to structures directory')
    parser.add_argument('--msa', type=str, default=None, help='Path to MSA directory (optional)')
    parser.add_argument('--output', type=str, default='pilot_activations', help='Output directory')
    parser.add_argument('--max-structures', type=int, default=100, help='Max structures to process')
    parser.add_argument('--layer', type=int, default=47, help='Layer index (47 for layer 48)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--recycling-steps', type=int, default=0, help='Number of recycling steps')
    
    args = parser.parse_args()
    
    collect_activations(
        checkpoint_path=args.checkpoint,
        structures_dir=args.structures,
        msa_dir=args.msa or args.structures,  # Use structures dir if MSA not provided
        output_dir=args.output,
        max_structures=args.max_structures,
        layer_idx=args.layer,
        device=args.device,
        recycling_steps=args.recycling_steps
    )


if __name__ == "__main__":
    main()
