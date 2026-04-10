"""Collect activations from multiple Pairformer layers simultaneously."""

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


class LayerActivationCollector:
    """Collects activations from a single pairformer layer."""
    
    def __init__(self, model, layer_idx, device='cuda'):
        """
        Initialize activation collector for a single layer.
        
        Args:
            model: Boltz model
            layer_idx: Index of pairformer layer
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
            x = input[0].detach().cpu()
            self.activations['input_s'].append(x)
        
        def hook_s_output(module, input, output):
            x = output.detach().cpu()
            self.activations['output_s'].append(x)
        
        # Hook for transition_z (pair representation MLP)
        def hook_z_input(module, input, output):
            x = input[0].detach().cpu()
            self.activations['input_z'].append(x)
        
        def hook_z_output(module, input, output):
            x = output.detach().cpu()
            self.activations['output_z'].append(x)
        
        # Register hooks
        if hasattr(self.target_layer, 'transition_s'):
            h1 = self.target_layer.transition_s.register_forward_hook(hook_s_input)
            h2 = self.target_layer.transition_s.register_forward_hook(hook_s_output)
            self.hooks.extend([h1, h2])
        
        if hasattr(self.target_layer, 'transition_z'):
            h3 = self.target_layer.transition_z.register_forward_hook(hook_z_input)
            h4 = self.target_layer.transition_z.register_forward_hook(hook_z_output)
            self.hooks.extend([h3, h4])
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_activations(self):
        """Clear stored activations."""
        for key in self.activations:
            self.activations[key] = []

    def get_batch(self):
        """Return collected activations as a batch and clear the buffer."""
        data = {}

        if self.activations['input_s']:
            # Single: [B, N, 384] -> keep as is
            input_s = torch.cat(self.activations['input_s'], dim=0)
            output_s = torch.cat(self.activations['output_s'], dim=0)
            data['input_s'] = input_s
            data['output_s'] = output_s

        if self.activations['input_z']:
            # Pair: [B, N, N, 128] -> flatten to [B, N*N, 128]
            input_z_list = []
            output_z_list = []
            for inp, out in zip(self.activations['input_z'], self.activations['output_z']):
                B, N1, N2, D = inp.shape
                input_z_list.append(inp.reshape(B, N1 * N2, D))
                output_z_list.append(out.reshape(B, N1 * N2, D))

            data['input_z'] = torch.cat(input_z_list, dim=0)
            data['output_z'] = torch.cat(output_z_list, dim=0)

        self.clear_activations()
        return data if data else None
    
    def save_batch(self, output_dir, batch_idx):
        """
        Save collected activations to npz file.
        
        Args:
            output_dir: Directory to save activations
            batch_idx: Batch index for filename
        """
        output_path = Path(output_dir) / f"batch_{batch_idx:05d}.npz"

        batch = self.get_batch()

        # Only save if we have data
        if batch:
            os.makedirs(output_dir, exist_ok=True)
            np.savez_compressed(
                output_path,
                **{key: value.numpy() for key, value in batch.items()}
            )

        return batch is not None


class MultiLayerActivationCollector:
    """Collects activations from multiple pairformer layers simultaneously."""
    
    def __init__(self, model, layer_indices, device='cuda'):
        """
        Initialize multi-layer activation collector.
        
        Args:
            model: Boltz model
            layer_indices: List of pairformer layer indices to collect from
            device: Device to run on
        """
        self.model = model
        self.layer_indices = layer_indices
        self.device = device
        
        # Create a collector for each layer
        self.collectors = {}
        for idx in layer_indices:
            self.collectors[idx] = LayerActivationCollector(model, idx, device)
        
        print(f"✓ Initialized collectors for layers: {layer_indices}")
    
    def remove_hooks(self):
        """Remove all hooks from all collectors."""
        for collector in self.collectors.values():
            collector.remove_hooks()
    
    def clear_activations(self):
        """Clear activations from all collectors."""
        for collector in self.collectors.values():
            collector.clear_activations()

    def pop_batches(self):
        """Return collected batches for all layers and clear their buffers."""
        batches = {}
        for layer_idx, collector in self.collectors.items():
            batches[layer_idx] = collector.get_batch()
        return batches
    
    def save_batch(self, output_base_dir, batch_idx):
        """
        Save collected activations from all layers.
        
        Args:
            output_base_dir: Base directory to save activations
            batch_idx: Batch index for filename
            
        Returns:
            dict mapping layer_idx -> bool (True if saved successfully)
        """
        results = {}
        for layer_idx, collector in self.collectors.items():
            layer_dir = Path(output_base_dir) / f"layer_{layer_idx:02d}"
            success = collector.save_batch(layer_dir, batch_idx)
            results[layer_idx] = success
        
        return results


def collect_activations_multi_layer(
    checkpoint_path,
    fasta_path,
    output_dir,
    layer_indices,
    max_proteins=10,
    device='cuda',
    recycling_steps=0
):
    """
    Collect activations from multiple pairformer layers.
    
    Args:
        checkpoint_path: Path to Boltz2 model checkpoint
        fasta_path: Path to FASTA file or directory with FASTA files
        output_dir: Base directory to save activations
        layer_indices: List of layer indices to collect from
        max_proteins: Maximum number of proteins to process
        device: Device to run on
        recycling_steps: Number of recycling steps (0 = final iteration only)
    """
    print("=" * 70)
    print("MULTI-LAYER ACTIVATION COLLECTION")
    print("=" * 70)
    print(f"Layers to collect: {layer_indices}")
    print(f"Output directory: {output_dir}")
    print(f"Max proteins: {max_proteins}")
    print(f"Device: {device}")
    print()
    
    # Load model
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
    
    # Load weights
    print("Loading weights from checkpoint...")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (pairformer has {len(model.pairformer_module.layers)} layers)")
    print()
    
    # Initialize multi-layer collector
    print(f"Initializing multi-layer activation collector...")
    collector = MultiLayerActivationCollector(model, layer_indices, device=device)
    print()
    
    # Initialize tokenizer and featurizer
    print("Initializing tokenizer and featurizer...")
    tokenizer = Boltz2Tokenizer()
    featurizer = Boltz2Featurizer()
    
    # Load canonical molecules
    from boltz.data.mol import load_canonicals
    moldir = Path("/usr/scratch/rmanimaran8/boltz/.boltz_cache/mols")
    if not moldir.exists():
        moldir = Path.home() / ".boltz_cache" / "mols"
    if not moldir.exists():
        raise ValueError(f"Molecules directory not found. Expected at {moldir}")
    molecules = load_canonicals(str(moldir))
    print(f"✓ Loaded {len(molecules)} canonical molecules")
    print()
    
    # Get FASTA files
    fasta_path = Path(fasta_path).resolve()  # Resolve relative paths
    if fasta_path.is_file():
        fasta_files = [fasta_path]
    elif fasta_path.is_dir():
        fasta_files = sorted(fasta_path.glob("*.fasta")) + sorted(fasta_path.glob("*.fa"))
    else:
        raise ValueError(f"Invalid FASTA path: {fasta_path}")
    
    if not fasta_files:
        raise ValueError(f"No FASTA files found at {fasta_path}")
    
    print(f"Found {len(fasta_files)} FASTA file(s)")
    print(f"Will process up to {max_proteins} proteins")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process proteins
    batch_count = 0
    protein_count = 0
    
    with torch.no_grad():
        for fasta_file in tqdm(fasta_files[:max_proteins], desc="Processing proteins"):
            if protein_count >= max_proteins:
                break
            
            try:
                print(f"\nProcessing {fasta_file.name}")
                
                # Parse FASTA
                from boltz.data.parse.fasta import parse_fasta
                target = parse_fasta(fasta_file, molecules, moldir, boltz2=True)
                
                # Load MSAs if specified
                msa_dict = {}
                for chain in target.record.chains:
                    if chain.msa_id and chain.msa_id != -1:
                        msa_path = Path(chain.msa_id)
                        if not msa_path.is_absolute():
                            msa_path = (fasta_file.parent / msa_path).resolve()
                        
                        if msa_path.exists():
                            from boltz.data.parse.a3m import parse_a3m
                            msa = parse_a3m(msa_path, taxonomy=None)
                            msa_dict[chain.chain_name] = msa
                
                # Create Input object
                from boltz.data.types import Input
                input_data = Input(
                    structure=target.structure,
                    msa=msa_dict,
                    record=target.record,
                    residue_constraints=target.residue_constraints,
                    templates=target.templates,
                    extra_mols=target.extra_mols,
                )
                
                # Tokenize
                print("  Tokenizing...")
                tokens = tokenizer.tokenize(input_data)
                
                # Featurize
                print("  Featurizing...")
                random_generator = np.random.default_rng(42)
                features = featurizer.process(
                    data=tokens,
                    molecules=molecules,
                    random=random_generator,
                    training=False,
                    max_seqs=128,
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
                    print(f"  Note: Model forward pass encountered: {e}")
                    print("  Continuing to check if activations were collected...")
                
                protein_count += 1
                
                # Save collected activations
                print(f"  Saving activations (batch {batch_count})...")
                results = collector.save_batch(output_dir, batch_count)
                
                # Report which layers saved successfully
                saved_layers = [idx for idx, success in results.items() if success]
                if saved_layers:
                    print(f"  ✓ Saved activations for layers: {saved_layers}")
                    batch_count += 1
                else:
                    print("  ⚠ No activations collected for any layer")
                
            except Exception as e:
                print(f"Error processing {fasta_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Cleanup
    collector.remove_hooks()
    
    # Print summary
    print()
    print("=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Proteins processed: {protein_count}")
    print(f"Batches saved: {batch_count}")
    print(f"Output directory: {output_dir}")
    print()
    
    # List what was saved for each layer
    for layer_idx in layer_indices:
        layer_dir = Path(output_dir) / f"layer_{layer_idx:02d}"
        if layer_dir.exists():
            npz_files = list(layer_dir.glob("*.npz"))
            print(f"  Layer {layer_idx:2d}: {len(npz_files):3d} batches in {layer_dir}")
        else:
            print(f"  Layer {layer_idx:2d}: No data saved")
    
    print()
    print("✓ Multi-layer collection complete!")


def main():
    parser = argparse.ArgumentParser(description='Collect activations from multiple Pairformer layers')
    parser.add_argument('--checkpoint', type=str, 
                        default='/usr/scratch/rmanimaran8/boltz/.boltz_cache/boltz2_conf.ckpt',
                        help='Path to Boltz2 checkpoint')
    parser.add_argument('--fasta', type=str, required=True, 
                        help='Path to FASTA file or directory with FASTA files')
    parser.add_argument('--output', type=str, default='multi_layer_activations', 
                        help='Output base directory')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 8, 16, 24, 32, 40],
                        help='Layer indices to collect from (default: 0 8 16 24 32 40)')
    parser.add_argument('--max-proteins', type=int, default=10, 
                        help='Max proteins to process')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use')
    parser.add_argument('--recycling-steps', type=int, default=0, 
                        help='Number of recycling steps')
    
    args = parser.parse_args()
    
    collect_activations_multi_layer(
        checkpoint_path=args.checkpoint,
        fasta_path=args.fasta,
        output_dir=args.output,
        layer_indices=args.layers,
        max_proteins=args.max_proteins,
        device=args.device,
        recycling_steps=args.recycling_steps
    )


if __name__ == "__main__":
    main()
