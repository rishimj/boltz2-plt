"""
PLT Insertion Module for Boltz2

This module provides utilities for inserting trained PLT (Per-Layer Transcoder)
models into the Boltz2 forward pass, allowing replacement or comparison of
the pair representation (z) at specific pairformer layers.

Key Design Decision:
    The PLT outputs token-level predictions [B, N, 128], but z is pair-level
    [B, N, N, 128]. We use OUTER SUM to reconstruct the pair representation:
    z[i,j] = y[i] + y[j] - symmetric, captures pairwise relationships.

Usage:
    from plt_insertion import PLTInsertion

    plt_inserter = PLTInsertion(
        plt_checkpoints_dir='checkpoints/',
        layer_indices=[0, 8, 16, 24, 32, 40]
    )

    # Register hooks for replacement
    plt_inserter.register_hooks(model, mode='replace')

    # Run forward pass
    output = model(feats, recycling_steps=0)

    # Get comparison results
    results = plt_inserter.get_comparison_results()

    # Clean up
    plt_inserter.remove_hooks()
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Any

import torch
import torch.nn as nn

# Add paths
script_dir = Path(__file__).resolve().parent
boltz_root = script_dir.parent.parent
sys.path.insert(0, str(boltz_root / "transcoder" / "universal_transcoder"))
sys.path.insert(0, str(boltz_root / "src"))

from universal_model import UniversalTranscoder


def load_transcoder(
    checkpoint_path: Path,
    device: str = 'cuda'
) -> UniversalTranscoder:
    """
    Load a trained PLT/UniversalTranscoder from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded and configured UniversalTranscoder
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get hyperparameters from checkpoint or use defaults
    hparams = checkpoint.get('hyperparameters', {})

    model = UniversalTranscoder(
        d_model=hparams.get('d_model', 384),
        d_hidden=hparams.get('d_hidden', 2048),
        d_pair=hparams.get('d_pair', 128),
        k=hparams.get('k', 16),
        auxk=hparams.get('auxk', 32),
        batch_size=hparams.get('batch_size', 10),
        dead_steps_threshold=hparams.get('dead_steps_threshold', 10000),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def reconstruct_pair_from_token(
    token_pred: torch.Tensor,
    method: str = 'outer_sum'
) -> torch.Tensor:
    """
    Reconstruct pair representation [B, N, N, D] from token prediction [B, N, D].

    Args:
        token_pred: Token-level prediction [B, N, D] or [B*N, D]
        method: Reconstruction method:
            - 'outer_sum': z[i,j] = y[i] + y[j] (symmetric)
            - 'outer_product': z[i,j] = y[i] * y[j] (symmetric)
            - 'broadcast_i': z[i,j] = y[i] (row broadcast)
            - 'broadcast_j': z[i,j] = y[j] (column broadcast)

    Returns:
        Pair representation [B, N, N, D]
    """
    if token_pred.dim() == 2:
        # Assume flat input, need batch and N info
        raise ValueError(
            "Token prediction must have shape [B, N, D]. "
            "Use reshape before calling this function."
        )

    B, N, D = token_pred.shape

    if method == 'outer_sum':
        # z[i,j] = y[i] + y[j] - symmetric outer sum
        # Captures additive pairwise relationships
        z_pred = token_pred.unsqueeze(2) + token_pred.unsqueeze(1)

    elif method == 'outer_product':
        # z[i,j] = y[i] * y[j] - symmetric outer product
        z_pred = token_pred.unsqueeze(2) * token_pred.unsqueeze(1)

    elif method == 'broadcast_i':
        # z[i,j] = y[i] - broadcast along rows
        z_pred = token_pred.unsqueeze(2).expand(B, N, N, D)

    elif method == 'broadcast_j':
        # z[i,j] = y[j] - broadcast along columns
        z_pred = token_pred.unsqueeze(1).expand(B, N, N, D)

    else:
        raise ValueError(f"Unknown reconstruction method: {method}")

    return z_pred


class PLTInsertion:
    """
    Manages PLT insertion into Boltz2 forward pass.

    Supports three modes:
    - 'capture': Just capture activations for comparison
    - 'replace': Replace z with PLT reconstruction
    - 'compare': Capture both original and PLT, compare
    """

    def __init__(
        self,
        plt_checkpoints_dir: Optional[Path] = None,
        layer_indices: List[int] = [0, 8, 16, 24, 32, 40],
        device: str = 'cuda',
        reconstruction_method: str = 'outer_sum'
    ):
        """
        Initialize PLT insertion manager.

        Args:
            plt_checkpoints_dir: Directory containing layer-specific PLT checkpoints
                Expected structure: {dir}/layer_XX/universal_transcoder_final.pt
            layer_indices: Layer indices where PLTs are applied
            device: Device for PLT models
            reconstruction_method: Method to reconstruct pair from token predictions
        """
        self.plt_checkpoints_dir = Path(plt_checkpoints_dir) if plt_checkpoints_dir else None
        self.layer_indices = layer_indices
        self.device = device
        self.reconstruction_method = reconstruction_method

        # PLT models (loaded on demand)
        self.plts: Dict[int, UniversalTranscoder] = {}

        # Storage for captured activations and comparisons
        self.captured_s: Dict[int, torch.Tensor] = {}
        self.captured_z_original: Dict[int, torch.Tensor] = {}
        self.captured_z_plt: Dict[int, torch.Tensor] = {}

        # Hook handles
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Current mode
        self.mode: str = 'capture'

        # Load PLTs if checkpoint dir provided
        if self.plt_checkpoints_dir:
            self._load_plts()

    def _load_plts(self) -> None:
        """Load PLT checkpoints for all specified layers."""
        for layer_idx in self.layer_indices:
            checkpoint_path = (
                self.plt_checkpoints_dir /
                f"layer_{layer_idx:02d}" /
                "universal_transcoder_final.pt"
            )

            if checkpoint_path.exists():
                print(f"  Loading PLT for layer {layer_idx} from {checkpoint_path}")
                self.plts[layer_idx] = load_transcoder(checkpoint_path, self.device)
            else:
                print(f"  Warning: No PLT checkpoint found for layer {layer_idx} at {checkpoint_path}")

    def register_hooks(
        self,
        model: nn.Module,
        mode: Literal['capture', 'replace', 'compare'] = 'capture'
    ) -> None:
        """
        Register forward hooks on model for specified mode.

        Args:
            model: Boltz2 model
            mode: Operation mode:
                - 'capture': Just capture original activations
                - 'replace': Replace z with PLT reconstruction
                - 'compare': Capture both original and PLT
        """
        self.mode = mode
        self.remove_hooks()  # Clear any existing hooks
        self.clear_captured()

        for layer_idx in self.layer_indices:
            if layer_idx >= len(model.pairformer_module.layers):
                print(f"  Warning: Layer {layer_idx} out of range, skipping")
                continue

            layer = model.pairformer_module.layers[layer_idx]

            # Hook to capture s (single representation)
            # This runs BEFORE the z hook, so s is available when z hook fires
            def make_capture_s_hook(idx):
                def hook(module, input, output):
                    # Store the OUTPUT of transition_s
                    self.captured_s[idx] = output.detach().clone()
                return hook

            # Hook for z (pair representation)
            if mode == 'capture':
                def make_capture_z_hook(idx):
                    def hook(module, input, output):
                        self.captured_z_original[idx] = output.detach().cpu().clone()
                    return hook
                z_hook = make_capture_z_hook(layer_idx)

            elif mode == 'replace':
                def make_replace_z_hook(idx):
                    def hook(module, input, output):
                        # Get the single rep we captured
                        s = self.captured_s.get(idx)
                        if s is None:
                            print(f"  Warning: No s captured for layer {idx}, using original z")
                            return output

                        # Get PLT for this layer
                        plt = self.plts.get(idx)
                        if plt is None:
                            print(f"  Warning: No PLT for layer {idx}, using original z")
                            return output

                        # Run PLT
                        B, N, _ = s.shape
                        s_flat = s.reshape(B * N, -1)

                        with torch.no_grad():
                            y1, y2, _, _, _ = plt(s_flat)

                        # Reshape to [B, N, D]
                        y2 = y2.reshape(B, N, -1)

                        # Reconstruct pair representation
                        z_pred = reconstruct_pair_from_token(
                            y2, method=self.reconstruction_method
                        )

                        # Store for comparison
                        self.captured_z_plt[idx] = z_pred.detach().cpu().clone()
                        self.captured_z_original[idx] = output.detach().cpu().clone()

                        # Return the PLT reconstruction (REPLACES original z)
                        return z_pred

                    return hook
                z_hook = make_replace_z_hook(layer_idx)

            elif mode == 'compare':
                def make_compare_z_hook(idx):
                    def hook(module, input, output):
                        # Store original
                        self.captured_z_original[idx] = output.detach().cpu().clone()

                        # Get s for PLT
                        s = self.captured_s.get(idx)
                        if s is None:
                            return output

                        # Get PLT for this layer
                        plt = self.plts.get(idx)
                        if plt is None:
                            return output

                        # Run PLT
                        B, N, _ = s.shape
                        s_flat = s.reshape(B * N, -1)

                        with torch.no_grad():
                            y1, y2, _, _, _ = plt(s_flat)

                        # Reshape and reconstruct
                        y2 = y2.reshape(B, N, -1)
                        z_pred = reconstruct_pair_from_token(
                            y2, method=self.reconstruction_method
                        )

                        # Store PLT prediction
                        self.captured_z_plt[idx] = z_pred.detach().cpu().clone()

                        # Return ORIGINAL (don't replace)
                        return output

                    return hook
                z_hook = make_compare_z_hook(layer_idx)

            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Register hooks
            # IMPORTANT: s hook must be registered BEFORE z hook
            if hasattr(layer, 'transition_s'):
                self.hooks.append(
                    layer.transition_s.register_forward_hook(make_capture_s_hook(layer_idx))
                )

            if hasattr(layer, 'transition_z'):
                self.hooks.append(
                    layer.transition_z.register_forward_hook(z_hook)
                )

        print(f"  Registered {len(self.hooks)} hooks for {len(self.layer_indices)} layers (mode: {mode})")

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_captured(self) -> None:
        """Clear all captured activations."""
        self.captured_s = {}
        self.captured_z_original = {}
        self.captured_z_plt = {}

    def get_comparison_results(self) -> Dict[int, Dict[str, Any]]:
        """
        Compute comparison metrics between original and PLT z values.

        Returns:
            Dict mapping layer_idx to comparison metrics
        """
        results = {}

        for layer_idx in sorted(set(self.captured_z_original.keys()) & set(self.captured_z_plt.keys())):
            z_orig = self.captured_z_original[layer_idx]
            z_plt = self.captured_z_plt[layer_idx]

            # Compute metrics
            diff = z_orig - z_plt
            mse = (diff ** 2).mean().item()
            nmse = mse / (z_orig.var().item() + 1e-8)
            max_diff = diff.abs().max().item()
            mean_diff = diff.abs().mean().item()

            # Correlation
            z_orig_flat = z_orig.flatten()
            z_plt_flat = z_plt.flatten()
            correlation = torch.corrcoef(torch.stack([z_orig_flat, z_plt_flat]))[0, 1].item()

            # R² score
            ss_res = ((z_orig_flat - z_plt_flat) ** 2).sum().item()
            ss_tot = ((z_orig_flat - z_orig_flat.mean()) ** 2).sum().item()
            r2 = 1 - (ss_res / (ss_tot + 1e-8))

            results[layer_idx] = {
                'mse': mse,
                'nmse': nmse,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'correlation': correlation,
                'r2': r2,
                'shape': list(z_orig.shape),
            }

        return results

    def print_comparison_results(self) -> None:
        """Pretty print comparison results."""
        results = self.get_comparison_results()

        print("\n" + "=" * 70)
        print("PLT vs ORIGINAL z COMPARISON")
        print("=" * 70)

        for layer_idx, metrics in sorted(results.items()):
            status = "GOOD" if metrics['nmse'] < 0.1 and metrics['r2'] > 0.9 else "CHECK"
            print(f"\nLayer {layer_idx} [{status}]:")
            print(f"  Shape: {metrics['shape']}")
            print(f"  NMSE:        {metrics['nmse']:.6f} (target < 0.1)")
            print(f"  R² Score:    {metrics['r2']:.6f} (target > 0.9)")
            print(f"  Correlation: {metrics['correlation']:.6f}")
            print(f"  Max Diff:    {metrics['max_diff']:.6f}")
            print(f"  Mean Diff:   {metrics['mean_diff']:.6f}")


class SingleLayerPLTInsertion:
    """
    Simplified PLT insertion for a single layer.

    Use this for testing/debugging individual layers before scaling up.
    """

    def __init__(
        self,
        plt_checkpoint: Path,
        layer_idx: int,
        device: str = 'cuda',
        reconstruction_method: str = 'outer_sum'
    ):
        """
        Initialize single-layer PLT insertion.

        Args:
            plt_checkpoint: Path to PLT checkpoint file
            layer_idx: Target layer index
            device: Device for PLT model
            reconstruction_method: Method for pair reconstruction
        """
        self.layer_idx = layer_idx
        self.device = device
        self.reconstruction_method = reconstruction_method

        # Load PLT
        print(f"Loading PLT for layer {layer_idx}...")
        self.plt = load_transcoder(Path(plt_checkpoint), device)

        # Storage
        self.captured_s: Optional[torch.Tensor] = None
        self.captured_z_original: Optional[torch.Tensor] = None
        self.captured_z_plt: Optional[torch.Tensor] = None

        # Hooks
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register_hooks(
        self,
        model: nn.Module,
        replace: bool = True
    ) -> None:
        """
        Register hooks on target layer.

        Args:
            model: Boltz2 model
            replace: If True, replace z with PLT. If False, just capture.
        """
        self.remove_hooks()

        layer = model.pairformer_module.layers[self.layer_idx]

        # Capture s hook
        def capture_s_hook(module, input, output):
            self.captured_s = output.detach().clone()

        # Z hook with optional replacement
        def z_hook(module, input, output):
            self.captured_z_original = output.detach().clone()

            if self.captured_s is None:
                return output

            # Run PLT
            B, N, _ = self.captured_s.shape
            s_flat = self.captured_s.reshape(B * N, -1)

            with torch.no_grad():
                _, y2, _, _, _ = self.plt(s_flat)

            y2 = y2.reshape(B, N, -1)
            z_pred = reconstruct_pair_from_token(y2, method=self.reconstruction_method)
            self.captured_z_plt = z_pred.detach().clone()

            if replace:
                return z_pred
            return output

        # Register
        self.hooks.append(layer.transition_s.register_forward_hook(capture_s_hook))
        self.hooks.append(layer.transition_z.register_forward_hook(z_hook))

        print(f"  Registered hooks for layer {self.layer_idx} (replace={replace})")

    def remove_hooks(self) -> None:
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_z_comparison(self) -> Dict[str, Any]:
        """Get comparison between original and PLT z."""
        if self.captured_z_original is None or self.captured_z_plt is None:
            return {}

        z_orig = self.captured_z_original.cpu()
        z_plt = self.captured_z_plt.cpu()

        diff = z_orig - z_plt
        mse = (diff ** 2).mean().item()
        nmse = mse / (z_orig.var().item() + 1e-8)

        z_orig_flat = z_orig.flatten()
        z_plt_flat = z_plt.flatten()
        ss_res = ((z_orig_flat - z_plt_flat) ** 2).sum().item()
        ss_tot = ((z_orig_flat - z_orig_flat.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'mse': mse,
            'nmse': nmse,
            'r2': r2,
            'max_diff': diff.abs().max().item(),
            'shape': list(z_orig.shape),
        }


if __name__ == '__main__':
    # Quick test
    print("PLT Insertion Module loaded successfully")
    print("Available classes: PLTInsertion, SingleLayerPLTInsertion")
    print("Available functions: load_transcoder, reconstruct_pair_from_token")
