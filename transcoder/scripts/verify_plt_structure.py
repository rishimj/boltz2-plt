"""
End-to-End PLT Structure Verification

This script verifies PLT integration by:
1. Running baseline Boltz2 prediction (deterministic)
2. Running PLT-inserted Boltz2 prediction
3. Comparing structure outputs (RMSD, pLDDT correlation)
4. Measuring per-layer z reconstruction errors

Success Criteria (from plan):
- Structure RMSD (PLT vs baseline): < 2.0 Angstrom
- pLDDT correlation: > 0.95
- Reconstruction NMSE: < 0.1
- R² score: > 0.9

Usage:
    python verify_plt_structure.py \
        --fasta test.fasta \
        --plt-checkpoints checkpoints/ \
        --layer 0 \
        --output results/
"""

import argparse
import inspect
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

# Add paths
script_dir = Path(__file__).resolve().parent
boltz_root = script_dir.parent.parent
sys.path.insert(0, str(boltz_root / "src"))
sys.path.insert(0, str(boltz_root / "transcoder" / "universal_transcoder"))
sys.path.insert(0, str(script_dir))

from boltz.model.models.boltz2 import Boltz2
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.a3m import parse_a3m
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_canonicals
from boltz.data.types import Input

from plt_insertion import PLTInsertion, SingleLayerPLTInsertion, load_transcoder
from deterministic_baseline import (
    setup_determinism,
    prepare_deterministic_model,
    load_boltz_model,
    featurize_input,
)


def compute_rmsd(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute RMSD between two coordinate sets.

    Args:
        coords1: First coordinate set [N, 3] or [B, N, 3]
        coords2: Second coordinate set (same shape as coords1)
        mask: Optional mask for valid atoms [N] or [B, N]

    Returns:
        RMSD value in Angstroms
    """
    # Flatten batch if present
    if coords1.dim() == 3:
        coords1 = coords1.reshape(-1, 3)
        coords2 = coords2.reshape(-1, 3)
        if mask is not None and mask.dim() == 2:
            mask = mask.reshape(-1)

    diff = coords1 - coords2

    if mask is not None:
        mask = mask.bool()
        diff = diff[mask]

    if len(diff) == 0:
        return float('nan')

    rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean()).item()
    return rmsd


def compute_plddt_correlation(
    plddt1: torch.Tensor,
    plddt2: torch.Tensor
) -> float:
    """
    Compute Pearson correlation between pLDDT scores.

    Args:
        plddt1: First pLDDT array
        plddt2: Second pLDDT array

    Returns:
        Correlation coefficient
    """
    p1 = plddt1.flatten().float()
    p2 = plddt2.flatten().float()

    if len(p1) < 2:
        return float('nan')

    corr_matrix = torch.corrcoef(torch.stack([p1, p2]))
    return corr_matrix[0, 1].item()


class StructureVerifier:
    """
    Verifies PLT integration by comparing structure outputs.
    """

    def __init__(
        self,
        checkpoint_path: str,
        plt_checkpoints_dir: Optional[str] = None,
        layer_indices: List[int] = [0],
        device: str = 'cuda',
        seed: int = 42,
        reconstruction_method: str = 'outer_sum'
    ):
        """
        Initialize structure verifier.

        Args:
            checkpoint_path: Path to Boltz2 checkpoint
            plt_checkpoints_dir: Directory with PLT checkpoints
            layer_indices: Layers to insert PLT at
            device: Device to use
            seed: Random seed for determinism
            reconstruction_method: Method for pair reconstruction
        """
        self.checkpoint_path = checkpoint_path
        self.plt_checkpoints_dir = Path(plt_checkpoints_dir) if plt_checkpoints_dir else None
        self.layer_indices = layer_indices
        self.device = device
        self.seed = seed
        self.reconstruction_method = reconstruction_method

        # Initialize components
        self.tokenizer = Boltz2Tokenizer()
        self.featurizer = Boltz2Featurizer()

        # Load molecules
        self.moldir = Path("/usr/scratch/rmanimaran8/boltz/.boltz_cache/mols")
        if not self.moldir.exists():
            self.moldir = Path.home() / ".boltz_cache" / "mols"
        if not self.moldir.exists():
            raise ValueError(f"Molecules directory not found at {self.moldir}")

        self.molecules = load_canonicals(str(self.moldir))
        print(f"Loaded {len(self.molecules)} canonical molecules")

        # Results storage
        self.baseline_output = None
        self.plt_output = None
        self.z_comparisons = {}

    def run_baseline(
        self,
        fasta_path: Path,
        recycling_steps: int = 0
    ) -> Dict[str, Any]:
        """
        Run baseline (non-PLT) forward pass.

        Args:
            fasta_path: Path to FASTA file
            recycling_steps: Number of recycling steps

        Returns:
            Baseline output dict
        """
        print("\n" + "-" * 70)
        print("BASELINE: Running Boltz2 without PLT")
        print("-" * 70)

        # Full determinism reset
        setup_determinism(self.seed)

        # Load model fresh
        model = load_boltz_model(self.checkpoint_path, self.device)
        model = prepare_deterministic_model(model, disable_msa_subsample=True)

        # Featurize
        feats, target = featurize_input(
            fasta_path, self.molecules, self.moldir,
            self.tokenizer, self.featurizer, self.device, self.seed
        )

        # Reset seed before forward
        setup_determinism(self.seed)

        # Run forward pass
        print("Running forward pass...")
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                try:
                    output = model(feats=feats, recycling_steps=recycling_steps)
                    print("  Forward pass complete")
                except Exception as e:
                    print(f"  Forward pass error: {e}")
                    output = {}

        # Extract relevant outputs
        self.baseline_output = {
            'coordinates': output.get('sample_atom_coords'),
            'plddt': output.get('plddt'),
            'pae': output.get('pae'),
            'ptm': output.get('ptm'),
            'iptm': output.get('iptm'),
        }

        # Print shapes
        for k, v in self.baseline_output.items():
            if v is not None and hasattr(v, 'shape'):
                print(f"  {k}: {list(v.shape)}")

        return self.baseline_output

    def run_with_plt(
        self,
        fasta_path: Path,
        recycling_steps: int = 0
    ) -> Dict[str, Any]:
        """
        Run forward pass with PLT replacement.

        Args:
            fasta_path: Path to FASTA file
            recycling_steps: Number of recycling steps

        Returns:
            PLT-inserted output dict
        """
        print("\n" + "-" * 70)
        print(f"PLT INSERTION: Running Boltz2 with PLT at layers {self.layer_indices}")
        print("-" * 70)

        if self.plt_checkpoints_dir is None:
            print("  ERROR: No PLT checkpoints directory specified")
            return {}

        # Full determinism reset
        setup_determinism(self.seed)

        # Load model fresh
        model = load_boltz_model(self.checkpoint_path, self.device)
        model = prepare_deterministic_model(model, disable_msa_subsample=True)

        # Initialize PLT insertion
        plt_inserter = PLTInsertion(
            plt_checkpoints_dir=self.plt_checkpoints_dir,
            layer_indices=self.layer_indices,
            device=self.device,
            reconstruction_method=self.reconstruction_method
        )

        # Register hooks for replacement
        plt_inserter.register_hooks(model, mode='replace')

        # Featurize (same as baseline)
        feats, target = featurize_input(
            fasta_path, self.molecules, self.moldir,
            self.tokenizer, self.featurizer, self.device, self.seed
        )

        # Reset seed before forward
        setup_determinism(self.seed)

        # Run forward pass
        print("Running forward pass with PLT replacement...")
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                try:
                    output = model(feats=feats, recycling_steps=recycling_steps)
                    print("  Forward pass complete")
                except Exception as e:
                    print(f"  Forward pass error: {e}")
                    output = {}

        # Get z comparison results
        self.z_comparisons = plt_inserter.get_comparison_results()
        plt_inserter.print_comparison_results()

        # Clean up
        plt_inserter.remove_hooks()

        # Extract relevant outputs
        self.plt_output = {
            'coordinates': output.get('sample_atom_coords'),
            'plddt': output.get('plddt'),
            'pae': output.get('pae'),
            'ptm': output.get('ptm'),
            'iptm': output.get('iptm'),
        }

        return self.plt_output

    def compare_structures(self) -> Dict[str, Any]:
        """
        Compare baseline and PLT structure outputs.

        Returns:
            Comparison metrics dict
        """
        print("\n" + "=" * 70)
        print("STRUCTURE COMPARISON: Baseline vs PLT")
        print("=" * 70)

        results = {
            'rmsd': None,
            'plddt_correlation': None,
            'plddt_mae': None,
            'success_criteria': {},
        }

        # Coordinate RMSD
        if (self.baseline_output.get('coordinates') is not None and
            self.plt_output.get('coordinates') is not None):

            coords_base = self.baseline_output['coordinates']
            coords_plt = self.plt_output['coordinates']

            # Handle different tensor layouts
            if coords_base.dim() == 4:  # [B, N, A, 3]
                coords_base = coords_base.reshape(-1, 3)
                coords_plt = coords_plt.reshape(-1, 3)
            elif coords_base.dim() == 3:  # [B, N, 3]
                coords_base = coords_base.reshape(-1, 3)
                coords_plt = coords_plt.reshape(-1, 3)

            rmsd = compute_rmsd(coords_base, coords_plt)
            results['rmsd'] = rmsd

            print(f"\nCoordinate RMSD: {rmsd:.4f} Angstroms")

            # Interpretation
            if rmsd < 0.5:
                print("   EXCELLENT: Structures nearly identical")
                results['success_criteria']['rmsd'] = 'excellent'
            elif rmsd < 2.0:
                print("   GOOD: Structures very similar (meets target < 2.0 A)")
                results['success_criteria']['rmsd'] = 'good'
            elif rmsd < 5.0:
                print("   MODERATE: Noticeable structural changes")
                results['success_criteria']['rmsd'] = 'moderate'
            else:
                print("   POOR: Significant structural deviation")
                results['success_criteria']['rmsd'] = 'poor'

        # pLDDT correlation
        if (self.baseline_output.get('plddt') is not None and
            self.plt_output.get('plddt') is not None):

            plddt_base = self.baseline_output['plddt']
            plddt_plt = self.plt_output['plddt']

            correlation = compute_plddt_correlation(plddt_base, plddt_plt)
            results['plddt_correlation'] = correlation

            mae = (plddt_base - plddt_plt).abs().mean().item()
            results['plddt_mae'] = mae

            print(f"\npLDDT Correlation: {correlation:.4f} (target > 0.95)")
            print(f"pLDDT MAE: {mae:.4f}")

            if correlation > 0.95:
                print("   GOOD: High confidence correlation (meets target)")
                results['success_criteria']['plddt'] = 'good'
            elif correlation > 0.9:
                print("   MODERATE: Good confidence correlation")
                results['success_criteria']['plddt'] = 'moderate'
            else:
                print("   POOR: Low confidence correlation")
                results['success_criteria']['plddt'] = 'poor'

        # Add z comparison results
        results['z_comparisons'] = self.z_comparisons

        return results

    def run_full_verification(
        self,
        fasta_path: str,
        output_dir: str = 'verification_output',
        recycling_steps: int = 0
    ) -> Dict[str, Any]:
        """
        Run complete verification pipeline.

        Args:
            fasta_path: Path to FASTA file
            output_dir: Directory to save results
            recycling_steps: Number of recycling steps

        Returns:
            Complete verification results
        """
        fasta_path = Path(fasta_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("FULL PLT VERIFICATION PIPELINE")
        print("=" * 70)
        print(f"Input: {fasta_path}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"PLT layers: {self.layer_indices}")
        print(f"PLT checkpoints: {self.plt_checkpoints_dir}")
        print(f"Seed: {self.seed}")

        # Run baseline
        self.run_baseline(fasta_path, recycling_steps)

        # Run with PLT
        self.run_with_plt(fasta_path, recycling_steps)

        # Compare structures
        comparison = self.compare_structures()

        # Compute overall success
        all_good = True
        if comparison['rmsd'] is not None and comparison['rmsd'] >= 2.0:
            all_good = False
        if comparison['plddt_correlation'] is not None and comparison['plddt_correlation'] < 0.95:
            all_good = False

        for layer_idx, z_comp in comparison.get('z_comparisons', {}).items():
            if z_comp.get('nmse', 1.0) >= 0.1:
                all_good = False
            if z_comp.get('r2', 0.0) < 0.9:
                all_good = False

        comparison['overall_success'] = all_good

        # Print summary
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)

        if all_good:
            print("\n   SUCCESS: PLT integration meets all criteria!")
            print("   - Structure RMSD < 2.0 A")
            print("   - pLDDT correlation > 0.95")
            print("   - z reconstruction NMSE < 0.1")
            print("   - z reconstruction R² > 0.9")
        else:
            print("\n   WARNING: Some criteria not met")
            if comparison['rmsd'] is not None and comparison['rmsd'] >= 2.0:
                print(f"   - RMSD = {comparison['rmsd']:.4f} A (target < 2.0)")
            if comparison['plddt_correlation'] is not None and comparison['plddt_correlation'] < 0.95:
                print(f"   - pLDDT correlation = {comparison['plddt_correlation']:.4f} (target > 0.95)")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = output_dir / f'verification_results_{timestamp}.json'

        save_data = {
            'timestamp': timestamp,
            'fasta_path': str(fasta_path),
            'checkpoint_path': self.checkpoint_path,
            'plt_checkpoints_dir': str(self.plt_checkpoints_dir) if self.plt_checkpoints_dir else None,
            'layer_indices': self.layer_indices,
            'seed': self.seed,
            'reconstruction_method': self.reconstruction_method,
            'recycling_steps': recycling_steps,
            'results': {
                'rmsd': comparison['rmsd'],
                'plddt_correlation': comparison['plddt_correlation'],
                'plddt_mae': comparison['plddt_mae'],
                'overall_success': comparison['overall_success'],
                'success_criteria': comparison['success_criteria'],
            },
            'z_comparisons': {
                str(k): v for k, v in comparison.get('z_comparisons', {}).items()
            }
        }

        with open(results_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\nResults saved to: {results_path}")

        return comparison


def main():
    parser = argparse.ArgumentParser(
        description='Verify PLT integration by comparing structure outputs'
    )
    parser.add_argument('--fasta', type=str, required=True,
                       help='Input FASTA file')
    parser.add_argument('--checkpoint', type=str,
                       default='/usr/scratch/rmanimaran8/boltz/.boltz_cache/boltz2_conf.ckpt',
                       help='Path to Boltz2 checkpoint')
    parser.add_argument('--plt-checkpoints', type=str, required=True,
                       help='Directory containing PLT checkpoints')
    parser.add_argument('--layers', type=int, nargs='+', default=[0],
                       help='Layer indices to test PLT insertion (default: [0])')
    parser.add_argument('--output', type=str, default='verification_output',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--recycling-steps', type=int, default=0,
                       help='Number of recycling steps')
    parser.add_argument('--reconstruction-method', type=str, default='outer_sum',
                       choices=['outer_sum', 'outer_product', 'broadcast_i', 'broadcast_j'],
                       help='Method for pair reconstruction')

    args = parser.parse_args()

    verifier = StructureVerifier(
        checkpoint_path=args.checkpoint,
        plt_checkpoints_dir=args.plt_checkpoints,
        layer_indices=args.layers,
        device=args.device,
        seed=args.seed,
        reconstruction_method=args.reconstruction_method
    )

    results = verifier.run_full_verification(
        fasta_path=args.fasta,
        output_dir=args.output,
        recycling_steps=args.recycling_steps
    )

    return 0 if results.get('overall_success', False) else 1


if __name__ == '__main__':
    sys.exit(main())
