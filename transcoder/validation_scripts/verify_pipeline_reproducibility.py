"""
Verify Boltz2 Pipeline Reproducibility

This script tests:
1. Baseline: Run Boltz2 normally, capture final structure output
2. Control: Run Boltz2 again, verify outputs are identical (reproducibility check)
3. Intervention: Replace layer 47 activations with transcoder reconstructions
4. Compare: Check if intervention changes final structure

If transcoder has perfect reconstruction (R² = 1.0), intervention should 
produce identical structures as baseline.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'universal_transcoder'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from boltz.main import Boltz
from boltz.data.parse.fasta import parse_fasta
from universal_model import UniversalTranscoder


class ReproducibilityTester:
    """Test if Boltz2 pipeline is reproducible with/without transcoder"""
    
    def __init__(self, checkpoint_path, transcoder_path=None):
        """
        Args:
            checkpoint_path: Path to Boltz2 checkpoint
            transcoder_path: Path to trained transcoder checkpoint (optional)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load Boltz2
        print(f"\nLoading Boltz2 from {checkpoint_path}")
        self.boltz = Boltz.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device
        )
        self.boltz.eval()
        print("✓ Boltz2 loaded")
        
        # Load transcoder if provided
        self.transcoder = None
        if transcoder_path:
            print(f"\nLoading transcoder from {transcoder_path}")
            checkpoint = torch.load(transcoder_path, map_location=self.device)
            self.transcoder = UniversalTranscoder(
                d_model=384,
                d_hidden=2048,
                d_pair=128,
                k=16
            ).to(self.device)
            self.transcoder.load_state_dict(checkpoint['model_state_dict'])
            self.transcoder.eval()
            print("✓ Transcoder loaded")
        
        # Storage for captured activations
        self.captured = {}
        self.hooks = []
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.captured = {}
    
    def run_baseline(self, fasta_path):
        """
        Run Boltz2 normally without any intervention.
        
        Returns:
            dict with:
                - coordinates: Final atomic coordinates
                - confidence: pLDDT scores
                - layer47_s: Single rep from layer 47
                - layer47_z: Pair rep from layer 47
        """
        print(f"\n{'='*60}")
        print("BASELINE: Normal Boltz2 Prediction")
        print(f"{'='*60}")
        
        self.clear_hooks()
        
        # Register hooks to capture layer 47 outputs
        def hook_s(module, input, output):
            self.captured['layer47_s'] = output.detach().cpu().clone()
            print(f"  ✓ Captured layer47_s: {output.shape}")
        
        def hook_z(module, input, output):
            self.captured['layer47_z'] = output.detach().cpu().clone()
            print(f"  ✓ Captured layer47_z: {output.shape}")
        
        layer47 = self.boltz.model.pairformer_module.layers[47]
        self.hooks.append(layer47.transition_s.register_forward_hook(hook_s))
        self.hooks.append(layer47.transition_z.register_forward_hook(hook_z))
        
        # Run prediction
        print(f"Running prediction on {fasta_path}...")
        with torch.no_grad():
            output = self.boltz.predict(fasta_path)
        
        result = {
            'coordinates': output.get('coordinates', None),
            'confidence': output.get('plddt', None),
            'layer47_s': self.captured.get('layer47_s'),
            'layer47_z': self.captured.get('layer47_z'),
        }
        
        print(f"✓ Baseline complete")
        if result['coordinates'] is not None:
            print(f"  Coordinates shape: {result['coordinates'].shape}")
        if result['confidence'] is not None:
            print(f"  Confidence mean: {result['confidence'].mean():.2f}")
        
        return result
    
    def run_control(self, fasta_path):
        """
        Run Boltz2 again to verify reproducibility.
        Should produce identical results to baseline (sanity check).
        """
        print(f"\n{'='*60}")
        print("CONTROL: Verify Reproducibility")
        print(f"{'='*60}")
        
        self.clear_hooks()
        
        print(f"Running prediction on {fasta_path}...")
        with torch.no_grad():
            output = self.boltz.predict(fasta_path)
        
        result = {
            'coordinates': output.get('coordinates', None),
            'confidence': output.get('plddt', None),
        }
        
        print(f"✓ Control complete")
        return result
    
    def run_intervention(self, fasta_path):
        """
        Run Boltz2 with layer 47 activations REPLACED by transcoder reconstructions.
        
        This tests if transcoder preserves functionally important information.
        """
        if self.transcoder is None:
            print("⚠️ No transcoder loaded - skipping intervention")
            return None
        
        print(f"\n{'='*60}")
        print("INTERVENTION: Replace Layer 47 with Transcoder Reconstructions")
        print(f"{'='*60}")
        
        self.clear_hooks()
        
        # Storage for single rep to use in intervention
        intervention_storage = {}
        
        # Hook 1: Capture single rep from layer 47
        def capture_s_hook(module, input, output):
            intervention_storage['output_s'] = output.detach().clone()
            print(f"  ✓ Captured layer47_s for transcoding: {output.shape}")
            return output  # Don't modify
        
        # Hook 2: Replace pair rep with transcoder reconstruction
        def replace_z_hook(module, input, output):
            """Replace layer 47 pair output with transcoder reconstruction"""
            # Get the single rep we just captured
            output_s = intervention_storage.get('output_s')
            
            if output_s is None:
                print("  ⚠️ No single rep captured - returning original")
                return output
            
            # Run through transcoder
            with torch.no_grad():
                # Reshape for transcoder: [batch, N, 384]
                batch_size = output_s.shape[0]
                N = output_s.shape[1]
                s_flat = output_s.reshape(batch_size, N, -1)
                
                # Get reconstruction
                y1_recon, y2_recon, _, _, _ = self.transcoder(s_flat)
                
                # y2_recon is [batch, N, 128] - expand to [batch, N, N, 128]
                # Assuming symmetric pair representation
                y2_expanded = y2_recon.unsqueeze(2).expand(batch_size, N, N, -1)
                
                print(f"  ✓ Transcoder reconstruction: {y2_recon.shape} → {y2_expanded.shape}")
                print(f"  ✓ REPLACING layer47_z with transcoder output!")
                
                # Return reconstruction instead of original
                return y2_expanded
        
        # Register hooks
        layer47 = self.boltz.model.pairformer_module.layers[47]
        self.hooks.append(layer47.transition_s.register_forward_hook(capture_s_hook))
        self.hooks.append(layer47.transition_z.register_forward_hook(replace_z_hook))
        
        # Run prediction with intervention
        print(f"Running prediction with transcoder intervention...")
        with torch.no_grad():
            output = self.boltz.predict(fasta_path)
        
        result = {
            'coordinates': output.get('coordinates', None),
            'confidence': output.get('plddt', None),
        }
        
        print(f"✓ Intervention complete")
        return result
    
    def compare_outputs(self, baseline, control, intervention=None):
        """
        Compare outputs from different runs.
        
        Args:
            baseline: Output from first run
            control: Output from second run (should be identical)
            intervention: Output from transcoder intervention run
        """
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")
        
        results = {}
        
        # 1. Reproducibility check: Baseline vs Control
        print("\n1. Reproducibility Check (Baseline vs Control)")
        print("-" * 60)
        
        if baseline['coordinates'] is not None and control['coordinates'] is not None:
            coord_diff = torch.abs(baseline['coordinates'] - control['coordinates']).max()
            print(f"   Max coordinate difference: {coord_diff:.2e}")
            
            if coord_diff < 1e-5:
                print("   ✅ PERFECT reproducibility - outputs identical!")
            else:
                print(f"   ⚠️ Small differences detected (likely numerical precision)")
            
            results['reproducibility_max_diff'] = coord_diff.item()
        
        # 2. Intervention effect: Baseline vs Intervention
        if intervention is not None:
            print("\n2. Intervention Effect (Baseline vs Transcoder)")
            print("-" * 60)
            
            if baseline['coordinates'] is not None and intervention['coordinates'] is not None:
                # Coordinate RMSD
                coord_diff = baseline['coordinates'] - intervention['coordinates']
                rmsd = torch.sqrt((coord_diff ** 2).sum(-1).mean())
                max_diff = torch.abs(coord_diff).max()
                
                print(f"   Coordinate RMSD: {rmsd:.4f} Å")
                print(f"   Max atom deviation: {max_diff:.4f} Å")
                
                # Interpretation
                if rmsd < 0.5:
                    print("   ✅ EXCELLENT: Structures nearly identical")
                    print("   → Transcoder preserves all critical information!")
                elif rmsd < 2.0:
                    print("   ✅ GOOD: Structures very similar")
                    print("   → Transcoder preserves most important features")
                elif rmsd < 5.0:
                    print("   ⚠️ MODERATE: Noticeable structural changes")
                    print("   → Transcoder loses some important information")
                else:
                    print("   ❌ POOR: Significant structural changes")
                    print("   → Transcoder does not preserve fold")
                
                results['intervention_rmsd'] = rmsd.item()
                results['intervention_max_diff'] = max_diff.item()
            
            # Confidence comparison
            if baseline['confidence'] is not None and intervention['confidence'] is not None:
                conf_corr = torch.corrcoef(torch.stack([
                    baseline['confidence'].flatten(),
                    intervention['confidence'].flatten()
                ]))[0, 1]
                
                conf_mae = torch.abs(baseline['confidence'] - intervention['confidence']).mean()
                
                print(f"\n   Confidence correlation: {conf_corr:.4f}")
                print(f"   Confidence MAE: {conf_mae:.4f}")
                
                results['confidence_correlation'] = conf_corr.item()
                results['confidence_mae'] = conf_mae.item()
        
        return results
    
    def run_full_test(self, fasta_path, output_dir='validation_output'):
        """
        Run complete validation pipeline:
        1. Baseline
        2. Control (reproducibility check)
        3. Intervention (if transcoder available)
        4. Comparison
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("FULL VALIDATION PIPELINE")
        print(f"{'='*60}")
        print(f"Input: {fasta_path}")
        print(f"Output: {output_dir}")
        
        # Run experiments
        baseline = self.run_baseline(fasta_path)
        control = self.run_control(fasta_path)
        intervention = self.run_intervention(fasta_path) if self.transcoder else None
        
        # Compare
        results = self.compare_outputs(baseline, control, intervention)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f'validation_results_{timestamp}.json'
        
        save_data = {
            'timestamp': timestamp,
            'fasta_path': str(fasta_path),
            'transcoder_used': self.transcoder is not None,
            'results': {k: float(v) if isinstance(v, (int, float, torch.Tensor)) else v 
                       for k, v in results.items()}
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
        
        # Save activations
        if baseline.get('layer47_s') is not None:
            activations_file = output_dir / f'layer47_activations_{timestamp}.npz'
            np.savez_compressed(
                activations_file,
                layer47_s=baseline['layer47_s'].numpy(),
                layer47_z=baseline['layer47_z'].numpy(),
            )
            print(f"✓ Layer 47 activations saved to {activations_file}")
        
        return results


def main():
    """Run validation experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Boltz2 pipeline reproducibility')
    parser.add_argument('--fasta', type=str, required=True,
                       help='Input FASTA file')
    parser.add_argument('--boltz-checkpoint', type=str,
                       default='/usr/scratch/rmanimaran8/boltz/.boltz_cache/boltz2_conf.ckpt',
                       help='Path to Boltz2 checkpoint')
    parser.add_argument('--transcoder', type=str,
                       default='../universal_transcoder/checkpoints/universal_transcoder_final.pt',
                       help='Path to transcoder checkpoint (optional)')
    parser.add_argument('--output', type=str,
                       default='validation_output',
                       help='Output directory')
    parser.add_argument('--no-transcoder', action='store_true',
                       help='Skip transcoder intervention (reproducibility check only)')
    
    args = parser.parse_args()
    
    # Initialize tester
    transcoder_path = None if args.no_transcoder else args.transcoder
    tester = ReproducibilityTester(
        checkpoint_path=args.boltz_checkpoint,
        transcoder_path=transcoder_path
    )
    
    # Run full test
    results = tester.run_full_test(
        fasta_path=args.fasta,
        output_dir=args.output
    )
    
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}")
    
    return results


if __name__ == '__main__':
    main()
