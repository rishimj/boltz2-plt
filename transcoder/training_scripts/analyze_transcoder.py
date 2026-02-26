"""
Analyze what the Universal Transcoder learned from Boltz activations.
"""
import torch
import numpy as np
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent / "universal_transcoder"))
from universal_model import UniversalTranscoder

# Try to import matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available, skipping visualizations")


def load_trained_model(checkpoint_path, device='cuda'):
    """Load the trained transcoder."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get hyperparameters
    hp = checkpoint['hyperparameters']
    
    # Initialize model
    model = UniversalTranscoder(
        d_model=hp['d_model'],
        d_hidden=hp['d_hidden'],
        d_pair=hp['d_pair'],
        k=hp['k'],
        auxk=hp['auxk'],
        dead_steps_threshold=hp['dead_steps_threshold'],
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from step {checkpoint['step']}")
    print(f"  Final loss: {checkpoint.get('loss', 'N/A')}")
    
    return model, hp


def analyze_protein(model, data, protein_name, device='cuda'):
    """Analyze transcoder predictions for a single protein."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {protein_name}")
    print(f"{'='*60}")
    
    # Load activations
    s1 = torch.from_numpy(data['input_s']).float().to(device)   # [1, N, 384]
    s2 = torch.from_numpy(data['output_s']).float().to(device)  # [1, N, 384]
    y1_true = torch.from_numpy(data['input_z']).float().to(device)   # [1, N, N, 128]
    y2_true = torch.from_numpy(data['output_z']).float().to(device)  # [1, N, N, 128]
    
    B, N, _ = s1.shape
    print(f"Protein size: {N} residues")
    print(f"Activations: s{s1.shape}, z{y1_true.shape}")
    
    # Flatten for model
    s1_flat = s1.reshape(B * N, -1)  # [B*N, 384]
    s2_flat = s2.reshape(B * N, -1)  # [B*N, 384]
    
    # Flatten targets
    y1_true_flat = y1_true.reshape(B, N * N, -1).reshape(B * N * N, -1)  # [B*N*N, 128]
    y2_true_flat = y2_true.reshape(B, N * N, -1).reshape(B * N * N, -1)  # [B*N*N, 128]
    
    # Run transcoder
    with torch.no_grad():
        y1_pred_s1, y2_pred_s1, aux_y1_s1, aux_y2_s1, dead_mask_s1 = model(s1_flat)
        y1_pred_s2, y2_pred_s2, aux_y1_s2, aux_y2_s2, dead_mask_s2 = model(s2_flat)
    
    # Get latent activations (before TopK)
    # We need to do a partial forward to get the latents
    with torch.no_grad():
        # Manually compute encoder activations
        x_norm1, mu1, std1 = model.LN(s1_flat)
        x_centered1 = x_norm1 - model.b_pre
        pre_acts1 = model.encoder(x_centered1) + model.b_enc
        hidden_s1 = model.topK_activation(pre_acts1, k=model.k)
        
        x_norm2, mu2, std2 = model.LN(s2_flat)
        x_centered2 = x_norm2 - model.b_pre
        pre_acts2 = model.encoder(x_centered2) + model.b_enc
        hidden_s2 = model.topK_activation(pre_acts2, k=model.k)
    
    # Expand predictions to match target shape [B*N*N, 128]
    y1_pred_s1_exp = y1_pred_s1.unsqueeze(1).expand(B * N, N, -1).reshape(B * N * N, -1)
    y2_pred_s1_exp = y2_pred_s1.unsqueeze(1).expand(B * N, N, -1).reshape(B * N * N, -1)
    y1_pred_s2_exp = y1_pred_s2.unsqueeze(1).expand(B * N, N, -1).reshape(B * N * N, -1)
    y2_pred_s2_exp = y2_pred_s2.unsqueeze(1).expand(B * N, N, -1).reshape(B * N * N, -1)
    
    # Compute reconstruction errors
    mse_y1_s1 = torch.mean((y1_pred_s1_exp - y1_true_flat)**2).item()
    mse_y2_s1 = torch.mean((y2_pred_s1_exp - y2_true_flat)**2).item()
    mse_y1_s2 = torch.mean((y1_pred_s2_exp - y1_true_flat)**2).item()
    mse_y2_s2 = torch.mean((y2_pred_s2_exp - y2_true_flat)**2).item()
    
    print(f"\nReconstruction MSE:")
    print(f"  y1 from s1: {mse_y1_s1:.4f}")
    print(f"  y2 from s1: {mse_y2_s1:.4f}")
    print(f"  y1 from s2: {mse_y1_s2:.4f}")
    print(f"  y2 from s2: {mse_y2_s2:.4f}")
    
    # Compute R² scores
    def r2_score(pred, true):
        ss_res = torch.sum((true - pred)**2)
        ss_tot = torch.sum((true - torch.mean(true))**2)
        return 1 - (ss_res / ss_tot)
    
    r2_y1_s1 = r2_score(y1_pred_s1_exp, y1_true_flat).item()
    r2_y2_s1 = r2_score(y2_pred_s1_exp, y2_true_flat).item()
    r2_y1_s2 = r2_score(y1_pred_s2_exp, y1_true_flat).item()
    r2_y2_s2 = r2_score(y2_pred_s2_exp, y2_true_flat).item()
    
    print(f"\nR² Scores:")
    print(f"  y1 from s1: {r2_y1_s1:.4f}")
    print(f"  y2 from s1: {r2_y2_s1:.4f}")
    print(f"  y1 from s2: {r2_y1_s2:.4f}")
    print(f"  y2 from s2: {r2_y2_s2:.4f}")
    
    # Analyze sparsity
    active_neurons_s1 = (hidden_s1 != 0).sum(dim=-1).float().mean().item()
    active_neurons_s2 = (hidden_s2 != 0).sum(dim=-1).float().mean().item()
    total_neurons = hidden_s1.shape[-1]
    num_dead_s1 = dead_mask_s1.sum().item()
    num_dead_s2 = dead_mask_s2.sum().item()
    
    print(f"\nSparsity:")
    print(f"  Active neurons (s1): {active_neurons_s1:.1f} / {total_neurons} ({100*active_neurons_s1/total_neurons:.2f}%)")
    print(f"  Active neurons (s2): {active_neurons_s2:.1f} / {total_neurons} ({100*active_neurons_s2/total_neurons:.2f}%)")
    print(f"  Dead neurons (s1): {num_dead_s1} / {total_neurons}")
    print(f"  Dead neurons (s2): {num_dead_s2} / {total_neurons}")
    
    # Find most active neurons (reshape to [N, d_hidden] for counting)
    hidden_s1_reshaped = hidden_s1.reshape(N, -1)  # [N, d_hidden]
    hidden_s2_reshaped = hidden_s2.reshape(N, -1)  # [N, d_hidden]
    
    neuron_counts_s1 = (hidden_s1_reshaped != 0).sum(dim=0)  # [d_hidden]
    neuron_counts_s2 = (hidden_s2_reshaped != 0).sum(dim=0)
    
    top_k = 10
    top_neurons_s1 = torch.topk(neuron_counts_s1, k=top_k)
    top_neurons_s2 = torch.topk(neuron_counts_s2, k=top_k)
    
    print(f"\nTop {top_k} most active neurons (s1):")
    for i, (idx, count) in enumerate(zip(top_neurons_s1.indices, top_neurons_s1.values)):
        print(f"  {i+1}. Neuron {idx.item()}: active in {count.item()}/{N} residues ({100*count.item()/N:.1f}%)")
    
    print(f"\nTop {top_k} most active neurons (s2):")
    for i, (idx, count) in enumerate(zip(top_neurons_s2.indices, top_neurons_s2.values)):
        print(f"  {i+1}. Neuron {idx.item()}: active in {count.item()}/{N} residues ({100*count.item()/N:.1f}%)")
    
    return {
        'protein_name': protein_name,
        'n_residues': N,
        'mse': {
            'y1_from_s1': mse_y1_s1,
            'y2_from_s1': mse_y2_s1,
            'y1_from_s2': mse_y1_s2,
            'y2_from_s2': mse_y2_s2,
        },
        'r2': {
            'y1_from_s1': r2_y1_s1,
            'y2_from_s1': r2_y2_s1,
            'y1_from_s2': r2_y1_s2,
            'y2_from_s2': r2_y2_s2,
        },
        'sparsity': {
            'active_s1': active_neurons_s1,
            'active_s2': active_neurons_s2,
            'total': total_neurons,
            'dead_s1': num_dead_s1,
            'dead_s2': num_dead_s2,
        },
        'top_neurons_s1': top_neurons_s1.indices.cpu().numpy().tolist(),
        'top_neurons_s2': top_neurons_s2.indices.cpu().numpy().tolist(),
    }


def visualize_activations(model, data, protein_name, output_dir, device='cuda'):
    """Create visualizations of transcoder behavior."""
    if not HAS_MATPLOTLIB:
        print(f"\nSkipping visualizations for {protein_name} (matplotlib not available)")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nCreating visualizations for {protein_name}...")
    
    # Load data
    s1 = torch.from_numpy(data['input_s']).float().to(device)
    s2 = torch.from_numpy(data['output_s']).float().to(device)
    y1_true = torch.from_numpy(data['input_z']).float().to(device)
    y2_true = torch.from_numpy(data['output_z']).float().to(device)
    
    N = s1.shape[1]
    
    with torch.no_grad():
        y1_pred_from_s1, y2_pred_from_s1, hidden_s1 = model(s1)
        y1_pred_from_s2, y2_pred_from_s2, hidden_s2 = model(s2)
    
    # Convert to numpy
    hidden_s1_np = hidden_s1[0].cpu().numpy()  # [N, d_hidden]
    hidden_s2_np = hidden_s2[0].cpu().numpy()
    
    # 1. Neuron activation heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Only show active neurons for clarity
    active_mask_s1 = (hidden_s1_np != 0).any(axis=0)
    active_mask_s2 = (hidden_s2_np != 0).any(axis=0)
    
    im1 = ax1.imshow(hidden_s1_np[:, active_mask_s1].T, aspect='auto', cmap='viridis')
    ax1.set_xlabel('Residue Position')
    ax1.set_ylabel('Active Neuron Index')
    ax1.set_title(f'Neuron Activations (input_s) - {protein_name}')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(hidden_s2_np[:, active_mask_s2].T, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Residue Position')
    ax2.set_ylabel('Active Neuron Index')
    ax2.set_title(f'Neuron Activations (output_s) - {protein_name}')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{protein_name}_neuron_activations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved neuron activation heatmap")
    
    # 2. Reconstruction quality
    y1_true_np = y1_true[0].cpu().numpy()  # [N, N, 128]
    y1_pred_np = y1_pred_from_s1[0].cpu().numpy()
    
    # Take first feature dimension for visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    im1 = ax1.imshow(y1_true_np[:, :, 0], cmap='RdBu_r', vmin=-3, vmax=3)
    ax1.set_title(f'True input_z (feature 0) - {protein_name}')
    ax1.set_xlabel('Residue j')
    ax1.set_ylabel('Residue i')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(y1_pred_np[:, :, 0], cmap='RdBu_r', vmin=-3, vmax=3)
    ax2.set_title(f'Predicted input_z (feature 0) - {protein_name}')
    ax2.set_xlabel('Residue j')
    ax2.set_ylabel('Residue i')
    plt.colorbar(im2, ax=ax2)
    
    error = np.abs(y1_true_np[:, :, 0] - y1_pred_np[:, :, 0])
    im3 = ax3.imshow(error, cmap='Reds', vmin=0, vmax=2)
    ax3.set_title(f'Absolute Error - {protein_name}')
    ax3.set_xlabel('Residue j')
    ax3.set_ylabel('Residue i')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{protein_name}_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved reconstruction comparison")
    
    # 3. Sparsity pattern
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sparsity_per_residue_s1 = (hidden_s1_np != 0).sum(axis=1)
    sparsity_per_residue_s2 = (hidden_s2_np != 0).sum(axis=1)
    
    ax.plot(sparsity_per_residue_s1, label='input_s', marker='o', markersize=3)
    ax.plot(sparsity_per_residue_s2, label='output_s', marker='s', markersize=3)
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Number of Active Neurons')
    ax.set_title(f'Sparsity per Residue - {protein_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{protein_name}_sparsity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved sparsity pattern")


def main():
    print("="*80)
    print("UNIVERSAL TRANSCODER ANALYSIS")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load model
    checkpoint_path = Path("universal_transcoder/checkpoints/universal_transcoder_final.pt")
    model, hp = load_trained_model(checkpoint_path, device)
    
    # Load training metrics
    metrics_path = Path("universal_transcoder/checkpoints/training_metrics.json")
    with open(metrics_path) as f:
        training_metrics = json.load(f)
    
    print(f"\nTraining summary:")
    print(f"  Steps: {training_metrics['final_metrics']['step']}")
    print(f"  Final loss: {training_metrics['final_metrics']['loss_total']:.2f}")
    print(f"  Training time: {training_metrics['training_time_seconds']:.1f}s")
    
    # Load activation data
    data_dir = Path("real_activations")
    batch_files = sorted(data_dir.glob("batch_*.npz"))
    
    print(f"\nFound {len(batch_files)} protein batch files")
    
    # Analyze each protein
    results = []
    for i, batch_file in enumerate(batch_files, 1):
        data = np.load(batch_file)
        protein_name = f"protein_{i}"
        
        # Analyze
        result = analyze_protein(model, data, protein_name, device)
        results.append(result)
        
        # Visualize
        visualize_activations(model, data, protein_name, "analysis_output", device)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nAnalyzed {len(results)} proteins:")
    for r in results:
        print(f"\n{r['protein_name']} ({r['n_residues']} residues):")
        print(f"  R² scores: y1={r['r2']['y1_from_s1']:.3f}, y2={r['r2']['y2_from_s1']:.3f}")
        print(f"  Sparsity: {r['sparsity']['active_s1']:.1f}/{r['sparsity']['total']} neurons")
    
    # Save results
    output_file = Path("analysis_output/analysis_results.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'hyperparameters': hp,
            'training_metrics': training_metrics['final_metrics'],
            'protein_results': results,
        }, f, indent=2)
    
    print(f"\n✓ Saved analysis results to {output_file}")
    print(f"✓ Visualizations saved to analysis_output/")
    print(f"\nAnalysis complete!")


if __name__ == '__main__':
    main()
