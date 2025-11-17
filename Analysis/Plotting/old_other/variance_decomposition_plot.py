#!/usr/bin/env python3
"""
Supplementary variance decomposition analysis visualization.
Shows signal-to-noise ratios and within-level variances for parameter sensitivity.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import date

cm = 1/2.54
FONT_SIZE = 8

FRIENDLY_COLORS = {
    'static': '#EE7733', 'random': '#0077BB', 'local (similar)': '#33BBEE',
    'local (opposite)': '#009988', 'bridge (similar)': '#CC3311', 'bridge (opposite)': '#EE3377',
    'wtf': '#BBBBBB', 'node2vec': '#44BB99'
}

def setup_style():
    plt.rcParams.update({
        'font.size': FONT_SIZE, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.figsize': (8.7*cm, 8*cm), 'axes.linewidth': 0.8,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.labelsize': FONT_SIZE-1, 'ytick.labelsize': FONT_SIZE-1,
        'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE
    })

def plot_variance_decomposition_panel(metrics_df, output_path):
    """
    Create 2x2 panel showing variance decomposition analysis
    Top row: Signal-to-noise ratios vs basin stability 
    Bottom row: Within-level variance vs basin stability
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(16*cm, 14*cm))
    fig.suptitle('Variance Decomposition Analysis', fontsize=FONT_SIZE, y=0.95)
    
    # Filter valid data
    valid_mask = (
        (metrics_df['cooperative_volume_percent'] > 0) &
        np.isfinite(metrics_df['cooperative_volume_percent']) &
        np.isfinite(metrics_df['stubbornness_signal_to_noise']) &
        np.isfinite(metrics_df['backfirer_signal_to_noise']) &
        np.isfinite(metrics_df['stubbornness_within_variance']) &
        np.isfinite(metrics_df['backfirer_within_variance'])
    )
    valid_data = metrics_df[valid_mask].copy()
    
    if len(valid_data) == 0:
        print("No valid data for variance decomposition plotting")
        return
    
    # Topology markers
    topology_markers = {'DPAH': 'x', 'cl': '+', 'Twitter': '*', 'FB': '.'}
    
    # Plot configurations: (y_column, title, ylabel)
    plots = [
        ('stubbornness_signal_to_noise', 'Stubbornness Signal-to-Noise', 'Signal-to-Noise Ratio'),
        ('backfirer_signal_to_noise', 'Backfirer Signal-to-Noise', 'Signal-to-Noise Ratio'),
        ('stubbornness_within_variance', 'Stubbornness Within-Level Variance', 'Within-Level Variance'),
        ('backfirer_within_variance', 'Backfirer Within-Level Variance', 'Within-Level Variance')
    ]
    
    for idx, (y_col, title, ylabel) in enumerate(plots):
        ax = axes[idx//2, idx%2]
        
        # Plot all points
        for i, (_, row) in enumerate(valid_data.iterrows()):
            color = FRIENDLY_COLORS.get(row['friendly_name'], 'black')
            marker = topology_markers.get(row['topology'], 'o')
            size = 25 if marker == '.' else 20
            
            ax.scatter(row['cooperative_volume_percent'], row[y_col], 
                      c=color, marker=marker, s=size, alpha=0.7, 
                      linewidth=0.3)
        
        ax.set_xlabel('Basin Stability (%)', fontsize=FONT_SIZE-2)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE-2)
        ax.set_title(title, fontsize=FONT_SIZE-1, pad=8)
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.tick_params(labelsize=FONT_SIZE-3)
    
    # Single legend for all subplots
    algo_elements = [Line2D([0], [0], marker='s', color=color, linestyle='None',
                           markersize=3, label=algo)
                    for algo, color in FRIENDLY_COLORS.items() 
                    if algo in valid_data['friendly_name'].values]
    
    fig.legend(handles=algo_elements, bbox_to_anchor=(0.5, 0.02), 
              loc='center', columnspacing=0.5, frameon=True, 
              fontsize=FONT_SIZE-3, ncol=6)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.9)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_sensitivity_comparison(metrics_df, output_path):
    """
    Compare Sobol sensitivity vs signal-to-noise ratio
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16*cm, 7*cm))
    fig.suptitle('Sensitivity Methods Comparison', fontsize=FONT_SIZE, y=0.95)
    
    # Filter valid data
    valid_mask = (
        np.isfinite(metrics_df['stubbornness_sensitivity']) &
        np.isfinite(metrics_df['backfirer_sensitivity']) &
        np.isfinite(metrics_df['stubbornness_signal_to_noise']) &
        np.isfinite(metrics_df['backfirer_signal_to_noise']) &
        (metrics_df['stubbornness_signal_to_noise'] > 0) &
        (metrics_df['backfirer_signal_to_noise'] > 0)
    )
    valid_data = metrics_df[valid_mask].copy()
    
    if len(valid_data) == 0:
        print("No valid data for sensitivity comparison")
        return
    
    topology_markers = {'DPAH': 'x', 'cl': '+', 'Twitter': '*', 'FB': '.'}
    
    # Stubbornness comparison
    ax = axes[0]
    for i, (_, row) in enumerate(valid_data.iterrows()):
        color = FRIENDLY_COLORS.get(row['friendly_name'], 'black')
        marker = topology_markers.get(row['topology'], 'o')
        size = 25 if marker == '.' else 20
        
        ax.scatter(row['stubbornness_sensitivity'], row['stubbornness_signal_to_noise'], 
                  c=color, marker=marker, s=size, alpha=0.7, linewidth=0.3)
    
    ax.set_xlabel('Sobol Sensitivity ($S_w$)', fontsize=FONT_SIZE-2)
    ax.set_ylabel('Signal-to-Noise Ratio', fontsize=FONT_SIZE-2)
    ax.set_title('Stubbornness', fontsize=FONT_SIZE-1)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.tick_params(labelsize=FONT_SIZE-3)
    
    # Backfirer comparison  
    ax = axes[1]
    for i, (_, row) in enumerate(valid_data.iterrows()):
        color = FRIENDLY_COLORS.get(row['friendly_name'], 'black')
        marker = topology_markers.get(row['topology'], 'o')
        size = 25 if marker == '.' else 20
        
        ax.scatter(row['backfirer_sensitivity'], row['backfirer_signal_to_noise'], 
                  c=color, marker=marker, s=size, alpha=0.7, linewidth=0.3)
    
    ax.set_xlabel('Sobol Sensitivity ($S_\\rho$)', fontsize=FONT_SIZE-2)
    ax.set_ylabel('Signal-to-Noise Ratio', fontsize=FONT_SIZE-2)
    ax.set_title('Backfirer', fontsize=FONT_SIZE-1)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.tick_params(labelsize=FONT_SIZE-3)
    
    # Single legend
    algo_elements = [Line2D([0], [0], marker='s', color=color, linestyle='None',
                           markersize=3, label=algo)
                    for algo, color in FRIENDLY_COLORS.items() 
                    if algo in valid_data['friendly_name'].values]
    
    fig.legend(handles=algo_elements, bbox_to_anchor=(0.5, 0.02), 
              loc='center', columnspacing=0.5, frameon=True, 
              fontsize=FONT_SIZE-3, ncol=6)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.9)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    # Look for heatmap metrics files
    stats_dir = "../../Output/Stats/stubborness_backfirer"
    if not os.path.exists(stats_dir):
        stats_dir = "../../Output/Stats"
    
    files = [f for f in os.listdir(stats_dir) if "heatmap_metrics_detailed" in f and f.endswith(".csv")]
    
    if not files:
        print("No heatmap_metrics_detailed files found")
        return
    
    # Auto-select the most recent file
    files.sort()
    latest_file = files[-1]
    print(f"Auto-selecting latest file: {latest_file}")
    
    # Load data
    data_path = os.path.join(stats_dir, latest_file)
    metrics_df = pd.read_csv(data_path)
    
    if metrics_df.empty:
        print("No data found in metrics file")
        return
    
    # Check if supplementary columns exist
    required_cols = ['stubbornness_signal_to_noise', 'backfirer_signal_to_noise', 
                     'stubbornness_within_variance', 'backfirer_within_variance']
    
    missing_cols = [col for col in required_cols if col not in metrics_df.columns]
    if missing_cols:
        print(f"Missing supplementary columns: {missing_cols}")
        print("Please regenerate data with updated heatmap_stats_multi.py")
        return
    
    # Create output directory
    output_dir = "../../Figs/Sensitivity"
    os.makedirs(output_dir, exist_ok=True)
    
    today = date.today().strftime("%Y%m%d")
    
    print("Generating variance decomposition plots...")
    
    # Variance decomposition panel plot
    decomp_output = f"{output_dir}/variance_decomposition_{today}.pdf"
    plot_variance_decomposition_panel(metrics_df, decomp_output)
    print(f"Variance decomposition plot saved: {decomp_output}")
    
    # Sensitivity comparison plot
    comparison_output = f"{output_dir}/sensitivity_comparison_{today}.pdf"
    plot_sensitivity_comparison(metrics_df, comparison_output)
    print(f"Sensitivity comparison plot saved: {comparison_output}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for param in ['stubbornness', 'backfirer']:
        snr_col = f'{param}_signal_to_noise'
        var_col = f'{param}_within_variance'
        sens_col = f'{param}_sensitivity'
        
        valid_snr = metrics_df[metrics_df[snr_col] > 0][snr_col]
        valid_var = metrics_df[metrics_df[var_col] > 0][var_col]
        valid_sens = metrics_df[metrics_df[sens_col] > 0][sens_col]
        
        print(f"{param.capitalize()}:")
        print(f"  SNR: mean={valid_snr.mean():.3f}, std={valid_snr.std():.3f}")
        print(f"  Within-var: mean={valid_var.mean():.4f}, std={valid_var.std():.4f}")  
        print(f"  Sobol sens: mean={valid_sens.mean():.3f}, std={valid_sens.std():.3f}")
    
    return metrics_df

if __name__ == "__main__":
    main()