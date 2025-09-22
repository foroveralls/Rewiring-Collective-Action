#!/usr/bin/env python3
"""
Publication-quality sensitivity analysis plot: Basin stability vs Sensitivity (σ)
Simple scatter plot using convergence_vs_cooperation.py styling.
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
    'wtf': '#BBBBBB', 'node2vec': '#44BB99',
    'empirical wtf': '#BBBBBB', 'empirical node2vec': '#44BB99'
}

def setup_style():
    plt.rcParams.update({
        'font.size': FONT_SIZE, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.figsize': (8.7*cm, 8*cm), 'axes.linewidth': 0.8,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.labelsize': FONT_SIZE-1, 'ytick.labelsize': FONT_SIZE-1,
        'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE
    })

def plot_sensitivity_analysis_panel(metrics_df, output_path, sensitivity_metric='backfirer_sensitivity'):
    """
    Create 2x2 panel comparing different axis transformations
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(16*cm, 14*cm))
    fig.suptitle('Transformation Comparison', fontsize=FONT_SIZE, y=0.95)
    
    # Filter valid data
    valid_mask = (
        (metrics_df['cooperative_volume_percent'] > 0) &
        (metrics_df[sensitivity_metric] >= 0) &
        np.isfinite(metrics_df[sensitivity_metric]) &
        np.isfinite(metrics_df['cooperative_volume_percent'])
    )
    valid_data = metrics_df[valid_mask].copy()
    
    if len(valid_data) == 0:
        print("No valid data for plotting")
        return
    
    # Topology markers
    topology_markers = {'DPAH': 'x', 'cl': '+', 'Twitter': '*', 'FB': '.'}
    
    # Define transformations and transform data directly
    transforms = [
        ('linear', 'linear', 'Linear-Linear'),
        ('log', 'linear', 'Log X - Linear Y'),
        ('linear', 'log', 'Linear X - Log Y'), 
        ('log', 'log', 'Log-Log')
    ]
    
    for idx, (x_transform, y_transform, title) in enumerate(transforms):
        ax = axes[idx//2, idx%2]
        
        # Transform data directly
        x_data = valid_data['cooperative_volume_percent'].copy()
        y_data = valid_data[sensitivity_metric].copy()
        
        if x_transform == 'log':
            x_data = np.log10(x_data)
        if y_transform == 'log':
            y_data = np.log10(y_data)
        
        # Plot all points with transformed data
        for i, (_, row) in enumerate(valid_data.iterrows()):
            color = FRIENDLY_COLORS.get(row['friendly_name'], 'black')
            marker = topology_markers.get(row['topology'], 'o')
            size = 25 if marker == '.' else 20
            
            ax.scatter(x_data.iloc[i], y_data.iloc[i], 
                      c=color, marker=marker, s=size, alpha=0.7, 
                      linewidth=0.3)
        
        # Set labels based on transformation
        if x_transform == 'log':
            ax.set_xlabel('log₁₀(Basin Stability %)', fontsize=FONT_SIZE-2)
        else:
            ax.set_xlabel('Basin Stability (%)', fontsize=FONT_SIZE-2)
            
        if y_transform == 'log':
            if 'backfirer' in sensitivity_metric:
                ax.set_ylabel('log₁₀(Backfirer Sens.)', fontsize=FONT_SIZE-2)
            else:
                ax.set_ylabel('log₁₀(Stubbornness Sens.)', fontsize=FONT_SIZE-2)
        else:
            if 'backfirer' in sensitivity_metric:
                ax.set_ylabel('Backfirer Sens. ($S_\\rho$)', fontsize=FONT_SIZE-2)
            else:
                ax.set_ylabel('Stubbornness Sens. ($S_w$)', fontsize=FONT_SIZE-2)
        
        ax.set_title(title, fontsize=FONT_SIZE-1, pad=8)
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.tick_params(labelsize=FONT_SIZE-3)
    
    # Single legend for all subplots
    algo_elements = [Line2D([0], [0], marker='s', color=color, linestyle='None',
                           markersize=3, label=algo)
                    for algo, color in FRIENDLY_COLORS.items() 
                    if algo in valid_data['friendly_name'].values]
    
    # Place legend outside the subplots
    fig.legend(handles=algo_elements, bbox_to_anchor=(0.5, 0.02), 
              loc='center', columnspacing=0.5, frameon=True, 
              fontsize=FONT_SIZE-3, ncol=6)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.9)
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
    
    # Create output directory
    output_dir = "../../Figs/Sensitivity"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for both robustness metrics
    today = date.today().strftime("%Y%m%d")
    
    print("Generating sensitivity analysis panel plots...")
    
    # Backfirer sensitivity panel plot
    if 'backfirer_sensitivity' in metrics_df.columns:
        backfirer_output = f"{output_dir}/sensitivity_panel_backfirer_{today}.pdf"
        plot_sensitivity_analysis_panel(metrics_df, backfirer_output, 'backfirer_sensitivity')
        print(f"Backfirer sensitivity panel plot saved: {backfirer_output}")
    
    # Stubbornness sensitivity panel plot
    if 'stubbornness_sensitivity' in metrics_df.columns:
        stubbornness_output = f"{output_dir}/sensitivity_panel_stubbornness_{today}.pdf"
        plot_sensitivity_analysis_panel(metrics_df, stubbornness_output, 'stubbornness_sensitivity')
        print(f"Stubbornness sensitivity panel plot saved: {stubbornness_output}")
    
    return metrics_df

if __name__ == "__main__":
    main()