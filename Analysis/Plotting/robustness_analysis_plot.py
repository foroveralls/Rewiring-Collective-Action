#!/usr/bin/env python3
"""
Publication-quality robustness analysis plot: Basin stability vs Robustness (1/σ)
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

def plot_robustness_analysis(metrics_df, output_path, robustness_metric='backfirer_robustness'):
    """
    Create simple robustness vs basin stability scatter plot
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.7*cm, 8*cm))
    
    # Filter valid data (exclude cases with no cooperation)
    valid_mask = (
        (metrics_df['cooperative_volume_percent'] > 0) &
        (metrics_df[robustness_metric] > 0) &
        np.isfinite(metrics_df[robustness_metric]) &
        np.isfinite(metrics_df['cooperative_volume_percent'])
    )
    valid_data = metrics_df[valid_mask].copy()
    
    if len(valid_data) == 0:
        print("No valid data for plotting")
        return
    
    # Topology markers
    topology_markers = {'DPAH': 'x', 'cl': '+', 'Twitter': '*', 'FB': '.'}
    
    # Plot all points
    for idx, row in valid_data.iterrows():
        color = FRIENDLY_COLORS.get(row['friendly_name'], 'black')
        marker = topology_markers.get(row['topology'], 'o')
        size = 40 if marker == '.' else 30
        
        ax.scatter(row['cooperative_volume_percent'], row[robustness_metric], 
                  c=color, marker=marker, s=size, alpha=0.7, 
                  edgecolors='black', linewidth=0.5)
    
    # Styling
    ax.set_xlabel('Basin Stability (%)')
    robustness_label = 'Backfirer Robustness' if 'backfirer' in robustness_metric else 'Stubbornness Robustness'
    ax.set_ylabel(f'{robustness_label} (1/σ)')
    ax.grid(True, alpha=0.3, linewidth=0.4)
    
    # Set axis limits with some padding
    x_min, x_max = valid_data['cooperative_volume_percent'].min(), valid_data['cooperative_volume_percent'].max()
    y_min, y_max = valid_data[robustness_metric].min(), valid_data[robustness_metric].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    ax.set_xlim(max(0, x_min - 0.05*x_range), x_max + 0.05*x_range)
    ax.set_ylim(max(0, y_min - 0.05*y_range), y_max + 0.05*y_range)
    
    # Adjust subplot to make room for legends
    plt.subplots_adjust(top=0.81, bottom=0.24)
    
    # Algorithm legend at bottom (horizontal)
    algo_elements = [Line2D([0], [0], marker='s', color=color, linestyle='None',
                           markersize=4, label=algo)
                    for algo, color in FRIENDLY_COLORS.items() 
                    if algo in valid_data['friendly_name'].values]
    algo_legend = ax.legend(handles=algo_elements, bbox_to_anchor=(0.5, -0.25), 
                           loc='center', columnspacing=0.8, frameon=True, 
                           fontsize=FONT_SIZE-2, ncol=4)
    
    # Topology legend at top
    topo_elements = [Line2D([0], [0], marker=marker, color='black', linestyle='None', 
                           markersize=5, label=topo) 
                    for topo, marker in topology_markers.items()
                    if topo in valid_data['topology'].values]
    fig.legend(handles=topo_elements, columnspacing=0.8, bbox_to_anchor=(0.53, 0.995), 
              loc='center', frameon=True, fontsize=FONT_SIZE-2, ncol=4)
    
    plt.tight_layout()
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
    output_dir = "../../Figs/Robustness"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for both robustness metrics
    today = date.today().strftime("%Y%m%d")
    
    print("Generating robustness analysis plots...")
    
    # Backfirer robustness plot
    if 'backfirer_robustness' in metrics_df.columns:
        backfirer_output = f"{output_dir}/robustness_basin_stability_backfirer_{today}.pdf"
        plot_robustness_analysis(metrics_df, backfirer_output, 'backfirer_robustness')
        print(f"Backfirer robustness plot saved: {backfirer_output}")
    
    # Stubbornness robustness plot
    if 'stubbornness_robustness' in metrics_df.columns:
        stubbornness_output = f"{output_dir}/robustness_basin_stability_stubbornness_{today}.pdf"
        plot_robustness_analysis(metrics_df, stubbornness_output, 'stubbornness_robustness')
        print(f"Stubbornness robustness plot saved: {stubbornness_output}")
    
    return metrics_df

if __name__ == "__main__":
    main()