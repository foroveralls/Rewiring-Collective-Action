#!/usr/bin/env python3
"""
Publication-quality Basin Stability vs Sensitivity analysis
Linear-linear plot with error bars, matching convergence_vs_cooperation.py style
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import date
from scipy import stats

cm = 1/2.54
FONT_SIZE = 8

FRIENDLY_COLORS = {
    'static': '#EE7733', 'random': '#0077BB', 'L-sim': '#33BBEE',
    'L-opp': '#009988', 'B-sim': '#CC3311', 'B-opp': '#EE3377',
    'wtf': '#BBBBBB', 'node2vec': '#44BB99'
}

FRIENDLY_NAMES = {
    'none_none': 'static', 'random_none': 'random', 'biased_same': 'L-sim',
    'biased_diff': 'L-opp', 'bridge_same': 'B-sim', 'bridge_diff': 'B-opp',
    'wtf_none': 'wtf', 'node2vec_none': 'node2vec'
}

def setup_style():
    plt.rcParams.update({
        'font.size': FONT_SIZE, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.figsize': (8.7*cm, 8*cm), 'axes.linewidth': 0.8,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.labelsize': FONT_SIZE-1, 'ytick.labelsize': FONT_SIZE-1,
        'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE
    })

def plot_basin_stability_sensitivity_linear(metrics_df, output_path):
    """
    Create linear-linear scatter plot of basin stability vs backfirer sensitivity with error bars
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.7*cm, 8*cm))
    
    # Filter valid data
    valid_mask = (
        (metrics_df['cooperative_volume_percent'] > 0) &
        (metrics_df['backfirer_sensitivity'] >= 0) &
        np.isfinite(metrics_df['backfirer_sensitivity']) &
        np.isfinite(metrics_df['cooperative_volume_percent']) &
        np.isfinite(metrics_df['backfirer_within_variance'])
    )
    valid_data = metrics_df[valid_mask].copy()
    
    if len(valid_data) == 0:
        print("No valid data for plotting")
        return
    
    # Transform friendly names to match convergence_vs_cooperation.py format
    # Create a reverse mapping for common variations
    name_transforms = {
        'empirical wtf': 'wtf',
        'bridge(similar)': 'B-sim',  
        'bridge (similar)': 'B-sim',
        'local(similar)': 'L-sim',
        'local (similar)': 'L-sim',
        'local(opposite)': 'L-opp',
        'local (opposite)': 'L-opp',
        'bridge(opposite)': 'B-opp',
        'bridge (opposite)': 'B-opp'
    }
    
    # Apply transformations if needed
    if 'friendly_name' in valid_data.columns:
        valid_data['friendly_name'] = valid_data['friendly_name'].replace(name_transforms)
    
    # Topology markers
    topology_markers = {'DPAH': 'x', 'cl': '+', 'Twitter': '^', 'FB': '.'}
    
    # Calculate error bars from within-level variance (standard deviation)
    valid_data['sensitivity_std'] = np.sqrt(valid_data['backfirer_within_variance'])
    
    # Proportionally scale error bars to fit within reasonable bounds while preserving relationships
    y_range = valid_data['backfirer_sensitivity'].max() - valid_data['backfirer_sensitivity'].min()
    max_reasonable_error = 0.08 * y_range  # Target max error as 8% of y-axis range
    max_raw_error = valid_data['sensitivity_std'].max()
    
    # Calculate scaling factor to preserve relative relationships
    if max_raw_error > 0:
        scale_factor = max_reasonable_error / max_raw_error
        valid_data['sensitivity_std_scaled'] = valid_data['sensitivity_std'] * scale_factor
    else:
        valid_data['sensitivity_std_scaled'] = valid_data['sensitivity_std']
    
    # Plot points with error bars
    for i, (_, row) in enumerate(valid_data.iterrows()):
        color = FRIENDLY_COLORS.get(row['friendly_name'], 'black')
        marker = topology_markers.get(row['topology'], 'o')
        size = 30 if marker == '.' else 20
        
        x = row['cooperative_volume_percent']
        y = row['backfirer_sensitivity']
        yerr = row['sensitivity_std_scaled']
        
        # Plot error bars first (behind the points)
        ax.errorbar(x, y, yerr=yerr, fmt='none', 
                   color=color, alpha=0.5, linewidth=0.6, capsize=1.4)
        
        # Plot the main point on top
        ax.scatter(x, y, c=color, marker=marker, s=size, alpha=0.7, 
                  edgecolors='black', linewidth=0.5, zorder=5)
    
    # Add linear best fit line
    x_data = valid_data['cooperative_volume_percent'].values
    y_data = valid_data['backfirer_sensitivity'].values
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    
    # Create line points across the data range
    x_min, x_max = x_data.min(), x_data.max()
    x_line = np.linspace(x_min, x_max, 100)
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, 'k--', alpha=0.7, linewidth=1.0, zorder=1)
    
    # Set labels
    ax.set_xlabel('Basin Stability (%)')
    ax.set_ylabel('Backfirer Sensitivity ($S_\\rho$)')
    
    # Grid and styling (match convergence_vs_cooperation.py exactly)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.tick_params(labelsize=FONT_SIZE-1)
    
    # Set axis limits - extend y-axis to accommodate error bars
    x_range = valid_data['cooperative_volume_percent'].max() - valid_data['cooperative_volume_percent'].min()
    y_range = valid_data['backfirer_sensitivity'].max() - valid_data['backfirer_sensitivity'].min()
    
    # Calculate max error bar extent for y-axis padding
    max_error_extent = valid_data['sensitivity_std_scaled'].max()
    y_padding = max(0.05 * y_range, max_error_extent * 1.2)  # Ensure error bars fit
    
    ax.set_xlim(valid_data['cooperative_volume_percent'].min() - 0.05 * x_range,
                valid_data['cooperative_volume_percent'].max() + 0.05 * x_range)
    ax.set_ylim(valid_data['backfirer_sensitivity'].min() - y_padding,
                valid_data['backfirer_sensitivity'].max() + y_padding)
    
    # Adjust subplot to make room for legends
    plt.subplots_adjust(top=0.81, bottom=0.24)
    
    # Algorithm legend at bottom (horizontal) first
    algo_elements = [Line2D([0], [0], marker='s', color=color, linestyle='None',
                           markersize=4, label=algo)
                    for algo, color in FRIENDLY_COLORS.items() 
                    if algo in valid_data['friendly_name'].values]
    algo_legend = ax.legend(handles=algo_elements, bbox_to_anchor=(0.5, -0.25), 
                           loc='center', columnspacing=0.8, frameon=True, fontsize=FONT_SIZE-2, 
                           ncol=4)
    
    # Topology legend at top using figlegend (more reliable)
    topo_elements = [Line2D([0], [0], marker=marker, color='black', linestyle='None', 
                           markersize=5, label=topo) 
                    for topo, marker in topology_markers.items()]
    fig.legend(handles=topo_elements, columnspacing=0.8, bbox_to_anchor=(0.53, 0.995), 
              loc='center', frameon=True, fontsize=FONT_SIZE-2, 
              ncol=4)
    
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
    print(f"Using latest file: {latest_file}")
    
    # Load data
    data_path = os.path.join(stats_dir, latest_file)
    metrics_df = pd.read_csv(data_path)
    
    if metrics_df.empty:
        print("No data found in metrics file")
        return
    
    # Create output directory
    output_dir = "../../Figs/Sensitivity"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plot
    today = date.today().strftime("%Y%m%d")
    output_path = f"{output_dir}/basin_stability_vs_sensitivity_linear_{today}.pdf"
    
    print("Generating basin stability vs sensitivity linear plot...")
    plot_basin_stability_sensitivity_linear(metrics_df, output_path)
    print(f"Plot saved: {output_path}")
    
    return metrics_df

if __name__ == "__main__":
    main()