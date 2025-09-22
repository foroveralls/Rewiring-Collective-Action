#!/usr/bin/env python3
"""
Compact visualization options for robustness analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import date

cm = 1/2.54
FONT_SIZE = 6

FRIENDLY_COLORS = {
    'static': '#EE7733', 'random': '#0077BB', 'local (similar)': '#33BBEE',
    'local (opposite)': '#009988', 'bridge (similar)': '#CC3311', 'bridge (opposite)': '#EE3377',
    'wtf': '#BBBBBB', 'node2vec': '#44BB99',
    'empirical wtf': '#BBBBBB', 'empirical node2vec': '#44BB99'
}

def setup_style():
    plt.rcParams.update({
        'font.size': FONT_SIZE, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.labelsize': FONT_SIZE-1, 'ytick.labelsize': FONT_SIZE-1,
        'axes.labelsize': FONT_SIZE-1, 'axes.titlesize': FONT_SIZE-1
    })

def create_correlation_heatmap(df, sensitivity_metric='backfirer_sensitivity'):
    """Combined correlation matrix with topology and algorithm in one view"""
    setup_style()
    
    # Filter valid data
    df_clean = df.dropna(subset=['cooperative_volume_percent', sensitivity_metric]).copy()
    
    # Get correlations for topology and algorithm
    groups = {'Topology': {}, 'Algorithm': {}}
    
    # By topology
    for topo in sorted(df_clean['topology'].unique()):
        subset = df_clean[df_clean['topology'] == topo]
        if len(subset) >= 3:
            r, p = stats.pearsonr(subset['cooperative_volume_percent'], subset[sensitivity_metric])
            groups['Topology'][topo] = {'r': r, 'p': p, 'n': len(subset)}
    
    # By algorithm  
    for algo in sorted(df_clean['friendly_name'].unique()):
        subset = df_clean[df_clean['friendly_name'] == algo]
        if len(subset) >= 3:
            r, p = stats.pearsonr(subset['cooperative_volume_percent'], subset[sensitivity_metric])
            groups['Algorithm'][algo] = {'r': r, 'p': p, 'n': len(subset)}
    
    # Create combined data for heatmap
    all_names = list(groups['Topology'].keys()) + list(groups['Algorithm'].keys())
    all_rs = ([groups['Topology'][k]['r'] for k in groups['Topology']] + 
             [groups['Algorithm'][k]['r'] for k in groups['Algorithm']])
    all_ps = ([groups['Topology'][k]['p'] for k in groups['Topology']] + 
             [groups['Algorithm'][k]['p'] for k in groups['Algorithm']])
    all_ns = ([groups['Topology'][k]['n'] for k in groups['Topology']] + 
             [groups['Algorithm'][k]['n'] for k in groups['Algorithm']])
    
    # Create single heatmap
    fig, ax = plt.subplots(figsize=(10*cm, 3*cm))
    
    # Reshape to single row
    corr_matrix = np.array(all_rs).reshape(1, -1)
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add text annotations
    for j, (r, p, n) in enumerate(zip(all_rs, all_ps, all_ns)):
        significance = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.1 else ''
        ax.text(j, 0, f'{r:.3f}{significance}\n(n={n})', 
                ha='center', va='center', fontsize=FONT_SIZE-2, fontweight='bold')
    
    # Customize labels
    clean_names = []
    for name in all_names:
        if name in ['DPAH', 'FB', 'Twitter', 'cl']:
            clean_names.append(name)  # Keep topology names as-is
        else:
            clean_names.append(name.replace('empirical ', '').replace(' (', '\n('))  # Clean algorithm names
    
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(clean_names, rotation=45, ha='right', fontsize=FONT_SIZE-2)
    ax.set_yticks([])
    
    # Add separator line between topology and algorithm groups
    n_topo = len(groups['Topology'])
    if n_topo > 0 and n_topo < len(all_names):
        ax.axvline(x=n_topo-0.5, color='black', linewidth=1, alpha=0.5)
    
    # Add group labels
    if n_topo > 0:
        ax.text(n_topo/2 - 0.5, -0.7, 'Topology', ha='center', fontsize=FONT_SIZE-1, fontweight='bold')
    if len(groups['Algorithm']) > 0:
        ax.text(n_topo + len(groups['Algorithm'])/2 - 0.5, -0.7, 'Algorithm', ha='center', fontsize=FONT_SIZE-1, fontweight='bold')
    
    # Title and colorbar
    metric_label = 'Backfirer' if 'backfirer' in sensitivity_metric else 'Stubbornness'
    ax.set_title(f'{metric_label} Sensitivity vs Basin Stability Correlations\n***p<0.001, **p<0.01, *p<0.05, †p<0.1', 
                 fontsize=FONT_SIZE, pad=15)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.05)
    cbar.set_label('Correlation (r)', fontsize=FONT_SIZE-1, labelpad=8)
    cbar.ax.tick_params(labelsize=FONT_SIZE-2)
    
    plt.tight_layout()
    
    # Create correlation table
    correlation_data = []
    for group_type, group_dict in groups.items():
        for name, stats in group_dict.items():
            significance = '***' if stats['p'] < 0.001 else '**' if stats['p'] < 0.01 else '*' if stats['p'] < 0.05 else '†' if stats['p'] < 0.1 else ''
            correlation_data.append({
                'Group_Type': group_type,
                'Name': name,
                'Correlation_r': stats['r'],
                'P_value': stats['p'],
                'N_samples': stats['n'],
                'Significance': significance
            })
    
    correlation_table = pd.DataFrame(correlation_data)
    
    return fig, correlation_table

def create_topology_algorithm_pairs_heatmap(df, sensitivity_metric='backfirer_sensitivity'):
    """Heatmap showing topology-algorithm pairs as individual data points"""
    setup_style()
    
    # Filter valid data
    df_clean = df.dropna(subset=['cooperative_volume_percent', sensitivity_metric]).copy()
    
    # Create topology-algorithm pair labels and get their values
    pairs = []
    basin_vals = []
    sens_vals = []
    
    for _, row in df_clean.iterrows():
        pair_name = f"{row['topology']}-{row['friendly_name']}"
        pairs.append(pair_name)
        basin_vals.append(row['cooperative_volume_percent'])
        sens_vals.append(row[sensitivity_metric])
    
    # Create matrix for visualization
    n_pairs = len(pairs)
    
    # Show both metrics as two rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14*cm, 4*cm), sharex=True)
    
    # Basin stability row
    basin_matrix = np.array(basin_vals).reshape(1, -1)
    im1 = ax1.imshow(basin_matrix, cmap='viridis', aspect='auto')
    
    # Add basin values as text
    for j, val in enumerate(basin_vals):
        ax1.text(j, 0, f'{val:.1f}', ha='center', va='center', 
                fontsize=FONT_SIZE-3, fontweight='bold', color='white')
    
    ax1.set_title('Basin Stability (%)', fontsize=FONT_SIZE-1, pad=5)
    ax1.set_yticks([])
    
    # Sensitivity row
    sens_matrix = np.array(sens_vals).reshape(1, -1)
    im2 = ax2.imshow(sens_matrix, cmap='plasma', aspect='auto')
    
    # Add sensitivity values as text
    for j, val in enumerate(sens_vals):
        ax2.text(j, 0, f'{val:.3f}', ha='center', va='center', 
                fontsize=FONT_SIZE-3, fontweight='bold', color='white')
    
    metric_label = 'Backfirer' if 'backfirer' in sensitivity_metric else 'Stubbornness'
    ax2.set_title(f'{metric_label} Sensitivity', fontsize=FONT_SIZE-1, pad=5)
    ax2.set_yticks([])
    
    # Clean pair names for x-axis
    clean_pairs = []
    for pair in pairs:
        # Replace algorithm names
        clean_pair = pair.replace('empirical ', '').replace('local (', 'loc(').replace('bridge (', 'brg(')
        clean_pairs.append(clean_pair)
    
    ax2.set_xticks(range(n_pairs))
    ax2.set_xticklabels(clean_pairs, rotation=90, ha='center', fontsize=FONT_SIZE-3)
    
    # Overall title
    fig.suptitle(f'Topology-Algorithm Pairs: Basin Stability vs {metric_label} Sensitivity', 
                 fontsize=FONT_SIZE, y=0.95)
    
    # Colorbars
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, aspect=15, pad=0.02)
    cbar1.ax.tick_params(labelsize=FONT_SIZE-3)
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=15, pad=0.02)
    cbar2.ax.tick_params(labelsize=FONT_SIZE-3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.2)
    return fig

def create_algorithm_faceted_plots(df, sensitivity_metric='backfirer_sensitivity'):
    """Faceted plots by algorithm (one subplot per algorithm)"""
    setup_style()
    
    # Filter valid data - much simpler filtering
    df_clean = df.dropna(subset=['cooperative_volume_percent', sensitivity_metric]).copy()
    
    algorithms = sorted(df_clean['friendly_name'].unique())
    topology_markers = {'DPAH': 'x', 'cl': '+', 'Twitter': '*', 'FB': '.'}
    
    # Create 2x4 grid for 8 algorithms
    fig, axes = plt.subplots(2, 4, figsize=(12*cm, 6*cm), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, algo in enumerate(algorithms):
        ax = axes[i]
        
        # Get all data for this algorithm across all topologies
        subset = df_clean[df_clean['friendly_name'] == algo]
        
        if len(subset) > 0:
            # Plot points colored by topology
            for topo in subset['topology'].unique():
                topo_data = subset[subset['topology'] == topo]
                marker = topology_markers.get(topo, 'o')
                size = 12 if marker == '.' else 8
                
                ax.scatter(topo_data['cooperative_volume_percent'], 
                          topo_data[sensitivity_metric],
                          c=FRIENDLY_COLORS.get(algo, 'black'), 
                          marker=marker, s=size, alpha=0.8, 
                          linewidth=0.4, label=topo if i == 0 else "")
        
        # Minimal styling
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.tick_params(labelsize=FONT_SIZE-2, pad=1)
        
        # Clean algorithm name for title
        clean_name = algo.replace('empirical ', '').replace(' (', '\n(')
        ax.set_title(f'{clean_name}\n(n={len(subset)})', 
                    fontsize=FONT_SIZE-1, pad=3)
        
        # Labels only on edges
        if i >= 4:  # Bottom row
            ax.set_xlabel('Basin Stability (%)', fontsize=FONT_SIZE-1, labelpad=2)
        if i % 4 == 0:  # Left column
            metric_label = 'Backfirer Sens.' if 'backfirer' in sensitivity_metric else 'Stubbornness Sens.'
            ax.set_ylabel(metric_label, fontsize=FONT_SIZE-1, labelpad=2)
    
    plt.suptitle(f'{"Backfirer" if "backfirer" in sensitivity_metric else "Stubbornness"} Sensitivity by Algorithm', 
                 fontsize=FONT_SIZE, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.4, wspace=0.3)
    return fig

def create_topology_faceted_plots(df, sensitivity_metric='backfirer_sensitivity'):
    """Faceted plots by topology (one subplot per topology)"""
    setup_style()
    
    # Filter valid data
    df_clean = df.dropna(subset=['cooperative_volume_percent', sensitivity_metric]).copy()
    
    topologies = sorted(df_clean['topology'].unique())
    topology_markers = {'DPAH': 'x', 'cl': '+', 'Twitter': '*', 'FB': '.'}
    
    # Create 2x2 grid for 4 topologies
    fig, axes = plt.subplots(2, 2, figsize=(8*cm, 6*cm), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, topo in enumerate(topologies):
        ax = axes[i]
        
        # Get all data for this topology across all algorithms
        subset = df_clean[df_clean['topology'] == topo]
        
        if len(subset) > 0:
            # Plot points colored by algorithm
            for algo in subset['friendly_name'].unique():
                algo_data = subset[subset['friendly_name'] == algo]
                color = FRIENDLY_COLORS.get(algo, 'black')
                marker = topology_markers.get(topo, 'o')
                size = 12 if marker == '.' else 8
                
                ax.scatter(algo_data['cooperative_volume_percent'], 
                          algo_data[sensitivity_metric],
                          c=color, marker=marker, s=size, alpha=0.8, 
                          linewidth=0.4, label=algo if i == 0 else "")
        
        # Minimal styling
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.tick_params(labelsize=FONT_SIZE-2, pad=1)
        
        ax.set_title(f'{topo}\n(n={len(subset)})', 
                    fontsize=FONT_SIZE-1, pad=3)
        
        # Labels only on edges
        if i >= 2:  # Bottom row
            ax.set_xlabel('Basin Stability (%)', fontsize=FONT_SIZE-1, labelpad=2)
        if i % 2 == 0:  # Left column
            metric_label = 'Backfirer Sens.' if 'backfirer' in sensitivity_metric else 'Stubbornness Sens.'
            ax.set_ylabel(metric_label, fontsize=FONT_SIZE-1, labelpad=2)
    
    plt.suptitle(f'{"Backfirer" if "backfirer" in sensitivity_metric else "Stubbornness"} Sensitivity by Topology', 
                 fontsize=FONT_SIZE, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.4, wspace=0.3)
    return fig

def main():
    """Generate faceted plots by algorithm and topology"""
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
    df = pd.read_csv(data_path)
    
    print(f"Available algorithms: {sorted(df['friendly_name'].unique())}")
    print(f"Available topologies: {sorted(df['topology'].unique())}")
    
    # Create output directory
    output_dir = "../../Figs/Sensitivity"
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    
    # Generate faceted plots for both metrics
    for metric in ['backfirer_sensitivity', 'stubbornness_sensitivity']:
        if metric in df.columns:
            metric_name = 'backfirer' if 'backfirer' in metric else 'stubbornness'
            
            print(f"\nGenerating plots for {metric_name}...")
            
            # Correlation heatmap (topology/algorithm groups)
            fig_heatmap, correlation_table = create_correlation_heatmap(df, metric)
            output_heatmap = f"{output_dir}/correlation_matrix_{metric_name}_{today}.pdf"
            fig_heatmap.savefig(output_heatmap, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved: {output_heatmap}")
            
            # Save correlation table as CSV
            table_output = f"{output_dir}/correlation_table_{metric_name}_{today}.csv"
            correlation_table.to_csv(table_output, index=False)
            print(f"Correlation table saved: {table_output}")
            
            plt.show()
            plt.close()
            
            # Topology-algorithm pairs heatmap  
            fig_pairs = create_topology_algorithm_pairs_heatmap(df, metric)
            output_pairs = f"{output_dir}/pairs_heatmap_{metric_name}_{today}.pdf"
            fig_pairs.savefig(output_pairs, dpi=300, bbox_inches='tight')
            print(f"Pairs heatmap saved: {output_pairs}")
            plt.show()
            plt.close()
            
            # Algorithm faceted plots
            fig_algo = create_algorithm_faceted_plots(df, metric)
            output_algo = f"{output_dir}/faceted_algorithm_{metric_name}_{today}.pdf"
            fig_algo.savefig(output_algo, dpi=300, bbox_inches='tight')
            print(f"Algorithm faceted plot saved: {output_algo}")
            plt.show()
            plt.close()
            
            # Topology faceted plots
            fig_topo = create_topology_faceted_plots(df, metric)
            output_topo = f"{output_dir}/faceted_topology_{metric_name}_{today}.pdf"
            fig_topo.savefig(output_topo, dpi=300, bbox_inches='tight')
            print(f"Topology faceted plot saved: {output_topo}")
            plt.show()
            plt.close()

if __name__ == "__main__":
    main()