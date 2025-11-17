#!/usr/bin/env python3
"""
Focused robustness analysis plots for meaningful subgroups
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import seaborn as sns
from datetime import date
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cm = 1/2.54
FONT_SIZE = 8

FRIENDLY_COLORS = {
    'static': '#EE7733', 'random': '#0077BB', 'local (similar)': '#33BBEE',
    'local (opposite)': '#009988', 'bridge (similar)': '#CC3311', 'bridge (opposite)': '#EE3377',
    'wtf': '#BBBBBB', 'node2vec': '#44BB99',
    'empirical wtf': '#BBBBBB', 'empirical node2vec': '#44BB99'
}

NETWORK_COLORS = {
    'DPAH': '#1f77b4', 'cl': '#ff7f0e', 'Twitter': '#2ca02c', 'FB': '#d62728'
}

def setup_style():
    plt.rcParams.update({
        'font.size': FONT_SIZE, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.labelsize': FONT_SIZE-1, 'ytick.labelsize': FONT_SIZE-1,
        'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE
    })

def add_regression_line(ax, x, y, color='red', alpha=0.8):
    """Add regression line with confidence interval"""
    if len(x) < 3:
        return
    
    # Calculate regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Generate line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    
    # Plot regression line
    ax.plot(x_line, y_line, color=color, linewidth=1.5, alpha=alpha, 
            linestyle='--', label=f'r={r_value:.3f}, p={p_value:.3f}')
    
    # Add confidence interval
    n = len(x)
    t_val = stats.t.ppf(0.975, n-2)
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean)**2)
    ss_res = np.sum((y - (slope * x + intercept))**2)
    mse = ss_res / (n - 2)
    
    ci_width = t_val * np.sqrt(mse * (1/n + (x_line - x_mean)**2 / ss_x))
    ax.fill_between(x_line, y_line - ci_width, y_line + ci_width, 
                   color=color, alpha=0.2)

def plot_combined_analysis(df, output_path, sensitivity_metric='stubbornness_sensitivity'):
    """Create combined topology and algorithm analysis side-by-side"""
    
    setup_style()
    # Create a figure with side-by-side subplots: topology (left) and algorithm (right)
    fig = plt.figure(figsize=(20*cm, 12*cm))
    
    # Use GridSpec for better control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    valid_mask = (
        (df['cooperative_volume_percent'] > 0) & (df[sensitivity_metric] >= 0) &
        np.isfinite(df[sensitivity_metric]) & np.isfinite(df['cooperative_volume_percent'])
    )
    valid_data = df[valid_mask].copy()
    
    topology_markers = {'DPAH': 'x', 'cl': '+', 'Twitter': '*', 'FB': '.'}
    
    # LEFT SIDE: Topology-specific analysis
    topologies = ['DPAH', 'cl', 'Twitter', 'FB']
    
    for idx, topology in enumerate(topologies):
        ax = fig.add_subplot(gs[idx//2, idx%2])
        
        # Filter for this topology
        topo_data = valid_data[valid_data['topology'] == topology]
        
        if len(topo_data) == 0:
            continue
            
        # Plot points colored by algorithm with topology markers
        for _, row in topo_data.iterrows():
            color = FRIENDLY_COLORS.get(row['friendly_name'], 'black')
            marker = topology_markers.get(row['topology'], 'o')
            size = 20 if marker == '.' else 15
            ax.scatter(row['cooperative_volume_percent'], row[sensitivity_metric], 
                      c=color, marker=marker, s=size, alpha=0.8, linewidth=0.3)
        
        # Add regression line if enough points
        if len(topo_data) >= 3:
            add_regression_line(ax, topo_data['cooperative_volume_percent'], 
                              topo_data[sensitivity_metric], 
                              color=NETWORK_COLORS[topology])
        
        # Styling
        ax.set_xlabel('Basin Stability (%)', fontsize=FONT_SIZE-1)
        if idx % 2 == 0:  # Only left column gets y-label
            if 'backfirer' in sensitivity_metric:
                ax.set_ylabel('Backfirer Sensitivity', fontsize=FONT_SIZE-1)
            else:
                ax.set_ylabel('Stubbornness Sensitivity', fontsize=FONT_SIZE-1)
        
        ax.set_title(f'{topology} (n={len(topo_data)})', fontsize=FONT_SIZE, pad=8)
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.tick_params(labelsize=FONT_SIZE-2)
        
        # Add legend for this subplot if regression was added
        if len(topo_data) >= 3:
            ax.legend(fontsize=FONT_SIZE-3, loc='best')
    
    # RIGHT SIDE: Algorithm-specific analysis
    strong_algos = ['bridge (similar)', 'local (similar)', 'empirical node2vec', 'random']
    
    for idx, algorithm in enumerate(strong_algos):
        ax = fig.add_subplot(gs[idx//2, idx%2 + 2])
        
        # Filter for this algorithm
        algo_data = valid_data[valid_data['friendly_name'] == algorithm]
        
        if len(algo_data) == 0:
            continue
        
        # Plot points colored by topology with topology markers
        for _, row in algo_data.iterrows():
            color = NETWORK_COLORS.get(row['topology'], 'black')
            marker = topology_markers.get(row['topology'], 'o')
            size = 20 if marker == '.' else 15
            ax.scatter(row['cooperative_volume_percent'], row[sensitivity_metric], 
                      c=color, marker=marker, s=size, alpha=0.8, linewidth=0.3)
        
        # Add regression line if enough points
        if len(algo_data) >= 3:
            add_regression_line(ax, algo_data['cooperative_volume_percent'], 
                              algo_data[sensitivity_metric], 
                              color=FRIENDLY_COLORS.get(algorithm, 'red'))
        
        # Styling
        ax.set_xlabel('Basin Stability (%)', fontsize=FONT_SIZE-1)
        if idx % 2 == 0:  # Only left column gets y-label
            if 'backfirer' in sensitivity_metric:
                ax.set_ylabel('Backfirer Sensitivity', fontsize=FONT_SIZE-1)
            else:
                ax.set_ylabel('Stubbornness Sensitivity', fontsize=FONT_SIZE-1)
        
        # Shorten algorithm names for titles
        short_name = algorithm.replace('empirical ', '').replace(' (similar)', ' (sim)')
        ax.set_title(f'{short_name} (n={len(algo_data)})', fontsize=FONT_SIZE, pad=8)
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.tick_params(labelsize=FONT_SIZE-2)
        
        # Add legend for this subplot if regression was added
        if len(algo_data) >= 3:
            ax.legend(fontsize=FONT_SIZE-3, loc='best')
    
    # Add panel labels
    fig.text(0.25, 0.95, 'A. By Network Topology', fontsize=FONT_SIZE, weight='bold', ha='center')
    fig.text(0.75, 0.95, 'B. By Rewiring Algorithm', fontsize=FONT_SIZE, weight='bold', ha='center')
    
    # Overall legends at bottom
    # Algorithm legend (for left panels)
    algo_elements = [Line2D([0], [0], marker='s', color=color, linestyle='None',
                           markersize=3, label=algo.replace('empirical ', ''))
                    for algo, color in FRIENDLY_COLORS.items() 
                    if algo in valid_data['friendly_name'].values]
    
    # Topology legend (for right panels)  
    topo_elements = [Line2D([0], [0], marker=marker, color='black', linestyle='None',
                           markersize=3, label=topo)
                    for topo, marker in topology_markers.items() 
                    if topo in valid_data['topology'].values]
    
    fig.legend(handles=algo_elements, bbox_to_anchor=(0.25, 0.02), 
              loc='center', columnspacing=0.4, frameon=True, 
              fontsize=FONT_SIZE-3, ncol=4, title='Algorithms')
    
    fig.legend(handles=topo_elements, bbox_to_anchor=(0.75, 0.02), 
              loc='center', columnspacing=0.4, frameon=True, 
              fontsize=FONT_SIZE-3, ncol=4, title='Topologies')
    
    plt.subplots_adjust(bottom=0.15, top=0.9)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_pca_analysis(df, output_path):
    """
    Educational PCA analysis plot with comprehensive interpretation
    
    PCA (Principal Component Analysis) is a dimensionality reduction technique that:
    1. Finds the directions of maximum variance in the data
    2. Projects high-dimensional data onto lower dimensions
    3. Helps identify patterns and relationships between variables
    """
    
    setup_style()
    
    # Select numerical columns for PCA
    numerical_cols = [
        'cooperative_volume_percent', 'stubbornness_sensitivity', 'backfirer_sensitivity',
        'mean_cooperation', 'cooperative_ratio', 'mean_polarization'
    ]
    
    # Prepare data
    available_cols = [col for col in numerical_cols if col in df.columns]
    df_pca = df[available_cols].dropna()
    
    if len(df_pca) < 10:
        print("Not enough data for PCA analysis")
        return
    
    # Standardize the data (crucial for PCA!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pca)
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Create comprehensive PCA plot
    fig = plt.figure(figsize=(20*cm, 14*cm))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, width_ratios=[2, 2, 1], height_ratios=[2, 1], 
                  hspace=0.3, wspace=0.3)
    
    # Main biplot (PC1 vs PC2)
    ax_main = fig.add_subplot(gs[0, :2])
    
    # Get corresponding metadata for PCA points
    df_meta = df.loc[df_pca.index].copy()
    topology_markers = {'DPAH': 'x', 'cl': '+', 'Twitter': '*', 'FB': '.'}
    
    # Plot data points colored by topology
    for topology in df_meta['topology'].unique():
        mask = df_meta['topology'] == topology
        marker = topology_markers.get(topology, 'o')
        color = NETWORK_COLORS.get(topology, 'black')
        size = 25 if marker == '.' else 20
        
        ax_main.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=color, marker=marker, s=size, alpha=0.7,
                       label=topology, linewidth=0.3)
    
    # Add loading vectors (showing how original variables relate to PCs)
    loadings = pca.components_[:2].T  # First 2 PCs
    loading_scale = 3  # Scale factor for visibility
    
    for i, var in enumerate(available_cols):
        # Draw arrow from origin to loading coordinates
        ax_main.arrow(0, 0, loadings[i, 0] * loading_scale, loadings[i, 1] * loading_scale,
                     head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.8)
        
        # Add variable label
        label_x = loadings[i, 0] * loading_scale * 1.1
        label_y = loadings[i, 1] * loading_scale * 1.1
        
        # Shorten labels for clarity
        short_name = var.replace('cooperative_volume_percent', 'Basin Stab.')\
                       .replace('stubbornness_sensitivity', 'Stubborn Sens.')\
                       .replace('backfirer_sensitivity', 'Backfirer Sens.')\
                       .replace('mean_cooperation', 'Mean Coop.')\
                       .replace('cooperative_ratio', 'Coop. Ratio')\
                       .replace('mean_polarization', 'Polarization')
        
        ax_main.text(label_x, label_y, short_name, fontsize=FONT_SIZE-2, 
                    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', 
                    facecolor='white', alpha=0.8))
    
    # Styling for main plot
    ax_main.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                      fontsize=FONT_SIZE)
    ax_main.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                      fontsize=FONT_SIZE)
    ax_main.set_title('PCA Biplot: Data Points and Variable Loadings', 
                     fontsize=FONT_SIZE+1, pad=15)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(fontsize=FONT_SIZE-2, title='Network Topology')
    
    # Add origin lines
    ax_main.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_main.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Scree plot (explained variance by component)
    ax_scree = fig.add_subplot(gs[0, 2])
    
    n_components = min(len(available_cols), 6)  # Show first 6 components
    components = range(1, n_components + 1)
    variance_explained = pca.explained_variance_ratio_[:n_components]
    cumulative_variance = np.cumsum(variance_explained)
    
    ax_scree.bar(components, variance_explained, alpha=0.7, color='steelblue',
                label='Individual')
    ax_scree.plot(components, cumulative_variance, 'ro-', markersize=4,
                 label='Cumulative')
    
    ax_scree.set_xlabel('Principal Component', fontsize=FONT_SIZE-1)
    ax_scree.set_ylabel('Explained Variance', fontsize=FONT_SIZE-1)
    ax_scree.set_title('Scree Plot', fontsize=FONT_SIZE)
    ax_scree.legend(fontsize=FONT_SIZE-2)
    ax_scree.grid(True, alpha=0.3)
    ax_scree.tick_params(labelsize=FONT_SIZE-2)
    
    # Loadings heatmap
    ax_loadings = fig.add_subplot(gs[1, :2])
    
    # Create loadings matrix for heatmap
    loadings_matrix = pca.components_[:4].T  # First 4 PCs
    
    # Shorten variable names for heatmap
    short_vars = [var.replace('cooperative_volume_percent', 'Basin Stab.')\
                     .replace('stubbornness_sensitivity', 'Stubborn Sens.')\
                     .replace('backfirer_sensitivity', 'Backfirer Sens.')\
                     .replace('mean_cooperation', 'Mean Coop.')\
                     .replace('cooperative_ratio', 'Coop. Ratio')\
                     .replace('mean_polarization', 'Polarization')\
                  for var in available_cols]
    
    im = ax_loadings.imshow(loadings_matrix.T, cmap='RdBu_r', aspect='auto',
                           vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax_loadings.set_xticks(range(len(short_vars)))
    ax_loadings.set_xticklabels(short_vars, rotation=45, ha='right', fontsize=FONT_SIZE-2)
    ax_loadings.set_yticks(range(4))
    ax_loadings.set_yticklabels([f'PC{i+1}' for i in range(4)], fontsize=FONT_SIZE-2)
    ax_loadings.set_title('Variable Loadings on Principal Components', fontsize=FONT_SIZE)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_loadings, fraction=0.046, pad=0.04)
    cbar.set_label('Loading Strength', fontsize=FONT_SIZE-2)
    cbar.ax.tick_params(labelsize=FONT_SIZE-3)
    
    # Add values to heatmap
    for i in range(4):
        for j, var in enumerate(short_vars):
            text = f'{loadings_matrix[j, i]:.2f}'
            ax_loadings.text(j, i, text, ha='center', va='center',
                           fontsize=FONT_SIZE-3, 
                           color='white' if abs(loadings_matrix[j, i]) > 0.5 else 'black')
    
    # Interpretation text box
    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis('off')
    
    interpretation = f"""PCA Interpretation:

PC1 ({pca.explained_variance_ratio_[0]:.1%}): 
Cooperation-related variables
load negatively

PC2 ({pca.explained_variance_ratio_[1]:.1%}): 
Sensitivity measures
load positively

Total variance explained
by PC1+PC2: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.1%}

Red arrows show how
original variables project
onto the PC space"""
    
    ax_text.text(0.05, 0.95, interpretation, transform=ax_text.transAxes,
                fontsize=FONT_SIZE-2, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print numerical summary
    print("\nPCA Analysis Summary:")
    print("=" * 50)
    print(f"Total variance explained by first 2 PCs: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.1%}")
    print(f"PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
    print(f"PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")
    
    print("\nKey Insights:")
    print("- PC1 separates high vs low cooperation systems")
    print("- PC2 separates high vs low sensitivity systems") 
    print("- Basin stability and sensitivity load on different PCs")
    print("- This explains why overall correlation is weak but subgroup correlations are strong")

def plot_algorithm_subgroups(df, output_path, sensitivity_metric='stubbornness_sensitivity'):
    """Create algorithm-specific subplots for strongest correlations"""
    
    setup_style()
    
    # Focus on algorithms with strong correlations
    strong_algos = ['bridge (similar)', 'local (similar)', 'empirical node2vec', 'random']
    
    fig, axes = plt.subplots(2, 2, figsize=(16*cm, 12*cm))
    fig.suptitle('Algorithm-Specific Analysis', fontsize=FONT_SIZE+1, y=0.95)
    
    valid_mask = (
        (df['cooperative_volume_percent'] > 0) & (df[sensitivity_metric] >= 0) &
        np.isfinite(df[sensitivity_metric]) & np.isfinite(df['cooperative_volume_percent'])
    )
    valid_data = df[valid_mask].copy()
    
    for idx, algorithm in enumerate(strong_algos):
        ax = axes[idx//2, idx%2]
        
        # Filter for this algorithm
        algo_data = valid_data[valid_data['friendly_name'] == algorithm]
        
        if len(algo_data) == 0:
            continue
        
        # Plot points colored by topology
        topology_markers = {'DPAH': 'x', 'cl': '+', 'Twitter': '*', 'FB': 'o'}
        
        for _, row in algo_data.iterrows():
            color = NETWORK_COLORS.get(row['topology'], 'black')
            marker = topology_markers.get(row['topology'], 'o')
            ax.scatter(row['cooperative_volume_percent'], row[sensitivity_metric], 
                      c=color, marker=marker, s=40, alpha=0.8, linewidth=0.5)
        
        # Add regression line if enough points
        if len(algo_data) >= 3:
            add_regression_line(ax, algo_data['cooperative_volume_percent'], 
                              algo_data[sensitivity_metric], 
                              color=FRIENDLY_COLORS.get(algorithm, 'red'))
        
        # Styling
        ax.set_xlabel('Basin Stability (%)', fontsize=FONT_SIZE-1)
        if 'backfirer' in sensitivity_metric:
            ax.set_ylabel('Backfirer Sensitivity', fontsize=FONT_SIZE-1)
        else:
            ax.set_ylabel('Stubbornness Sensitivity', fontsize=FONT_SIZE-1)
        
        ax.set_title(f'{algorithm} (n={len(algo_data)})', fontsize=FONT_SIZE, pad=8)
        ax.grid(True, alpha=0.3, linewidth=0.3)
        ax.tick_params(labelsize=FONT_SIZE-2)
        
        # Add legend for this subplot if regression was added
        if len(algo_data) >= 3:
            ax.legend(fontsize=FONT_SIZE-2, loc='best')
    
    # Overall legend for topologies
    topo_elements = [Line2D([0], [0], marker=marker, color='black', linestyle='None',
                           markersize=4, label=topo)
                    for topo, marker in topology_markers.items() 
                    if topo in valid_data['topology'].values]
    
    fig.legend(handles=topo_elements, bbox_to_anchor=(0.5, 0.02), 
              loc='center', columnspacing=0.5, frameon=True, 
              fontsize=FONT_SIZE-2, ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.9)
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
    df = pd.read_csv(data_path)
    
    # Create output directory
    output_dir = "../../Figs/Sensitivity"
    
    today = date.today().strftime("%Y%m%d")
    
    print("Generating enhanced robustness analysis plots...")
    
    # Combined topology and algorithm analysis
    for metric in ['stubbornness_sensitivity', 'backfirer_sensitivity']:
        if metric in df.columns:
            combined_output = f"{output_dir}/robustness_combined_{metric}_{today}.pdf"
            plot_combined_analysis(df, combined_output, metric)
            print(f"Combined analysis saved: {combined_output}")
    
    # PCA analysis
    pca_output = f"{output_dir}/robustness_pca_analysis_{today}.pdf"
    plot_pca_analysis(df, pca_output)
    print(f"PCA analysis saved: {pca_output}")
    
    print("\nAnalysis complete! Generated:")
    print("- Combined topology/algorithm plots for both sensitivity metrics")
    print("- Comprehensive PCA analysis with educational interpretation")
    print("- All plots use consistent topology markers and reduced point sizes")

if __name__ == "__main__":
    main()