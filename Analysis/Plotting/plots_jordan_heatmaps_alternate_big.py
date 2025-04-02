#!/usr/bin/env python3
"""
Modified heatmap visualization script with horizontal layout.
Creates a comprehensive grid with topologies on rows and scenarios on columns.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib import transforms
from datetime import date

# ====================== CONFIGURATION ======================
# Constants
FONT_SIZE = 14
FRIENDLY_COLORS = {
    'static': '#EE7733',      # Orange
    'random': '#0077BB',      # Blue
    'local (similar)': '#33BBEE',    # Cyan
    'local (opposite)': '#009988',   # Teal
    'bridge (similar)': '#CC3311',   # Red
    'bridge (opposite)': '#EE3377',  # Magenta
    'wtf': '#BBBBBB',         # Grey
    'node2vec': '#44BB99'     # Blue-green
}

FRIENDLY_NAMES = {
    'none_none': 'static',
    'random_none': 'random',
    'biased_same': 'local (similar)',
    'biased_diff': 'local (opposite)',
    'bridge_same': 'bridge (similar)',
    'bridge_diff': 'bridge (opposite)',
    'wtf_none': 'wtf',
    'node2vec_none': 'node2vec'
}

# Excluded scenarios and topologies
EXCLUDED_SCENARIOS = ['static']
EXCLUDED_TOPOLOGIES = ['cl', 'DPAH']  # Add topologies to exclude here, e.g., ['Twitter']

# ====================== FUNCTIONS ======================

def setup_plotting_style():
    """Updated plotting style configuration"""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.dpi': 300
    })
    sns.set_theme(font_scale=FONT_SIZE/12)
    sns.set(style="ticks")
    sns.set(rc={
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        "axes.grid": True,
        "grid.color": 'white',
        'grid.linestyle': 'solid', 
        "axes.edgecolor": "black",
        "patch.edgecolor": "black",
        "patch.linewidth": 0.5,
        "axes.spines.bottom": True,
        "grid.alpha": 0.8,
        "xtick.bottom": True,
        "ytick.left": True
    })

def get_data_file():
    """Get the data file path from user input."""
    file_list = [f for f in os.listdir("../../Output") 
                 if f.endswith(".csv") and "heatmap" in f]
    
    if not file_list:
        print("No suitable files found in the Output directory.")
        exit()
    
    # Print file list with indices
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")
    
    file_index = int(input("Enter the index of the file you want to plot: "))
    return os.path.join("../../Output", file_list[file_index])

def prepare_dataframe(df):
    """Prepare the DataFrame for plotting."""
    # Adjust the "rewiring" column for specific modes
    df.loc[df['mode'].isin(['wtf', 'node2vec']), 'rewiring'] = 'empirical'
    
    # Replace 'nan' with 'none' in 'rewiring' and 'mode' columns
    df['rewiring'] = df['rewiring'].fillna('none')
    df['mode'] = df['mode'].fillna('none')
    
    # Create scenario column
    df['scenario'] = df['rewiring'] + ' ' + df['mode']
    
    return df

def get_friendly_name(scenario):
    """Convert scenario name to friendly name."""
    parts = scenario.split()
    key = f"{parts[1]}_{parts[0]}" if len(parts) > 1 else f"{parts[0]}_none"
    return FRIENDLY_NAMES.get(key, scenario)

def create_comprehensive_heatmap_grid(df, value_columns, column_labels):
    """
    Create a comprehensive grid of heatmaps showing all scenarios across all topologies.
    Topologies are on rows, scenarios are on columns (horizontal layout).
    
    Parameters:
    df - DataFrame with heatmap data
    value_columns - List of metrics to display (e.g., 'state', 'state_std')
    column_labels - Dictionary mapping column names to display labels
    """
    # Get unique scenarios and topologies
    scenarios = df['scenario'].unique()
    topologies = df['topology'].unique()
    
    # Filter out excluded topologies
    topologies = [t for t in topologies if t not in EXCLUDED_TOPOLOGIES]
    
    # Convert to friendly names and filter out excluded scenarios
    friendly_scenarios = {}
    for scenario in scenarios:
        friendly_name = get_friendly_name(scenario)
        if friendly_name not in EXCLUDED_SCENARIOS:
            friendly_scenarios[scenario] = friendly_name
    
    # Sort scenarios by friendly name
    sorted_scenarios = sorted(friendly_scenarios.keys(), key=lambda s: friendly_scenarios[s])
    
    # Number of value columns (metrics) to display
    n_metrics = len(value_columns)
    
    # Swap rows and columns: topologies as rows, scenarios as columns
    n_rows = len(topologies)
    n_cols = len(sorted_scenarios) * n_metrics
    
    # Create figure with more compact aspect ratio
    fig_width = min(18, n_cols * 1.4)  # Reduced width factor and maximum
    fig_height = max(6, n_rows * 2)  # Ensure minimum height
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Use GridSpec for more control over subplot placement
    gs = fig.add_gridspec(n_rows, n_cols, 
                         hspace=0.3,  # Reduced vertical spacing
                         wspace=0.03,  # Tighter horizontal spacing
                         left=0.07,
                         right=0.97,
                         top=0.9,
                         bottom=0.12)
    
    # Define colormaps
    coop_cmap = sns.diverging_palette(20, 220, as_cmap=True, center="light")
    polar_cmap = sns.color_palette("Reds", as_cmap=True)
    
    # Loop through topologies (rows) and scenarios (columns)
    for row, topology in enumerate(topologies):
        # Add topology name on the left
        if row == 0:
            # Add column labels for the scenarios at the top - grouped by pair of metrics
            for s, scenario in enumerate(sorted_scenarios):
                friendly_scenario = friendly_scenarios[scenario]
                scenario_color = FRIENDLY_COLORS.get(friendly_scenario, 'black')
                col_idx = s * n_metrics
                fig.text(0.07 + (col_idx + n_metrics/2) * 0.9/n_cols, 0.94, 
                         friendly_scenario, 
                         ha='center', va='bottom', 
                         fontsize=FONT_SIZE-1,  # Smaller font
                         fontweight='bold',
                         color=scenario_color)
        
        # Add topology label on the left - more compact
        fig.text(0.02, 0.12 + (n_rows - row - 0.5) * 0.78/n_rows, 
                 topology.upper(),
                 ha='left', va='center',
                 fontsize=FONT_SIZE-1,  # Smaller font
                 fontweight='bold',
                 rotation=90)
        
        for s, scenario in enumerate(sorted_scenarios):
            friendly_scenario = friendly_scenarios[scenario]
            
            # Get topology data for this scenario
            scenario_topo_data = df[(df['scenario'] == scenario) & (df['topology'] == topology)]
            
            if scenario_topo_data.empty:
                continue
                
            for m, metric in enumerate(value_columns):
                # Calculate column index
                col = s * n_metrics + m
                
                # Create axis
                ax = fig.add_subplot(gs[row, col])
                
                # Create pivot table
                try:
                    heatmap_data = scenario_topo_data.pivot_table(
                        index='polarisingNode_f',
                        columns='stubbornness',
                        values=metric,
                        aggfunc='mean'
                    )
                    
                    # Reverse the y-axis order for consistent display
                    heatmap_data = heatmap_data.iloc[::-1]
                    
                    # Determine colormap and limits based on metric
                    if metric == 'state':
                        cmap = coop_cmap
                        vmin, vmax = -1, 1
                        center = 0
                        cbar_label = 'Cooperation'
                    else:
                        cmap = polar_cmap
                        vmin, vmax = 0, 1
                        center = None
                        cbar_label = 'Polarization'
                    
                    # Create heatmap
                    hm = sns.heatmap(heatmap_data,
                                   ax=ax,
                                   cmap=cmap,
                                   center=center,
                                   vmin=vmin,
                                   vmax=vmax,
                                   cbar=col % n_metrics == n_metrics - 1,  # Only show colorbar for last metric
                                   cbar_kws={'label': cbar_label} if col % n_metrics == n_metrics - 1 else {})
                    
                    # Format tick labels - more compact
                    if row == n_rows - 1:  # Only show x labels on bottom row
                        ax.set_xticklabels([f'{x:.1f}' for x in heatmap_data.columns], rotation=45, fontsize=FONT_SIZE-4)
                    else:
                        ax.set_xticklabels([])
                    
                    if col % n_metrics == 0:  # Only show y labels on leftmost column of each scenario
                        ax.set_yticklabels([f'{y:.1f}' for y in heatmap_data.index], rotation=0, fontsize=FONT_SIZE-4)
                    else:
                        ax.set_yticklabels([])
                    
                    # Add metric label
                    if row == 0:
                        if metric == 'state':
                            ax.set_title(r'$\langle x \rangle$', fontsize=FONT_SIZE-2)
                        else:
                            ax.set_title(r'$\sigma(x)$', fontsize=FONT_SIZE-2)
                    
                    # Add axis labels more efficiently
                    # Only add y-label to the first column
                    if col == 0:
                        ax.set_ylabel('Polarizing Node\nFraction', fontsize=FONT_SIZE-3)
                    else:
                        ax.set_ylabel('')
                    
                    # Add x-axis label only for the bottom row and central columns
                    if row == n_rows - 1 and col % (2 * n_metrics) == n_metrics - 1:
                        ax.set_xlabel('Stubbornness', fontsize=FONT_SIZE-3)
                    else:
                        ax.set_xlabel('')
                
                except Exception as e:
                    print(f"Error plotting {scenario} for {topology}, {metric}: {e}")
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
    
    # Add more compact super title
    fig.suptitle('Network Dynamics: Topologies × Rewiring Algorithms',
                fontsize=FONT_SIZE+2,
                fontweight='bold',
                y=0.98)
    
    return fig

def save_figure(fig, output_name='comprehensive_heatmap_horizontal'):
    """Save the figure to file with LaTeX-friendly settings."""
    save_path = f'../../Figs/Heatmaps/{output_name}_{date.today()}.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save with specific PDF settings for better text rendering
    fig.savefig(save_path, 
                bbox_inches='tight', 
                dpi=300,
                format='pdf',
                backend='pdf',
                transparent=True)
    
    # Also save as PNG for quick viewing
    png_path = save_path.replace('.pdf', '.png')
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    
    print(f"Saved figure to {save_path} and {png_path}")
    plt.show()

def main():
    """Main execution function."""
    # Setup
    setup_plotting_style()
    
    # Load and prepare data
    data_path = get_data_file()
    df = pd.read_csv(data_path)
    df = prepare_dataframe(df)
    
    # Define plotting parameters with friendly column names
    value_columns = ['state', 'state_std']
    column_labels = {
        'state': 'avg(x)',
        'state_std': 'std(x)'
    }
    
    # Create comprehensive grid
    fig = create_comprehensive_heatmap_grid(df, value_columns, column_labels)
    
    # Save figure
    save_figure(fig, 'comprehensive_heatmap_horizontal')

if __name__ == "__main__":
    main()