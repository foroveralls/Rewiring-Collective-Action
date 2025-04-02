#!/usr/bin/env python3
"""
Visualization script for cooperativity data from parameter sweeps.
Creates a grid of heatmaps showing individual run distribution across topologies and scenarios.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from datetime import date

# ====================== CONFIGURATION ======================
# Output directory for plots
OUTPUT_DIR = "../../Figs/Heatmaps"

# Plot settings
BASE_FONT_SIZE = 8
cm = 1/2.54

# Define font sizes for different elements
TITLE_FONT_SIZE = BASE_FONT_SIZE + 1
AXIS_LABEL_FONT_SIZE = BASE_FONT_SIZE
TICK_FONT_SIZE = BASE_FONT_SIZE - 2
LEGEND_FONT_SIZE = BASE_FONT_SIZE - 1
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE

# Scenario names and colors mapping
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

# ====================== FUNCTIONS ======================

def setup_plotting():
    """Configure plotting style."""
    plt.rcParams.update({
        'font.size': BASE_FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.dpi': 300,
        'figure.figsize': (17.8*cm, 8.9*cm),
        'axes.labelsize': AXIS_LABEL_FONT_SIZE,
        'axes.titlesize': TITLE_FONT_SIZE,
        'xtick.labelsize': TICK_FONT_SIZE,
        'ytick.labelsize': TICK_FONT_SIZE,
        'legend.fontsize': LEGEND_FONT_SIZE,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'axes.linewidth': 0.8,
    })
    sns.set_style("white")

def get_friendly_name(mode, rewiring):
    """Get user-friendly algorithm name."""
    if mode is None:
        return "Unknown"
    
    if rewiring is None or pd.isna(rewiring) or rewiring == 'None':
        rewiring = "none"
    
    mode = str(mode).lower()
    rewiring = str(rewiring).lower()
    
    key = f"{mode}_{rewiring}"
    if mode in ["none", "random", "wtf", "node2vec"]:
        key = f"{mode}_none"
    
    return FRIENDLY_NAMES.get(key, f"{mode} ({rewiring})")

def get_data_file():
    """Get the data file path from user input."""
    file_list = [f for f in os.listdir("../../Output") 
                 if f.endswith(".csv") and "heatmap_sweep" in f]
    
    if not file_list:
        print("No suitable files found in the Output directory.")
        sys.exit(1)
    
    # Print file list with indices
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")
    
    file_index = int(input("Enter the index of the file you want to plot: "))
    return os.path.join("../../Output", file_list[file_index])

def preprocess_data(df, param_name, max_param_value=0.06):
    """Preprocess the data for visualization."""
    # Filter to parameter values <= max_param_value
    if param_name in df.columns:
        df = df[df[param_name] <= max_param_value]
    
    # Handle NaN values and standardize column names
    df['rewiring'] = df['rewiring'].fillna('none')
    df['mode'] = df['mode'].fillna('none')
    
    # Create scenario key for mapping
    df['scenario_key'] = df.apply(lambda row: f"{row['mode']}_{row['rewiring']}", axis=1)
    
    # Add friendly name column
    df['friendly_name'] = df.apply(lambda row: get_friendly_name(row['mode'], row['rewiring']), axis=1)
    
    # Round parameter values to 2 decimal places
    df[f'{param_name}_rounded'] = df[param_name].round(2)
    
    return df, max_param_value

def create_state_distribution_grid(df, param_name, max_param_value):
    """
    Create a grid of 2D heatmaps showing distribution of individual final states.
    
    Parameters:
    df - Preprocessed DataFrame with state data
    param_name - Name of the parameter being swept
    max_param_value - Maximum parameter value to display
    """
    # Get unique topologies and scenarios
    all_topologies = sorted(df['topology'].unique())
    
    # Group by mode and rewiring to get unique scenarios
    scenarios = df.groupby(['mode', 'rewiring']).size().reset_index()
    all_scenarios = []
    
    for _, row in scenarios.iterrows():
        mode, rewiring = row['mode'], row['rewiring']
        friendly_name = get_friendly_name(mode, rewiring)
        all_scenarios.append((mode, rewiring, friendly_name))
    
    # Sort scenarios based on FRIENDLY_NAMES order
    def get_scenario_index(scenario):
        key = f"{scenario[0]}_{scenario[1]}"
        for i, name in enumerate(FRIENDLY_NAMES.keys()):
            if name == key:
                return i
        return 999  # Place unknown scenarios at the end
    
    all_scenarios.sort(key=get_scenario_index)
    
    # Grid layout
    n_rows = len(all_topologies)
    n_cols = len(all_scenarios)
    
    fig = plt.figure(figsize=(17.8*cm, 8.9*cm))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.15, hspace=0.3)
    
    # Common x-axis settings for all plots
    x_ticks = np.round(np.arange(0, max_param_value+0.001, 0.02), 2)
    x_ticklabels = [f"{x:.2f}" for x in x_ticks]
    
    # Common y-axis settings
    y_bins = np.linspace(-1, 1, 21)  # 20 bins between -1 and 1
    
    # Define custom colormap for density
    cmap = plt.cm.viridis
    
    # Create a storage for all histogram data to normalize globally
    all_hist_data = []
    
    # First pass: calculate histograms and find global max for normalization
    for topology in all_topologies:
        for mode, rewiring, _ in all_scenarios:
            # Filter data for this combination
            cell_data = df[(df['topology'] == topology) & 
                          (df['mode'] == mode) & 
                          (df['rewiring'] == rewiring)]
            
            if not cell_data.empty:
                # Create 2D histogram of states
                param_vals = cell_data[f'{param_name}_rounded'].values
                state_vals = cell_data['state'].values
                
                # Calculate counts for each x,y bin
                H, xedges, yedges = np.histogram2d(
                    param_vals, state_vals, 
                    bins=[np.unique(np.round(param_vals, 2)), y_bins]
                )
                
                all_hist_data.append((H, xedges, yedges))
    
    # Find the global maximum count for normalization
    global_max = max([h.max() for h, _, _ in all_hist_data if len(h) > 0]) if all_hist_data else 1
    
    # Process each cell in grid
    hist_idx = 0
    for row_idx, topology in enumerate(all_topologies):
        for col_idx, (mode, rewiring, friendly_name) in enumerate(all_scenarios):
            # Create subplot
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Filter data for this combination
            cell_data = df[(df['topology'] == topology) & 
                          (df['mode'] == mode) & 
                          (df['rewiring'] == rewiring)]
            
            if cell_data.empty:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', 
                       fontsize=ANNOTATION_FONT_SIZE)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Get histogram data from first pass
                H, xedges, yedges = all_hist_data[hist_idx]
                hist_idx += 1
                
                # Normalize counts (relative to global max)
                H_norm = H / global_max
                
                # Plot the heatmap with log normalization for better visibility of low counts
                im = ax.pcolormesh(
                    xedges, yedges, H.T,  # Transpose to put states on y-axis
                    cmap=cmap, 
                    norm=LogNorm(vmin=0.1, vmax=global_max),  # Log scale helps show low counts
                    shading='auto'
                )
                
                # Set axis limits
                ax.set_xlim(0, max_param_value)
                ax.set_ylim(-1, 1)
                
                # Format axes
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_ticklabels, rotation=45, ha='right', fontsize=TICK_FONT_SIZE)
                
                # Add a grid for visual clarity
                ax.grid(False)
            
            # Add labels
            if col_idx == 0:  # First column gets topology label
                ax.text(-0.5, 0.5, topology.upper(), transform=ax.transAxes, 
                       rotation=90, fontsize=AXIS_LABEL_FONT_SIZE+1, 
                       fontweight='bold', va='center', ha='center')
                # Add y-axis ticks and label for first column
                ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax.set_ylabel('State', fontsize=AXIS_LABEL_FONT_SIZE)
            else:
                # Hide y-ticks for other columns
                ax.set_yticks([])
            
            if row_idx == 0:  # First row gets scenario title
                title_color = FRIENDLY_COLORS.get(friendly_name, 'black')
                ax.set_title(friendly_name, fontsize=TITLE_FONT_SIZE, 
                           color=title_color, fontweight='bold')
            
            if row_idx == n_rows - 1:  # Last row gets x-axis label
                ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=AXIS_LABEL_FONT_SIZE)
    
    # Add a colorbar to the right of the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Count', fontsize=AXIS_LABEL_FONT_SIZE)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    sweep_id = os.path.basename(df.name).split('_')[-1].split('.')[0] if hasattr(df, 'name') else today
    save_path = f'{OUTPUT_DIR}/state_heatmap_{param_name}_{sweep_id}'
    
    for ext in ['pdf', 'png']:
        plt.savefig(f"{save_path}.{ext}", bbox_inches='tight', dpi=300)
    
    plt.show()
    print(f"Saved state distribution heatmap to {save_path}")

def main():
    """Main execution function."""
    # Setup plotting style
    setup_plotting()
    
    # Get data file
    filepath = get_data_file()
    print(f"Using data file: {filepath}")
    
    # Load data
    df = pd.read_csv(filepath)
    df.name = filepath  # Store filename for later use
    
    # Determine parameter being swept
    param_columns = [col for col in df.columns if col not in ['state', 'state_std', 'rewiring', 'mode', 'topology']]
    if 'political_climate' in param_columns:
        param_name = 'political_climate'
    elif param_columns:
        param_name = param_columns[0]
    else:
        param_name = "Unknown Parameter"
    
    print(f"Parameter being swept: {param_name}")
    
    # Preprocess data - limiting to parameter values <= 0.06
    df, max_param_value = preprocess_data(df, param_name, max_param_value=0.06)
    
    # Create visualization
    create_state_distribution_grid(df, param_name, max_param_value)

if __name__ == "__main__":
    main()