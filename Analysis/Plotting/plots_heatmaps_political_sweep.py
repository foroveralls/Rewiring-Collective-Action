#!/usr/bin/env python3
"""
Visualization script for cooperativity data from parameter sweeps.
Creates a grid of heatmaps showing final state distributions across topologies and scenarios.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from datetime import date

# ====================== CONFIGURATION ======================
# Output directory for plots
OUTPUT_DIR = "../../Figs/Heatmaps"

# Plot settings
BASE_FONT_SIZE = 8
cm = 1/2.54

# Define font sizes for different elements - reduced overall
TITLE_FONT_SIZE = BASE_FONT_SIZE
AXIS_LABEL_FONT_SIZE = BASE_FONT_SIZE - 1
TICK_FONT_SIZE = BASE_FONT_SIZE - 2
LEGEND_FONT_SIZE = BASE_FONT_SIZE - 2
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE - 1

# Algorithms to exclude from visualization (leave empty to include all)
EXCLUDED_ALGORITHMS = ["none_none"]

# Heatmap bin settings
PARAM_BINS = 12  # Number of bins for parameter values
STATE_BINS = 20  # Number of bins for state values

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
        'figure.figsize': (17.8*cm, 12*cm),  # Increased height only, width constrained by journal
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

def preprocess_data(df, param_name, max_param_value=0.10):
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
    
    return df

def create_state_heatmap_grid(df, param_name, max_param_value=0.06):
    """
    Create a grid of heatmaps showing state distribution by parameter value across topologies and scenarios.
    
    Parameters:
    df - Preprocessed DataFrame with state data
    param_name - Name of the parameter being swept
    max_param_value - Maximum value of the parameter to display
    """
    # Get unique topologies and scenarios
    all_topologies = sorted(df['topology'].unique())
    
    # Group by mode and rewiring to get unique scenarios
    scenarios = df.groupby(['mode', 'rewiring']).size().reset_index()
    all_scenarios = []
    
    for _, row in scenarios.iterrows():
        mode, rewiring = row['mode'], row['rewiring']
        scenario_key = f"{mode}_{rewiring}".lower()
        
        # Skip excluded algorithms
        if scenario_key in EXCLUDED_ALGORITHMS:
            continue
            
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
    
    # Grid layout with minimal space horizontally between plots
    n_rows = len(all_topologies)
    n_cols = len(all_scenarios)
    
    # Create figure with increased height but maintaining journal-required width
    fig = plt.figure(figsize=(17.8*cm, 12*cm))
    
    # Create GridSpec with minimal space horizontally between plots
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.35, hspace=0.3)
    
    # Use viridis colormap for consistency with trajectory plots
    cmap = plt.cm.viridis
    
    # Define the 2D histogram bins
    param_bins = np.linspace(0, max_param_value, PARAM_BINS+1)
    state_bins = np.linspace(-1, 1, STATE_BINS+1)
    
    # Create parameter values array for consistent tick spacing
    param_vals = np.linspace(0, max_param_value, PARAM_BINS)
    
    # Process each cell in grid
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
                # Compute 2D histogram manually for better control
                H, xedges, yedges = np.histogram2d(
                    cell_data[param_name], 
                    cell_data['state'],
                    bins=[param_bins, state_bins]
                )
                
                # Simple log transformation for enhancing visibility of low counts
                H_log = np.log1p(H)  # log(1+x) to handle zeros
                
                # Normalize to get full color range
                if H_log.max() > 0:
                    H_norm = H_log / H_log.max()
                else:
                    H_norm = H_log
                
                # Plot the normalized 2D histogram
                im = ax.pcolormesh(xedges, yedges, H_norm.T, cmap=cmap, vmin=0, vmax=1)
                
                # Set dynamic axis limits
                ax.set_xlim(0, max_param_value)
                ax.set_ylim(-1, 1)
                
                # Set x-ticks with consistent spacing like in trajectory plot
                x_ticks = np.arange(len(param_vals))
                x_tick_labels = [f'{val:.1f}' for val in param_vals]  # 1 decimal place
                tick_spacing = max(1, len(param_vals) // 5)  # 5 ticks max
                ax.set_xticks(param_vals[::tick_spacing])
                ax.set_xticklabels([f'{val:.1f}' for val in param_vals[::tick_spacing]], 
                                  fontsize=TICK_FONT_SIZE)
                
                # Set y-ticks consistently
                y_ticks = [-1, -0.5, 0, 0.5, 1]
                ax.set_yticks(y_ticks)
                
                # Only remove y-axis tick labels for non-first columns, keep the ticks
                if col_idx > 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=TICK_FONT_SIZE)
                
                # Apply tick parameters for consistent appearance
                ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE,
                              width=0.8, length=2, pad=2)
                
                # Add grid for better readability
                ax.grid(True, linestyle='--', alpha=0.2, linewidth=0.2)
            
            # Add labels with improved positioning for journal width
            if col_idx == 0:  # First column gets ⟨x⟩ label
                ax.set_ylabel('⟨x⟩', fontsize=AXIS_LABEL_FONT_SIZE, labelpad=5)
                
                # Add topology label further from plot (beyond ⟨x⟩)
                ax.text(-0.85, 0.5, topology.upper(), transform=ax.transAxes, 
                       rotation=90, fontsize=AXIS_LABEL_FONT_SIZE+1, 
                       fontweight='bold', va='center', ha='center')
            
            if row_idx == 0:  # First row gets scenario title
                title_color = FRIENDLY_COLORS.get(friendly_name, 'black')
                ax.set_title(friendly_name, fontsize=TITLE_FONT_SIZE-1, 
                           color=title_color, pad=2, fontweight='bold')
            
            if row_idx == n_rows - 1:  # Last row gets x-axis label
                ax.set_xlabel(param_name, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=5)
    
    # Add a colorbar with diagonal tick labels
    cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('log(Count+1)', fontsize=AXIS_LABEL_FONT_SIZE-1)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    cbar.outline.set_linewidth(0.4)  # Make colorbar outline thinner
    
    # Rotate colorbar ticks diagonally to save space
    plt.setp(cbar_ax.get_yticklabels(), rotation=45, ha='left')
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    sweep_id = os.path.basename(df.name).split('_')[-1].split('.')[0] if hasattr(df, 'name') else today
    save_path = f'{OUTPUT_DIR}/heatmap_states_grid_{param_name}_{sweep_id}'
    
    for ext in ['pdf', 'png']:
        plt.savefig(f"{save_path}.{ext}", bbox_inches='tight', dpi=300)
    
    plt.show()
    print(f"Saved heatmap states grid to {save_path}")

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
    
    # Set the max parameter value to display
    max_param_value = 0.06
    
    # Print excluded algorithms if any
    if EXCLUDED_ALGORITHMS:
        print(f"Excluding algorithms: {', '.join(EXCLUDED_ALGORITHMS)}")
    
    # Preprocess data - limiting to parameter values <= max_param_value
    df = preprocess_data(df, param_name, max_param_value=max_param_value)
    
    # Create visualization with dynamic x-axis
    create_state_heatmap_grid(df, param_name, max_param_value=max_param_value)

if __name__ == "__main__":
    main()