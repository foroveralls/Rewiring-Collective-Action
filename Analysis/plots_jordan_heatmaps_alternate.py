import seaborn as sns
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import transforms

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

def setup_plotting_style():
    """Updated plotting style configuration"""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.figsize': (12, 6),  # New: wider format for horizontal layout
        'figure.dpi': 300
    })
    sns.set_theme(font_scale=FONT_SIZE/12)
    sns.set(style="ticks")
    sns.set(rc={
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        "axes.grid": True,
        "grid.color": 'white',  # Changed to white for heatmap
        'grid.linestyle': 'solid', 
        "axes.edgecolor": "black",
        "patch.edgecolor": "black",
        "patch.linewidth": 0.5,  # Reduced from original
        "axes.spines.bottom": True,
        "grid.alpha": 0.8,  # Increased for better visibility
        "xtick.bottom": True,
        "ytick.left": True
    })


def get_data_file():
    """Get the data file path from user input."""
    file_list = [f for f in os.listdir("../Output") 
                 if f.endswith(".csv") and "heatmap" in f]
    
    if not file_list:
        print("No suitable files found in the Output directory.")
        exit()
    
    # Print file list with indices
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")
    
    file_index = int(input("Enter the index of the file you want to plot: "))
    return os.path.join("../Output", file_list[file_index])




def prepare_dataframe(df):
    """Prepare the DataFrame for plotting."""
    # Adjust the "rewiring" column for specific modes
    df.loc[df['mode'].isin(['wtf', 'node2vec']), 'rewiring'] = 'empirical'
    
  
    # Replace 'nan' with 'none' in 'rewiring' and 'mode' columns
    df['rewiring'] = df['rewiring'].fillna('none')
    df['mode'] = df['mode'].fillna('none')
    
    # Create scenario column
    df['scenario'] = df['rewiring'] + ' ' + df['mode']
    df['scenario'] = df['rewiring'] + ' ' + df['mode']

   # df['polarisingNode_f'] = df['polarisingNode_f'].round(1)
    #df['stubbornness'] = df['stubbornness'].round(1)
    
    return df

def get_friendly_name(scenario):
    """Convert scenario name to friendly name."""
    parts = scenario.split()
    print(parts)
    key = f"{parts[1]}_{parts[0]}" if len(parts) > 1 else f"{parts[0]}_none"
    return FRIENDLY_NAMES.get(key, scenario)

def format_ticks(x, pos):
    """Format tick labels to 2 decimal places."""
    try:
        return f'{float(x):.1f}'  # Changed to more robust formatting
    except (ValueError, TypeError):
        return str(x)



def set_axis_labels(ax, is_first_col, is_last_row, is_first_row, col, 
                   friendly_scenario, column_labels):
    if is_first_col:
        ax.set_ylabel('Polarizing Node Fraction', fontsize=FONT_SIZE)
    
        
def create_heatmap_grid(df, value_columns, unique_scenarios, topology, column_labels):
    """Create grid of heatmaps with improved layout and label positioning."""
    # Filter out scenarios with no data
    scenarios_with_data = []
    for scenario in unique_scenarios:
        scenario_data = df[(df['scenario'] == scenario) & (df['topology'] == topology)]
        if not scenario_data.empty:
            scenarios_with_data.append(scenario)
    
    n_scenarios = len(scenarios_with_data)
    n_cols = 4  # Two scenarios per row, each with avg and std
    n_rows = (n_scenarios + 1) // 2
    
    # Create figure with precise control over spacing
    fig = plt.figure(figsize=(20, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, 
                         hspace=0.5,
                         wspace=0.4,
                         left=0.1,
                         right=0.9,
                         top=0.92,
                         bottom=0.15)
    
    # Initialize axes array
    axes = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j] = fig.add_subplot(gs[i, j])
    
    # Define colormaps once
    coop_cmap = sns.diverging_palette(20, 220, as_cmap=True, center="light")
    polar_cmap = sns.color_palette("Reds", as_cmap=True)
    
    for i, scenario in enumerate(scenarios_with_data):
        row = i // 2
        col = (i % 2) * 2
        
        friendly_scenario = get_friendly_name(scenario)
        scenario_data = df[(df['scenario'] == scenario) & 
                         (df['topology'] == topology)]
        
            
        # Add scenario label at the top of each pair
        ax1, ax2 = axes[row, col], axes[row, col + 1]
        ax1.text(1.0, 1.15, friendly_scenario,
                transform=ax1.transAxes,
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=FONT_SIZE,
                color=FRIENDLY_COLORS.get(friendly_scenario, 'black'))
        
        for k, value_col in enumerate(value_columns):
            ax = axes[row, col + k]
            
            # Create pivot table
            heatmap_data = scenario_data.pivot_table(
                index='polarisingNode_f',
                columns='stubbornness',
                values=value_col,
                aggfunc='mean'
            )
            
            # Reverse the y-axis order
            heatmap_data = heatmap_data.iloc[::-1]
            
            # Set parameters based on metric type
            if value_col == 'state':
                cmap = coop_cmap
                vmin, vmax = -1, 1
                center = 0
                cbar_label = 'Cooperation'
              
            else:
                cmap = polar_cmap,
                vmin, vmax = 0,
                center = None,
                cbar_label = 'Polarization'
            
            # Single heatmap creation
            hm = sns.heatmap(heatmap_data,
                           ax=ax,
                           cmap=cmap,
                           center=center,
                           vmin=vmin,
                           vmax=vmax,
                           cbar=True,
                           cbar_kws={'label': cbar_label})
            
            # Grid and formatting
            ax.grid(True, which='major', color='white', linewidth=0.8)
            
            # Format tick labels
            ax.set_yticklabels([f'{y:.1f}' for y in heatmap_data.index], rotation=0)
            ax.set_xticklabels([f'{x:.1f}' for x in heatmap_data.columns], rotation=0)
            
            # Set axis labels
            set_axis_labels(ax, col + k == 0 or col + k == 2, 
                          row == n_rows - 1, row == 0,
                          value_col, friendly_scenario, column_labels)
            
            # Add metric labels
            if value_col == 'state':
                ax.set_title(r'$\langle x \rangle$', fontsize=FONT_SIZE+2)
            else:
                ax.set_title(r'$\sigma(x)$', fontsize=FONT_SIZE+2)
    
    # Hide empty subplots
    if n_scenarios % 2:
        for j in range(-2, 0):
            axes[-1, j].set_visible(False)
    
    # Add topology title
    fig.suptitle(f'Network Topology: {topology}',
                fontsize=FONT_SIZE+6,
                fontweight='bold',
                y=0.98)
    
    return fig
def save_figure(fig, topology):
    """Save the figure to file with LaTeX-friendly settings."""
    save_path = f'../Figs/Heatmaps/heatmap_{topology}.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save with specific PDF settings for better text rendering
    fig.savefig(save_path, 
                bbox_inches='tight', 
                dpi=300,
                format='pdf',
                backend='pdf',
                transparent=True)
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
        'state': 'avg(x)',     # Unicode angle brackets with s
        'state_std': 'std(x)'    # Unicode sigma with s
    }
    unique_scenarios = df['scenario'].unique()
    unique_topologies = df['topology'].unique()
    
    # Create and save plots for each topology
    for topology in unique_topologies:
        fig = create_heatmap_grid(df, value_columns, unique_scenarios, topology, column_labels)
        save_figure(fig, topology)


if __name__ == "__main__":
    main()