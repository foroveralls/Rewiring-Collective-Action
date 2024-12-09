import seaborn as sns
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    """Configure the global plotting style settings."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,  # Use TrueType fonts
        'ps.fonttype': 42,   # Use TrueType fonts
        'svg.fonttype': 'none'  # Use system fonts
    })
    sns.set_theme(font_scale=FONT_SIZE/12)
    sns.set(style="ticks")
    sns.set(rc={
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        "axes.grid": True,
        "grid.color": 'black',
        'grid.linestyle': 'dotted', 
        "axes.edgecolor": "black",
        "patch.edgecolor": "black",
        "patch.linewidth": 0,
        "axes.spines.bottom": True,
        "grid.alpha": 0.5,
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
    
    # Create scenario column
    df['scenario'] = df['rewiring'] + ' ' + df['mode']
    
    return df

def get_friendly_name(scenario):
    """Convert scenario name to friendly name."""
    parts = scenario.split()
    key = f"{parts[1]}_{parts[0]}" if len(parts) > 1 else f"{parts[0]}_none"
    return FRIENDLY_NAMES.get(key, scenario)

def format_ticks(x, pos):
    """Format tick labels to 2 decimal places."""
    return f'{x:.2f}'

def create_single_heatmap(ax, data, col, show_cbar, scenario_name, column_labels):
    """Create a single heatmap subplot."""
    if data.empty:
        ax.text(0.5, 0.5, 'No data available',
               horizontalalignment='center',
               verticalalignment='center',
               transform=ax.transAxes,
               fontsize=FONT_SIZE)
        return
    
    # For each cell, calculate the fraction of runs that ended in each state
    # and use the most common final state
    heatmap_data = data.applymap(lambda x: np.mean(x) if isinstance(x, list) else x)
    
    hm = sns.heatmap(heatmap_data,
                     ax=ax,
                     cmap='viridis',
                     cbar=show_cbar,
                     cbar_kws={'label': 'value' if show_cbar else None,
                              'format': ticker.FuncFormatter(format_ticks)})
    
    # Add scatter points where there's significant variance in final states
    # (indicating potential multistability)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            cell_data = data.iloc[i, j]
            if isinstance(cell_data, list) and len(cell_data) > 0:
                if np.std(cell_data) > 0.1:  # Threshold for significant variance
                    ax.plot(j + 0.5, i + 0.5, 'k.', markersize=5)
    
    ax.invert_yaxis()
    return hm
def set_axis_labels(ax, is_first_col, is_last_row, is_first_row, col, friendly_scenario, column_labels):
    """Set axis labels and titles with LaTeX-friendly rotation."""
    if is_first_col:
        ax.set_ylabel('Polarising Node f', fontsize=FONT_SIZE)
        
        # Use transform_rotates_text for better PDF rendering
        ax.text(-0.15, 0.5, friendly_scenario,
               transform=ax.transAxes,
               ha='right',
               va='center',
               fontweight='bold',
               fontsize=FONT_SIZE,
               rotation=45,
               rotation_mode='anchor',  # Important for better text rendering
               color=FRIENDLY_COLORS.get(friendly_scenario, 'black'))
    else:
        ax.set_ylabel('')
    
    if is_last_row:
        ax.set_xlabel('Stubbornness', fontsize=FONT_SIZE)
    else:
        ax.set_xlabel('')
    
    if is_first_row:
        ax.set_title(column_labels[col], fontsize=FONT_SIZE, fontweight='bold')
        
def create_heatmap_grid(df, value_columns, unique_scenarios, topology, column_labels):
    """Create grid of heatmaps for a given topology."""
    fig, axes = plt.subplots(nrows=len(unique_scenarios),
                            ncols=len(value_columns),
                            figsize=(18, 5 * len(unique_scenarios)),
                            sharex=True, sharey=True)
    
    
    for i, scenario in enumerate(unique_scenarios):
        friendly_scenario = get_friendly_name(scenario)
        scenario_data = df[(df['scenario'] == scenario) & (df['topology'] == topology)]
        
        for k, col in enumerate(value_columns):
            ax = axes[i, k] if len(unique_scenarios) > 1 else axes[k]
            heatmap_data = scenario_data.pivot_table(
                index='polarisingNode_f',
                columns='stubbornness',
                values=col,
                aggfunc=lambda x: list(x)
            )
            
            create_single_heatmap(ax, heatmap_data, col, k == len(value_columns) - 1, friendly_scenario, column_labels)
            set_axis_labels(ax, k == 0, i == len(unique_scenarios) - 1, i == 0, col, friendly_scenario, column_labels)
    
    # Adjust suptitle position (x=0.55 moves it slightly right)
    fig.suptitle(f'Topology: {topology}',
               fontsize=FONT_SIZE+6,
               fontweight='bold',
               y=1.01,
               x=0.55)  # Added x parameter to center the title
  
    return fig

def save_figure(fig, topology):
    """Save the figure to file with LaTeX-friendly settings."""
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
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