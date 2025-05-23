#!/usr/bin/env python3
"""
Modified heatmap visualization script with compact horizontal layout.
Creates a grid with topologies as 2-row blocks and scenarios as columns.
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
BASE_FONT_SIZE = 10
cm = 1/2.54

# Define font sizes for different elements
TITLE_FONT_SIZE = BASE_FONT_SIZE
AXIS_LABEL_FONT_SIZE = BASE_FONT_SIZE - 1
TICK_FONT_SIZE = BASE_FONT_SIZE - 3
LEGEND_FONT_SIZE = BASE_FONT_SIZE - 2
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE - 1
FRIENDLY_COLORS = {
    'static': '#EE7733',      # Orange
    'random': '#0077BB',      # Blue
    'L-sim': '#33BBEE',       # Cyan
    'L-opp': '#009988',       # Teal
    'B-sim': '#CC3311',       # Red
    'B-opp': '#EE3377',       # Magenta
    'WTF': '#BBBBBB',         # Grey
    'N2V': '#44BB99'     # Blue-green
}

FRIENDLY_NAMES = {
    'none_none': 'static',
    'random_none': 'random',
    'biased_same': 'L-sim',
    'biased_diff': 'L-opp',
    'bridge_same': 'B-sim',
    'bridge_diff': 'B-opp',
    'wtf_none': 'WTF',
    'node2vec_none': 'N2V'
}

EXCLUDED_SCENARIOS = ['static']
EXCLUDED_TOPOLOGIES = ['cl', 'DPAH']

# ====================== FUNCTIONS ======================

def setup_plotting_style():
    plt.rcParams.update({
        'font.size': BASE_FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.dpi': 300,
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

def get_data_file():
    file_list = [f for f in os.listdir("../../Output") 
                 if f.endswith(".csv") and "heatmap" in f]
    
    if not file_list:
        print("No suitable files found in the Output directory.")
        exit()
    
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")
    
    file_index = int(input("Enter the index of the file you want to plot: "))
    return os.path.join("../../Output", file_list[file_index])

def prepare_dataframe(df):
    df['rewiring'] = df['rewiring'].fillna('none')
    df['mode'] = df['mode'].fillna('none')
    df['scenario'] = df['rewiring'] + ' ' + df['mode']
    return df

def get_friendly_name(scenario):
    parts = scenario.split()
    key = f"{parts[1]}_{parts[0]}" if len(parts) > 1 else f"{parts[0]}_none"
    return FRIENDLY_NAMES.get(key, scenario)

def create_comprehensive_heatmap_grid(df, value_columns, column_labels):
    """Create compact grid: topologies as 2-row blocks, scenarios as columns"""
    scenarios = df['scenario'].unique()
    topologies = [t for t in df['topology'].unique() if t not in EXCLUDED_TOPOLOGIES]
    
    # Filter scenarios
    friendly_scenarios = {}
    for scenario in scenarios:
        friendly_name = get_friendly_name(scenario)
        if friendly_name not in EXCLUDED_SCENARIOS:
            friendly_scenarios[scenario] = friendly_name
    
    sorted_scenarios = sorted(friendly_scenarios.keys(), key=lambda s: friendly_scenarios[s])
    
    # Layout: 2 rows per topology, scenarios as columns
    n_topology_blocks = len(topologies)
    n_rows = n_topology_blocks * 2
    n_cols = len(sorted_scenarios)
    
    fig = plt.figure(figsize=(17.8*cm, max(10*cm, n_rows * 2*cm)))
    
    gs = fig.add_gridspec(n_rows, n_cols, 
                         hspace=0.15, wspace=0.05,
                         left=0.08, right=0.92, top=0.9, bottom=0.15)
    
    # Colormaps
    coop_cmap = sns.diverging_palette(20, 220, as_cmap=True, center="light")
    polar_cmap = sns.color_palette("Reds", as_cmap=True)
    
    # Add scenario labels at top
    for s, scenario in enumerate(sorted_scenarios):
        friendly_scenario = friendly_scenarios[scenario]
        scenario_color = FRIENDLY_COLORS.get(friendly_scenario, 'black')
        fig.text(0.08 + (s + 0.5) * 0.84/n_cols, 0.91, 
                 friendly_scenario, 
                 ha='center', va='bottom', 
                 fontsize=TITLE_FONT_SIZE, fontweight='bold',
                 color=scenario_color)
    
    # Create heatmaps
    for t_idx, topology in enumerate(topologies):
        # Add topology label on left (moved closer)
        fig.text(0.003, 0.15 + (n_topology_blocks - t_idx - 0.5) * 0.75/n_topology_blocks, 
                 topology.upper(),
                 ha='center', va='center',
                 fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold', rotation=90)
        
        for s, scenario in enumerate(sorted_scenarios):
            scenario_topo_data = df[(df['scenario'] == scenario) & (df['topology'] == topology)]
            
            if scenario_topo_data.empty:
                continue
                
            for m, metric in enumerate(value_columns):
                row = t_idx * 2 + m
                ax = fig.add_subplot(gs[row, s])
                
                try:
                    heatmap_data = scenario_topo_data.pivot_table(
                        index='polarisingNode_f',
                        columns='stubbornness',
                        values=metric,
                        aggfunc='mean'
                    ).iloc[::-1]
                    
                    # Set colormap and limits
                    if metric == 'state':
                        cmap, vmin, vmax, center = coop_cmap, -1, 1, 0
                        cbar_label = '⟨x⟩'
                    else:
                        cmap, vmin, vmax, center = polar_cmap, 0, 1, None
                        cbar_label = '$\sigma(x)$'
                    
                    # Only show colorbar on rightmost column for bottom rows of each topology block
                    show_cbar = (s == n_cols - 1) #and (row % 2 == 1)
                    
                    sns.heatmap(heatmap_data, ax=ax, cmap=cmap, center=center,
                    vmin=vmin, vmax=vmax, cbar=show_cbar,
                    linewidths=0.2, linecolor='white',
                    cbar_kws={'label': cbar_label} if show_cbar else {})

                    # Force tick visibility with explicit styling
                    ax.tick_params(axis='both', which='major', 
                                   labelsize=TICK_FONT_SIZE, 
                                   colors='black',  # Force black color
                                   width=1,      # Thicker ticks
                                   length=1)       # Longer ticks
                    
                    # X-axis handling
                    if row == n_rows - 1:
                        n_x = len(heatmap_data.columns)
                        x_indices = range(0, n_x, max(1, n_x//5))
                        ax.set_xticks([i for i in x_indices])
                        ax.set_xticklabels([f'{heatmap_data.columns[i]:.1f}' for i in x_indices], 
                                           rotation=45, fontsize=TICK_FONT_SIZE, color='black')
                    else:
                        ax.set_xticklabels([])
                    
                    # Y-axis handling  
                    if s == 0:
                        n_y = len(heatmap_data.index)
                        y_indices = range(0, n_y, max(1, n_y//5))
                        ax.set_yticks([i for i in y_indices])
                        ax.set_yticklabels([f'{heatmap_data.index[i]:.1f}' for i in y_indices], 
                                           rotation=0, fontsize=TICK_FONT_SIZE, color='black')
                    else:
                        ax.set_yticklabels([])
                    
                    # Ensure ticks are visible
                    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=(row == n_rows - 1))
                    ax.tick_params(axis='y', left=True, right=False, labelleft=(s == 0))
                  
                    # Remove individual axis labels
                    
                    ax.set_xlabel('')
                    ax.set_ylabel('')

                except Exception as e:
                    print(f"Error plotting {scenario} for {topology}, {metric}: {e}")
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
    
    # Add axis labels (only once at bottom and left)
    fig.text(0.5, 0.04, 'Stubbornness, $\\mathbf{w_i}$', ha='center', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    fig.text(0.02, 0.52, 'Polarizing Node Fraction, $\\mathbf{\\omega}$', va='center', rotation=90, 
         fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    return fig

def save_figure(fig, output_name='compact_empirical_heatmap'):
    save_path = f'../../Figs/Heatmaps/{output_name}_{date.today()}.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf', transparent=True)
    fig.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    
    print(f"Saved figure to {save_path}")
    plt.show()

def main():
    setup_plotting_style()
    data_path = get_data_file()
    df = pd.read_csv(data_path)
    df = prepare_dataframe(df)
    
    value_columns = ['state', 'state_std']
    column_labels = {'state': 'avg(x)', 'state_std': 'std(x)'}
    
    fig = create_comprehensive_heatmap_grid(df, value_columns, column_labels)
    save_figure(fig, 'compact_empirical_heatmap')

if __name__ == "__main__":
    main()