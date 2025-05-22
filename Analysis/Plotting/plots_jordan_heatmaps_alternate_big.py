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
FONT_SIZE = 14
FRIENDLY_COLORS = {
    'static': '#EE7733',      # Orange
    'random': '#0077BB',      # Blue
    'L-sim': '#33BBEE',       # Cyan
    'L-opp': '#009988',       # Teal
    'B-sim': '#CC3311',       # Red
    'B-opp': '#EE3377',       # Magenta
    'wtf': '#BBBBBB',         # Grey
    'node2vec': '#44BB99'     # Blue-green
}

FRIENDLY_NAMES = {
    'none_none': 'static',
    'random_none': 'random',
    'biased_same': 'L-sim',
    'biased_diff': 'L-opp',
    'bridge_same': 'B-sim',
    'bridge_diff': 'B-opp',
    'wtf_none': 'wtf',
    'node2vec_none': 'node2vec'
}

EXCLUDED_SCENARIOS = ['static']
EXCLUDED_TOPOLOGIES = ['cl', 'DPAH']

# ====================== FUNCTIONS ======================

def setup_plotting_style():
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.dpi': 300
    })
    sns.set_theme(font_scale=FONT_SIZE/12)
    sns.set(style="ticks")

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
    
    fig = plt.figure(figsize=(7.0, max(4.0, n_rows * 0.8)))  # 17.8cm = 7.0 inches
    
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
        fig.text(0.08 + (s + 0.5) * 0.84/n_cols, 0.92, 
                 friendly_scenario, 
                 ha='center', va='bottom', 
                 fontsize=FONT_SIZE-1, fontweight='bold',
                 color=scenario_color)
    
    # Create heatmaps
    for t_idx, topology in enumerate(topologies):
        # Add topology label on left (moved closer)
        fig.text(0.02, 0.15 + (n_topology_blocks - t_idx - 0.5) * 0.75/n_topology_blocks, 
                 topology.upper(),
                 ha='center', va='center',
                 fontsize=FONT_SIZE, fontweight='bold', rotation=90)
        
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
                        cbar_label = 'Cooperation'
                    else:
                        cmap, vmin, vmax, center = polar_cmap, 0, 1, None
                        cbar_label = 'Polarization'
                    
                    # Only show colorbar on rightmost column for bottom rows of each topology block
                    show_cbar = (s == n_cols - 1)
                    
                    sns.heatmap(heatmap_data, ax=ax, cmap=cmap, center=center,
                               vmin=vmin, vmax=vmax, cbar=show_cbar,
                               linewidths=0.2, linecolor='white',

                               cbar_kws={'label': cbar_label} if show_cbar else {})

                    
                    # X-axis: only on bottom row, shorter ticks
                    if row == n_rows - 1:
                        # Keep original ticks but make them shorter
                        ax.tick_params(axis='x', length=2)
                        # Get every other tick to reduce crowding
                        x_ticks = ax.get_xticks()[::2] if len(ax.get_xticks()) > 5 else ax.get_xticks()
                        x_labels = [f'{heatmap_data.columns[int(i)]:.1f}' for i in x_ticks if int(i) < len(heatmap_data.columns)]
                        ax.set_xticks(x_ticks[:len(x_labels)])
                        ax.set_xticklabels(x_labels, rotation=45, fontsize=FONT_SIZE-4)
                    else:
                        ax.set_xticks([])
                        ax.set_xticklabels([])
                    
                    # Y-axis: only on leftmost column, shorter ticks
                    if s == 0:
                        ax.tick_params(axis='y', length=2)
                        # Get every other tick to reduce crowding
                        y_ticks = ax.get_yticks()[::2] if len(ax.get_yticks()) > 5 else ax.get_yticks()
                        y_labels = [f'{heatmap_data.index[int(i)]:.1f}' for i in y_ticks if int(i) < len(heatmap_data.index)]
                        ax.set_yticks(y_ticks[:len(y_labels)])
                        ax.set_yticklabels(y_labels, rotation=0, fontsize=FONT_SIZE-4)
                    else:
                        ax.set_yticks([])
                        ax.set_yticklabels([])
                    
                    # Remove individual axis labels
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                
                except Exception as e:
                    print(f"Error plotting {scenario} for {topology}, {metric}: {e}")
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])

    # Add axis labels (only once at bottom and left)
    fig.text(0.5, 0.06, 'Stubbornness, $w_i$', ha='center', fontsize=FONT_SIZE-1, fontweight='bold')
    fig.text(0.04, 0.52, 'Polarizing Node Fraction, $\phi$', va='center', rotation=90, 
     fontsize=FONT_SIZE-1, fontweight='bold')

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