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
BASE_FONT_SIZE = 8
cm = 1/2.54  # centimeters to inches conversion

# Define font sizes for different elements
TITLE_FONT_SIZE = BASE_FONT_SIZE
AXIS_LABEL_FONT_SIZE = BASE_FONT_SIZE - 1
TICK_FONT_SIZE = BASE_FONT_SIZE - 2
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

EXCLUDED_SCENARIOS = ['static', 'B-sim', 'L-opp', 'N2V']
EXCLUDED_TOPOLOGIES = ['cl', 'DPAH', 'FB']

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
    # Explicit override, so the corrected grid can be selected non-interactively
    # (its filename does not follow the old sweep-id convention).
    override = os.environ.get("HEATMAP_CSV")
    if override:
        print(f"Using HEATMAP_CSV override: {override}")
        return override

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

def get_clean_ticks(n_vals):
    """Get clean tick positions and labels from 0 to 1"""
    tick_labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    tick_positions = [i * (n_vals - 1) / 5 for i in range(6)]
    return tick_positions, tick_labels


def get_friendly_name(scenario):
    parts = scenario.split()
    key = f"{parts[1]}_{parts[0]}" if len(parts) > 1 else f"{parts[0]}_none"
    return FRIENDLY_NAMES.get(key, scenario)

def create_heatmap_grid(df, value_columns, column_labels, for_combined=False):
    """Create compact grid: topologies as 2-row blocks, scenarios as columns

    Args:
        df: DataFrame with heatmap data
        value_columns: Columns to plot
        column_labels: Labels for columns
        for_combined: If True, adjust figure size for combined panel figure
    """
    scenarios = df['scenario'].unique()
    topologies = [t for t in df['topology'].unique() if t not in EXCLUDED_TOPOLOGIES]

    # Filter scenarios
    friendly_scenarios = {}
    for scenario in scenarios:
        friendly_name = get_friendly_name(scenario)
        if friendly_name not in EXCLUDED_SCENARIOS:
            friendly_scenarios[scenario] = friendly_name

    # UPDATED: Custom ordering instead of alphabetical
    desired_order = ['B-opp', 'B-sim', 'L-opp', 'L-sim', 'random', 'N2V', 'WTF']
    sorted_scenarios = sorted(friendly_scenarios.keys(),
                             key=lambda s: desired_order.index(friendly_scenarios[s])
                             if friendly_scenarios[s] in desired_order else 999)

    # Layout: 2 rows per topology, scenarios as columns
    n_topology_blocks = len(topologies)
    n_rows = n_topology_blocks * 2
    n_cols = len(sorted_scenarios)

    # Use full width when combining — heatmap gets its own row in 2-row layout
    if for_combined:
        fig = plt.figure(figsize=(17.8*cm, max(8*cm, n_rows * 2*cm)))
        gs = fig.add_gridspec(n_rows, n_cols,
                             hspace=0.15, wspace=0.05,
                             left=0.08, right=0.86, top=0.9, bottom=0.15)
    else:
        fig = plt.figure(figsize=(17.8*cm, max(10*cm, n_rows * 2*cm)))
        # Leave more space on right for colorbars (0.96 instead of 0.92)
        gs = fig.add_gridspec(n_rows, n_cols,
                             hspace=0.15, wspace=0.05,
                             left=0.08, right=0.86, top=0.9, bottom=0.15)
    
    # Colormaps
    coop_cmap = sns.diverging_palette(20, 220, as_cmap=True, center="light")
    polar_cmap = sns.color_palette("Reds", as_cmap=True)
    
    # Add scenario labels at top
    # Position depends on whether this is for combined figure or standalone
    label_left = 0.08
    label_width = 0.78  # from left=0.08 to right=0.86

    for s, scenario in enumerate(sorted_scenarios):
        friendly_scenario = friendly_scenarios[scenario]
        scenario_color = FRIENDLY_COLORS.get(friendly_scenario, 'black')
        fig.text(label_left + (s + 0.5) * label_width/n_cols, 0.91,
                 friendly_scenario,
                 ha='center', va='bottom',
                 fontsize=TITLE_FONT_SIZE,
                 color=scenario_color)
    
    # Create heatmaps
    for t_idx, topology in enumerate(topologies):
        # Add topology label on left (position depends on layout)
        label_x = 0.003
        fig.text(label_x, 0.15 + (n_topology_blocks - t_idx - 0.5) * 0.75/n_topology_blocks,
                 topology.upper(),
                 ha='center', va='center',
                 fontsize=AXIS_LABEL_FONT_SIZE, rotation=90)
        
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
                        aggfunc='mean')
                    #).iloc[::-1]
                    
                    #heatmap_data = heatmap_data.iloc[::-1]
                    
                    # Set colormap and limits
                    if metric == 'state':
                        cmap, vmin, vmax, center = coop_cmap, -1, 1, 0
                        cbar_label = '$\\langle a^* \\rangle$'
                    else:
                        cmap, vmin, vmax, center = polar_cmap, 0, 1, None
                        cbar_label = '$\\langle P^* \\rangle$'
                    
                    # Don't show colorbar in heatmap - we'll add them manually later
                    sns.heatmap(heatmap_data, ax=ax, cmap=cmap, center=center,
                                vmin=vmin, vmax=vmax, cbar=False,
                                linewidths=0, linecolor='white')
                    
                    # Add grid for better readability
                    ax.grid(True, linestyle='--', alpha=0.2, linewidth=0.2)
                    
                    
                    # Add black border around subplot
                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(0.4)
                        spine.set_visible(True)
                                        

                    ax.invert_yaxis()
                    
                    # Force tick visibility with explicit styling
                    ax.tick_params(axis='both', which='major', 
                                   labelsize=TICK_FONT_SIZE, 
                                   colors='black',
                                   width=1,
                                   length=1)
                    
                    # Unified tick handling
                    n_x, n_y = len(heatmap_data.columns), len(heatmap_data.index)
                    x_indices = range(0, n_x, max(1, n_x//5))
                    y_indices = range(0, n_y, max(1, n_y//5))
                    
                    ax.set_xticks([i + 0.5 for i in x_indices])
                    ax.set_yticks([i + 0.5 for i in y_indices])
                    
                    # X-axis labels (bottom row only)
                    if row == n_rows - 1:
                        ax.set_xticklabels([f'{i/(n_x-1):.1f}' for i in x_indices], 
                                          rotation=45, fontsize=TICK_FONT_SIZE, color='black')
                    else:
                        ax.set_xticklabels([])
                    
                    # Y-axis labels (leftmost column only) - same range as x-axis
                    if s == 0:
                        # Use same formula as x-axis but reverse for correct visual positioning
                        ax.set_yticklabels([f'{i/(n_y-1):.1f}' for i in y_indices], 
                                          rotation=0, fontsize=TICK_FONT_SIZE, color='black')
                    else:
                        ax.set_yticklabels([])
                    
                    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=(row == n_rows - 1))
                    ax.tick_params(axis='y', left=True, right=False, labelleft=(s == 0))
                    
                    if s == 0:  # Only leftmost column gets labels
                        clean_labels = [f'{i/(n_y-1):.1f}' for i in y_indices]
                        ax.set_yticklabels(clean_labels, rotation=0, fontsize=TICK_FONT_SIZE, color='black')
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
    # Adjust label positions based on layout
    y_label_x = 0.02
    x_label_y = 0.04
    y_label_fontsize = AXIS_LABEL_FONT_SIZE

    fig.text(0.5, x_label_y, 'Stubbornness, $w_i$', ha='center',
             fontsize=y_label_fontsize)
    fig.text(y_label_x, 0.55, 'Diverger fraction, $\\rho$', va='center', rotation=90,
             fontsize=y_label_fontsize)

    # Add manual colorbars outside the gridspec to avoid squishing the WTF column
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    # Create colorbar axes on the right side
    # Cooperation colorbar (top half)
    cbar_ax1 = fig.add_axes([0.88, 0.55, 0.015, 0.32])  # [left, bottom, width, height]
    norm1 = Normalize(vmin=-1, vmax=1)
    cb1 = ColorbarBase(cbar_ax1, cmap=coop_cmap, norm=norm1, orientation='vertical')
    cb1.set_label('Opinion, $\\langle a^* \\rangle$', fontsize=AXIS_LABEL_FONT_SIZE)
    cb1.ax.tick_params(labelsize=TICK_FONT_SIZE, length=2, width=0.5)

    # Polarization colorbar (bottom half)
    cbar_ax2 = fig.add_axes([0.88, 0.18, 0.015, 0.32])  # [left, bottom, width, height]
    norm2 = Normalize(vmin=0, vmax=1)
    cb2 = ColorbarBase(cbar_ax2, cmap=polar_cmap, norm=norm2, orientation='vertical')
    cb2.set_label('Polarization, $\\langle P^* \\rangle$', fontsize=AXIS_LABEL_FONT_SIZE)
    cb2.ax.tick_params(labelsize=TICK_FONT_SIZE, length=2, width=0.5)

    return fig

def save_figure(fig, output_name='stubborness_backfirer_heatmap'):
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
    
    fig = create_heatmap_grid(df, value_columns, column_labels)
    save_figure(fig)

if __name__ == "__main__":
    main()