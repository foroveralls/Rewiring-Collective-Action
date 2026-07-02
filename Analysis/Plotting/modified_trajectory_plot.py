#!/usr/bin/env python3
"""
Modified trajectory plot with:
- Truncation at t=80,000
- Steady-state markers on right spine
- Reduced white space
- Compatibility for horizontal combination with transformation_grid_plot
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator

# Import from original plots_lines
from plots_lines import (
    line_params, PLOT_COLORS, NETWORK_DISPLAY_NAMES, set_plot_style, process_data
)

cm = 1/2.54
FONT_SIZE = 10  # Increased from 7 for better readability

def configure_axis_style_truncated(ax, t_max, scale_type='linear', show_ylabel=True,
                                  show_xlabel=True, add_steady_state_markers=False):
    """Apply axis styling with truncated view and optional steady-state markers"""
    ax.set_ylim(-0.6, 1.1)

    if scale_type == 'symlog':
        ax.set_xscale('symlog', linthresh=50000)
        ax.set_xlim(0, t_max)
        xlabel = "Time, t (symlog)"
    else:  # linear
        ax.set_xlim(0, t_max)
        sci_formatter = ScalarFormatter(useMathText=True)
        sci_formatter.set_scientific(True)
        sci_formatter.set_powerlimits((-2, 1))
        ax.xaxis.set_major_formatter(sci_formatter)

        # Dynamic x-ticks based on t_max
        step = 10000 if t_max <= 60000 else 20000 if t_max <= 120000 else 50000
        ticks = np.arange(0, t_max + step, step)
        ax.set_xticks(ticks[ticks <= t_max])
        xlabel = "Time, t"

    # Grid configuration
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=line_params["grid_line_width"], zorder=1)
    ax.set_axisbelow(True)

    # Spine configuration
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(line_params["axis_line_width"])
        spine.set_zorder(100)

    # Y-axis configuration
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # Minor tick locators
    if scale_type == 'linear':
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Tick parameters
    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=3,
        width=line_params["tick_major_width"],
        colors='black',
        zorder=100,
        bottom=True, top=False, left=True, right=add_steady_state_markers,
        labelbottom=show_xlabel, labeltop=False,
        labelleft=show_ylabel, labelright=False
    )

    ax.tick_params(
        axis='both',
        which='minor',
        direction='out',
        length=1.5,
        width=line_params["tick_minor_width"],
        colors='black',
        zorder=100,
        bottom=True, top=False, left=True, right=False,
        labelbottom=False, labeltop=False,
        labelleft=False, labelright=False
    )

    return xlabel

def plot_network_dynamics_truncated(data, t_max=80000, scale_type='linear',
                                   add_steady_state_markers=True, output_file=None,
                                   for_combined=False, bands_df=None):
    """
    Plot network dynamics with truncated time window and steady-state markers

    Args:
        data: Processed dataframe (should contain ALL timesteps, not filtered)
        t_max: Maximum time to DISPLAY on x-axis (default 80,000)
        scale_type: 'linear' or 'symlog'
        add_steady_state_markers: If True, add markers on right spine showing steady state from max timestep
        output_file: Path to save figure
        for_combined: If True, use taller figure size (14cm) to match transformation grid height
        bands_df: Optional DataFrame from Output/trajectory_bands.csv with across-run
            IQR of the per-run mean opinion (cols: type,scenario,rewiring,t,
            opinion_q25/q50/q75). If given, q25-q75 ribbons are drawn behind the
            opinion lines (R1.7 ensemble uncertainty). Polarization is left unbanded
            to keep the panel readable.
    """
    # Create color mapping
    scenario_color_map = {}
    for scenario in data['scenario_grouped'].unique():
        base_scenario = '_'.join(scenario.split('_')[:2]).lower()
        scenario_color_map[scenario] = PLOT_COLORS.get(base_scenario, '#FE6900')

    # Create separate dataframes - KEEP ALL DATA for steady-state calculation
    avg_state_data_full = data[data['measure'] == 'avg_state'].copy()
    polarization_data_full = data[data['measure'] == 'polarization'].copy()

    # Find the actual maximum timestep in the FULL data
    actual_t_max = avg_state_data_full['t'].max()
    print(f"Full data extends to t={actual_t_max}, but plot will be truncated at t={t_max}")
    print(f"Steady-state will be calculated from last 10% of full data (t >= {actual_t_max * 0.9:.0f})")

    # Now create FILTERED versions for visualization only
    avg_state_data = avg_state_data_full[avg_state_data_full['t'] <= t_max].copy()
    polarization_data = polarization_data_full[polarization_data_full['t'] <= t_max].copy()

    # Create plot
    g = sns.relplot(
        data=avg_state_data,
        x='t', y='value',
        hue='scenario_grouped',
        col='type',
        linewidth=line_params["data_line_width"],
        linestyle='-',
        kind='line',
        col_wrap=2,
        height=4, aspect=1,
        palette=scenario_color_map,
        legend=False
    )

    # Reduce figure size for better space utilization
    # Use taller figure (14cm) when for_combined=True to match transformation grid height
    fig_height = 14*cm if for_combined else 10*cm
    g.fig.set_size_inches(15*cm, fig_height)

    # Add polarization lines
    for ax_idx, ax in enumerate(g.axes.flat):
        network_type = list(avg_state_data['type'].unique())[ax_idx]

        for scenario in polarization_data['scenario_grouped'].unique():
            pol_data = polarization_data[
                (polarization_data['scenario_grouped'] == scenario) &
                (polarization_data['type'] == network_type)
            ]

            if not pol_data.empty:
                ax.plot(
                    pol_data['t'],
                    pol_data['value'],
                    linestyle='--',
                    dashes=(4, 2),
                    color=scenario_color_map[scenario],
                    linewidth=line_params["data_line_width"]
                )

    # Across-run IQR ribbons around the opinion lines (R1.7 ensemble uncertainty).
    # Drawn behind the data lines (zorder 1.2 > grid, < lines); opinion only.
    if bands_df is not None:
        color_by_norm = {k.lower(): v for k, v in scenario_color_map.items()}
        bdf = bands_df.copy()
        bdf['sg_norm'] = (bdf['scenario'].astype(str) + '_'
                          + bdf['rewiring'].astype(str)).str.lower()
        bdf = bdf[bdf['t'] <= t_max]
        for ax_idx, ax in enumerate(g.axes.flat):
            network_type = list(avg_state_data['type'].unique())[ax_idx]
            for sg_norm, color in color_by_norm.items():
                sub = bdf[(bdf['sg_norm'] == sg_norm)
                          & (bdf['type'] == network_type)].sort_values('t')
                if sub.empty:
                    continue
                ax.fill_between(sub['t'], sub['opinion_q25'], sub['opinion_q75'],
                                color=color, alpha=0.18, linewidth=0, zorder=1.2)

    # Calculate steady-state values from the actual max timestep in the FULL data
    # (not limited by display t_max - we want the true steady state)
    steady_state_values = {}
    if add_steady_state_markers:
        # Use last 10% of ACTUAL data, not display window
        t_threshold = actual_t_max * 0.9

        for network_type in avg_state_data_full['type'].unique():
            steady_state_values[network_type] = {}

            for scenario in avg_state_data_full['scenario_grouped'].unique():
                # Get data in steady-state region from the FULL dataset (not truncated)
                steady_data = avg_state_data_full[
                    (avg_state_data_full['scenario_grouped'] == scenario) &
                    (avg_state_data_full['type'] == network_type) &
                    (avg_state_data_full['t'] >= t_threshold)
                ]

                if not steady_data.empty:
                    steady_state_values[network_type][scenario] = steady_data['value'].mean()

    # Apply styling to each subplot
    xlabel = "Time, t"
    for i, ax in enumerate(g.axes.flat):
        is_bottom_row = i >= 2
        is_left_col = i % 2 == 0

        xlabel = configure_axis_style_truncated(ax, t_max, scale_type,
                                               show_ylabel=is_left_col,
                                               show_xlabel=is_bottom_row,
                                               add_steady_state_markers=add_steady_state_markers)

        # Set titles
        network_type = list(avg_state_data['type'].unique())[i]
        ax.set_title(NETWORK_DISPLAY_NAMES.get(network_type, network_type))

        # Add steady-state markers on right spine
        if add_steady_state_markers and network_type in steady_state_values:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks([])

            # Add small markers for each scenario
            for scenario, ss_value in steady_state_values[network_type].items():
                color = scenario_color_map[scenario]
                ax2.plot([t_max], [ss_value], marker='>', markersize=4,
                        color=color, markeredgewidth=0.5, markeredgecolor='black',
                        clip_on=False, zorder=200)

            ax2.spines['right'].set_visible(True)
            ax2.spines['right'].set_linewidth(line_params["axis_line_width"])

        # Only show scientific notation offset on bottom row
        if not is_bottom_row and scale_type == 'linear':
            ax.xaxis.offsetText.set_visible(False)

    # Set axis labels
    g.set_axis_labels(xlabel, r"Opinion, $\langle a \rangle$")

    # Reduced spacing for compactness - further reduce horizontal white space
    # Increased bottom margin from 0.18 to 0.20 to add more space for legend
    g.fig.subplots_adjust(top=0.88, bottom=0.20, hspace=0.20, wspace=0.12, left=0.10, right=0.93)

    # Add legends
    fig = g.fig
    # Move top legend up slightly
    legend_ax = fig.add_axes([0.15, 0.94, 0.7, 0.03])
    legend_ax.axis('off')

    line_style_elements = [
        Line2D([], [], color='black', linestyle='-', label='opinion'),
        Line2D([], [], color='black', linestyle='--', dashes=(4, 2), label='polarization')
    ]
    if add_steady_state_markers:
        line_style_elements.append(
            Line2D([], [], color='gray', marker='>', linestyle='None',
                  markersize=4, label='steady state')
        )

    legend_ax.legend(handles=line_style_elements, ncol=3, loc='center',
                    frameon=True, bbox_to_anchor=(0.5, 0.5), fontsize=FONT_SIZE-2)

    # Move bottom legend up slightly and reduce font size
    bottom_legend_ax = fig.add_axes([0.15, 0.04, 0.7, 0.04])
    bottom_legend_ax.axis('off')

    color_elements = [
        Line2D([], [], color=PLOT_COLORS['none_none'], label='static'),
        Line2D([], [], color=PLOT_COLORS['random_none'], label='random'),
        Line2D([], [], color=PLOT_COLORS['biased_same'], label='L-sim'),
        Line2D([], [], color=PLOT_COLORS['biased_diff'], label='L-opp'),
        Line2D([], [], color=PLOT_COLORS['bridge_same'], label='B-sim'),
        Line2D([], [], color=PLOT_COLORS['bridge_diff'], label='B-opp'),
    ]

    if any('wtf' in s for s in data['scenario_grouped']):
        color_elements.append(Line2D([], [], color=PLOT_COLORS['wtf_none'], label='WTF'))

    if any('node2vec' in s for s in data['scenario_grouped']):
        color_elements.append(Line2D([], [], color=PLOT_COLORS['node2vec_none'], label='N2V'))

    bottom_legend_ax.legend(handles=color_elements, ncol=4, loc='center',
                          frameon=True, bbox_to_anchor=(0.5, 0.5), fontsize=FONT_SIZE-2)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return g

# Main execution
if __name__ == "__main__":
    set_plot_style()

    file_list = [f for f in os.listdir("../../Output") if f.endswith(".csv") and "default_run_avg" in f]

    if not file_list:
        print("No suitable files found in the Output directory.")
        exit()

    for i, file in enumerate(file_list):
        print(f"{i}: {file}")

    file_index = int(input("Enter the index of the file you want to plot: "))

    if file_index < 0 or file_index >= len(file_list):
        print("Invalid file index.")
        exit()

    scale = input("Scale type (linear/symlog): ").lower().strip()
    if scale not in ['linear', 'symlog']:
        scale = 'linear'

    data = pd.read_csv(os.path.join("../../Output", file_list[file_index]))
    t_max = 65000  #Display truncated at 70,000
    get_N, get_n = file_list[file_index].split("_")[4], file_list[file_index].split("_")[6]

    # Process data WITHOUT filtering by t_max to preserve full data for steady-state calculation
    # We'll apply the t_max truncation inside the plotting function for visualization only
    req_cols = ['t', 'avg_state', 'std_states', 'scenario', 'rewiring', 'type']
    for col in req_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found")

    data['rewiring'] = data['rewiring'].fillna('none')
    data['scenario'] = data['scenario'].fillna('none')
    data['scenario_grouped'] = data['scenario'].str.cat(data['rewiring'], sep='_')

    processed_data = pd.melt(
        data.drop(columns=['scenario', 'rewiring']).rename(columns={'std_states': 'polarization'}),
        id_vars=['t', 'type', 'scenario_grouped'],
        value_vars=['avg_state', 'polarization'],
        var_name='measure', value_name='value'
    )

    today = date.today()
    suffix = f"_{scale}" if scale != 'linear' else ""

    # Load across-run IQR bands if available (produced by summarize_individual_csv.py)
    bands_path = os.path.join("../../Output", "trajectory_bands.csv")
    bands_df = None
    if os.path.exists(bands_path):
        bands_df = pd.read_csv(bands_path, keep_default_na=False)
        for c in ['t', 'opinion_q25', 'opinion_q50', 'opinion_q75']:
            bands_df[c] = pd.to_numeric(bands_df[c], errors='coerce')
        print(f"Loaded IQR bands: {bands_path} ({len(bands_df)} rows)")
    else:
        print(f"No bands file at {bands_path}; plotting without ribbons")

    # Generate standard version
    plot_network_dynamics_truncated(
        processed_data, t_max, scale, add_steady_state_markers=True,
        output_file=f"../../Figs/Trajectories/network_dynamics_truncated_N{get_N}_n{get_n}_{today}{suffix}.png",
        bands_df=bands_df
    )

    # Generate version for combination with transformation grid (taller)
    plot_network_dynamics_truncated(
        processed_data, t_max, scale, add_steady_state_markers=True,
        output_file=f"../../Figs/Trajectories/network_dynamics_truncated_for_combined_N{get_N}_n{get_n}_{today}{suffix}.png",
        for_combined=True, bands_df=bands_df
    )

    plt.show()
    print(f"Truncated trajectory plot saved to Figs/Trajectories/ as PNG")
    print(f"Taller version for combination saved with 'for_combined' suffix")
