#!/usr/bin/env python3
"""
Enhanced convergence rate and cooperation heatmap visualization.
Combines processed convergence rate data with trajectory sweep cooperation data.
Creates a grid with topologies as 2-row blocks and scenarios as columns.
Row 1: Convergence rate distribution
Row 2: Final cooperation level (mean state)

Compatible with:
- Convergence rate data from process_convergence_rates.py
- Cooperation data from trajectory_sweep.py output
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib import transforms
from datetime import date
import pickle
from pathlib import Path

# ====================== CONFIGURATION ======================
BASE_FONT_SIZE = 9
cm = 1/2.54

# Define font sizes
TITLE_FONT_SIZE = BASE_FONT_SIZE
AXIS_LABEL_FONT_SIZE = BASE_FONT_SIZE - 1
TICK_FONT_SIZE = BASE_FONT_SIZE - 3
LEGEND_FONT_SIZE = BASE_FONT_SIZE - 2
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE - 1

FRIENDLY_COLORS = {
    'static': '#EE7733',
    'random': '#0077BB',
    'L-sim': '#33BBEE',
    'L-opp': '#009988',
    'B-sim': '#CC3311',
    'B-opp': '#EE3377',
    'WTF': '#BBBBBB',
    'N2V': '#44BB99'
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

def get_convergence_rate_files():
    """Find processed convergence rate pickle files"""
    input_dir = "../../Output/ProcessedRates"

    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        return None

    files = [f for f in os.listdir(input_dir) if f.endswith('.pkl') and 'ALL' in f]

    if not files:
        print("No processed convergence rate files found.")
        return None

    files.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)), reverse=True)

    print("\nAvailable convergence rate files:")
    for i, file in enumerate(files):
        print(f"{i}: {file}")

    file_index = int(input("Enter the index of the convergence rate file: "))
    return os.path.join(input_dir, files[file_index])

def get_cooperation_file():
    """Find trajectory sweep CSV files with cooperation data"""
    output_dir = "../../Output"

    files = [f for f in os.listdir(output_dir)
             if f.endswith('.csv') and 'param_sweep_individual' in f]

    if not files:
        print("No trajectory sweep files found.")
        return None

    files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)

    print("\nAvailable trajectory sweep files:")
    for i, file in enumerate(files):
        print(f"{i}: {file}")

    try:
        file_index = int(input("Enter the index of the cooperation data file: "))
        return os.path.join(output_dir, files[file_index])
    except:
        print("Invalid selection. Skipping cooperation data.")
        return None

def load_convergence_data(filepath):
    """Load pre-processed convergence rate data"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        if 'all_results' not in data:
            print(f"Error: Invalid data format - missing 'all_results' key")
            return None

        print(f"Loaded convergence data with param_name: {data.get('param_name', 'unknown')}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_cooperation_data(filepath, param_name):
    """Load and aggregate cooperation data from trajectory sweep CSV"""
    if filepath is None:
        return None

    try:
        print(f"Loading cooperation data from {filepath}...")

        # Read CSV - get final timestep values for each run
        df = pd.read_csv(filepath, low_memory=False)

        # Get the final timestep for each run
        final_states = df.loc[df.groupby('model_run')['t'].idxmax()]

        # Create scenario identifier
        final_states['scenario_id'] = final_states.apply(
            lambda row: f"{row['scenario']}_{row['rewiring']}"
            if str(row['scenario']).lower() in ['biased', 'bridge']
            else row['scenario'],
            axis=1
        )

        # Aggregate by topology, scenario, and parameter
        coop_data = {}

        for topology in final_states['type'].unique():
            coop_data[topology] = {}

            topology_data = final_states[final_states['type'] == topology]

            for scenario_id in topology_data['scenario_id'].unique():
                scenario_data = topology_data[topology_data['scenario_id'] == scenario_id]

                # Get parameter values and compute statistics
                param_vals = sorted(scenario_data[param_name].unique())

                rates_dict = {}
                distributions_dict = {}

                for pval in param_vals:
                    param_data = scenario_data[scenario_data[param_name] == pval]
                    states = param_data['avg_state'].values

                    if len(states) > 0:
                        rates_dict[pval] = np.mean(states)
                        distributions_dict[pval] = states.tolist()

                if rates_dict:
                    coop_data[topology][scenario_id] = {
                        'rates': rates_dict,
                        'distributions': distributions_dict,
                        'param_values': param_vals
                    }

        print(f"Loaded cooperation data for {len(coop_data)} topologies")
        return coop_data

    except Exception as e:
        print(f"Error loading cooperation data: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_friendly_name(scenario, rewiring='none'):
    """Get user-friendly algorithm name"""
    if scenario is None:
        return "Unknown"

    if rewiring is None or pd.isna(rewiring) or rewiring == 'None':
        rewiring = "none"

    scenario = str(scenario).lower()
    rewiring = str(rewiring).lower()

    # Handle combined ID format (e.g., "biased_diff")
    if "_" in scenario:
        parts = scenario.split("_")
        scenario = parts[0]
        rewiring = parts[1]

    key = f"{scenario}_{rewiring}"
    if scenario in ["none", "random", "wtf", "node2vec"]:
        key = f"{scenario}_none"

    friendly_name = FRIENDLY_NAMES.get(key, f"{scenario} ({rewiring})")
    return friendly_name

def create_combined_heatmap_grid(conv_data, coop_data=None):
    """
    Create compact grid: topologies as 2-row blocks, scenarios as columns
    Row 1: Convergence rate distribution
    Row 2: Cooperation (final state) if available, else median convergence rate
    """
    all_results = conv_data['all_results']
    param_name = conv_data.get('param_name', 'polarisingNode_f')

    # Get all unique topologies and scenarios
    all_topologies = [t for t in all_results.keys() if t not in EXCLUDED_TOPOLOGIES]

    # Preferred topology order
    preferred_order = ["DPAH", "Twitter", "cl", "FB"]
    all_topologies = sorted(all_topologies,
                           key=lambda t: preferred_order.index(t) if t in preferred_order else 999)

    # Create set of all scenarios
    all_scenarios = set()
    for topology, results in all_results.items():
        all_scenarios.update(results.keys())

    # Extract rewiring modes
    rewiring_modes = {}
    for topology, results in all_results.items():
        for scenario in results:
            if '_' in scenario:
                parts = scenario.split('_')
                rewiring_modes[scenario] = parts[1]
            elif scenario.lower() in ['none', 'random', 'wtf', 'node2vec', 'static']:
                rewiring_modes[scenario] = 'none'
            else:
                rewiring_modes[scenario] = 'none'

    # Filter scenarios
    friendly_scenarios = {}
    for scenario in all_scenarios:
        friendly_name = get_friendly_name(scenario, rewiring_modes.get(scenario, 'none'))
        if friendly_name not in EXCLUDED_SCENARIOS:
            friendly_scenarios[scenario] = friendly_name

    # Custom ordering
    desired_order = ['B-opp', 'B-sim', 'L-opp', 'L-sim', 'random', 'N2V', 'WTF']
    sorted_scenarios = sorted(friendly_scenarios.keys(),
                             key=lambda s: desired_order.index(friendly_scenarios[s])
                             if friendly_scenarios[s] in desired_order else 999)

    # Layout
    n_topology_blocks = len(all_topologies)
    n_rows = n_topology_blocks * 2
    n_cols = len(sorted_scenarios)

    fig = plt.figure(figsize=(17.8*cm, max(10*cm, n_rows * 2*cm)))

    gs = fig.add_gridspec(n_rows, n_cols,
                         hspace=0.15, wspace=0.05,
                         left=0.08, right=0.92, top=0.9, bottom=0.15)

    # Colormaps
    conv_cmap = sns.diverging_palette(250, 10, as_cmap=True, center="light")
    coop_cmap = sns.diverging_palette(20, 220, as_cmap=True, center="light")

    # Add scenario labels at top
    for s, scenario in enumerate(sorted_scenarios):
        friendly_scenario = friendly_scenarios[scenario]
        scenario_color = FRIENDLY_COLORS.get(friendly_scenario, 'black')
        fig.text(0.08 + (s + 0.5) * 0.84/n_cols, 0.91,
                 friendly_scenario,
                 ha='center', va='bottom',
                 fontsize=TITLE_FONT_SIZE, fontweight='bold',
                 color=scenario_color)

    # Track last images for colorbars
    last_conv_im = None
    last_coop_im = None

    # Create heatmaps
    for t_idx, topology in enumerate(all_topologies):
        # Add topology label
        fig.text(0.003, 0.15 + (n_topology_blocks - t_idx - 0.5) * 0.75/n_topology_blocks,
                 topology.upper(),
                 ha='center', va='center',
                 fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold', rotation=90)

        topology_conv = all_results.get(topology, {})
        topology_coop = coop_data.get(topology, {}) if coop_data else {}

        for s, scenario in enumerate(sorted_scenarios):
            if scenario not in topology_conv:
                continue

            scenario_conv = topology_conv[scenario]
            scenario_coop = topology_coop.get(scenario, None)
            param_vals = scenario_conv.get('param_values', [])

            if not param_vals:
                continue

            # Row 1: Convergence rate heatmap
            row_conv = t_idx * 2
            ax_conv = fig.add_subplot(gs[row_conv, s])

            try:
                # Build convergence rate 2D histogram
                conv_matrix = []
                for val in param_vals:
                    rates = scenario_conv['distributions'].get(val, [])
                    if rates:
                        conv_matrix.append(rates)
                    else:
                        conv_matrix.append([0])

                if conv_matrix:
                    n_bins = 20
                    all_rates_flat = [r for rates in conv_matrix for r in rates]

                    if all_rates_flat:
                        # Use percentiles for robust range
                        rate_min = np.percentile(all_rates_flat, 5)
                        rate_max = np.percentile(all_rates_flat, 95)

                        # Build heatmap
                        heatmap_data = np.zeros((len(param_vals), n_bins))
                        rate_bin_edges = np.linspace(rate_min, rate_max, n_bins + 1)

                        for i, rates in enumerate(conv_matrix):
                            if rates:
                                hist, _ = np.histogram(rates, bins=rate_bin_edges)
                                if np.sum(hist) > 0:
                                    # Sqrt normalization for better visibility
                                    hist = np.sqrt(hist / np.max(hist))
                                heatmap_data[i, :] = hist

                        # Plot
                        im = ax_conv.imshow(heatmap_data.T, aspect='auto', origin='lower',
                                          extent=[0, len(param_vals)-1, rate_min, rate_max],
                                          cmap=conv_cmap)
                        last_conv_im = im

                        # Add median line
                        medians = [np.median(rates) if rates else 0 for rates in conv_matrix]
                        ax_conv.plot(range(len(medians)), medians, 'r-', linewidth=0.8)

                        # Zero reference
                        if rate_min <= 0 <= rate_max:
                            ax_conv.axhline(y=0, color='black', linestyle='--',
                                          linewidth=0.5, alpha=0.7)

                # Styling
                for spine in ax_conv.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(0.4)
                    spine.set_visible(True)

                ax_conv.tick_params(axis='both', which='major',
                                   labelsize=TICK_FONT_SIZE, colors='black')

                n_x = len(param_vals)
                x_indices = range(0, n_x, max(1, n_x//5))
                ax_conv.set_xticks(x_indices)
                ax_conv.set_xticklabels([])

                if s == 0:
                    y_ticks = get_clean_yticks(rate_min, rate_max) if 'rate_min' in locals() else []
                    ax_conv.set_yticks(y_ticks)
                    ax_conv.set_yticklabels([f'{y:.1f}' for y in y_ticks],
                                           fontsize=TICK_FONT_SIZE)
                else:
                    ax_conv.set_yticklabels([])

                ax_conv.set_xlabel('')
                ax_conv.set_ylabel('')

            except Exception as e:
                print(f"Error plotting convergence for {scenario}/{topology}: {e}")
                ax_conv.text(0.5, 0.5, "No data", ha='center', va='center')
                ax_conv.set_xticks([])
                ax_conv.set_yticks([])

            # Row 2: Cooperation
            row_coop = t_idx * 2 + 1
            ax_coop = fig.add_subplot(gs[row_coop, s])

            try:
                if scenario_coop:
                    # Use actual cooperation data
                    coop_vals = []
                    for val in param_vals:
                        states = scenario_coop['distributions'].get(val, [])
                        if states:
                            coop_vals.append(np.mean(states))
                        else:
                            coop_vals.append(0)

                    coop_array = np.array(coop_vals).reshape(1, -1)
                else:
                    # Fallback: use median convergence rate
                    median_rates = []
                    for val in param_vals:
                        rates = scenario_conv['distributions'].get(val, [])
                        if rates:
                            median_rates.append(np.median(rates))
                        else:
                            median_rates.append(0)
                    coop_array = np.array(median_rates).reshape(1, -1)

                im2 = ax_coop.imshow(coop_array, aspect='auto', cmap=coop_cmap,
                                    vmin=-1, vmax=1, interpolation='bilinear')
                last_coop_im = im2

                # Styling
                for spine in ax_coop.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(0.4)
                    spine.set_visible(True)

                ax_coop.tick_params(axis='both', which='major',
                                   labelsize=TICK_FONT_SIZE, colors='black')

                ax_coop.set_xticks(x_indices)
                if row_coop == n_rows - 1:
                    ax_coop.set_xticklabels([f'{param_vals[i]:.1f}' for i in x_indices],
                                           rotation=45, fontsize=TICK_FONT_SIZE)
                else:
                    ax_coop.set_xticklabels([])

                ax_coop.set_yticks([])
                ax_coop.set_xlabel('')
                ax_coop.set_ylabel('')

            except Exception as e:
                print(f"Error plotting cooperation for {scenario}/{topology}: {e}")
                ax_coop.text(0.5, 0.5, "No data", ha='center', va='center')
                ax_coop.set_xticks([])
                ax_coop.set_yticks([])

    # Add colorbars
    if last_conv_im:
        cbar_ax1 = fig.add_axes([0.93, 0.55, 0.015, 0.3])
        cbar1 = fig.colorbar(last_conv_im, cax=cbar_ax1)
        cbar1.set_label('√(Count)', fontsize=LEGEND_FONT_SIZE)
        cbar1.ax.tick_params(labelsize=TICK_FONT_SIZE)

    if last_coop_im:
        cbar_ax2 = fig.add_axes([0.93, 0.15, 0.015, 0.3])
        cbar2 = fig.colorbar(last_coop_im, cax=cbar_ax2)
        cbar2.set_label('⟨a⟩', fontsize=LEGEND_FONT_SIZE)
        cbar2.ax.tick_params(labelsize=TICK_FONT_SIZE)

    # Axis labels
    fig.text(0.5, 0.04, f'{param_name.replace("_", " ").title()}, $\\mathbf{{\\rho}}$',
             ha='center', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    fig.text(0.02, 0.52, 'Convergence Rate / Cooperation', va='center', rotation=90,
             fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')

    return fig

def get_clean_yticks(rate_min, rate_max, num_ticks=5):
    """Generate clean y-ticks"""
    ticks = np.linspace(rate_min, rate_max, num_ticks)
    return ticks

def save_figure(fig, output_name='convergence_cooperation_heatmap'):
    save_path = f'../../Figs/Heatmaps/{output_name}_{date.today()}.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf', transparent=True)
    fig.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)

    print(f"Saved figure to {save_path}")
    plt.show()

def main():
    setup_plotting_style()

    # Get convergence rate data (required)
    conv_file = get_convergence_rate_files()
    if conv_file is None:
        print("No convergence rate file selected. Exiting.")
        return

    conv_data = load_convergence_data(conv_file)
    if conv_data is None:
        print("Failed to load convergence data. Exiting.")
        return

    # Get cooperation data (optional)
    coop_file = get_cooperation_file()
    if coop_file:
        param_name = conv_data.get('param_name', 'polarisingNode_f')
        coop_data = load_cooperation_data(coop_file, param_name)
    else:
        coop_data = None

    # Create the grid
    fig = create_combined_heatmap_grid(conv_data, coop_data)
    save_figure(fig)

if __name__ == "__main__":
    main()
