
"""
Enhanced visualization script for convergence rate data from a single unified file.
Creates a comprehensive grid display of all scenarios across all topologies.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import gc
from pathlib import Path
from matplotlib.colors import LogNorm, Normalize
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from datetime import date

# debug helper

def print_debug_info(all_results, rewiring_modes):
    """Print debug information about the data structure."""
    print("\nDEBUG INFO:")
    print("===========")
    for topology, scenarios in all_results.items():
        print(f"Topology: {topology}")
        for scenario in scenarios:
            friendly = get_friendly_name(scenario, rewiring_modes.get(scenario, 'none'))
            print(f"  Scenario: {scenario} -> Friendly: {friendly}")
            print(f"    Color key: {friendly.lower()}")
            print(f"    Color: {FRIENDLY_COLORS.get(friendly.lower(), 'black')}")
    print("===========\n")#!/usr/bin/env python3


# ====================== CONFIGURATION ======================
# Input directory with processed data
INPUT_DIR = "../../Output/ProcessedRates"

USE_MEAN = False

# Output directory for plots
OUTPUT_DIR = "../../Figs/ConvergenceRates"

# Plot settings - ADJUSTED FOR JOURNAL SPECIFICATIONS
BASE_FONT_SIZE = 8  # Reduced from 14 to fit journal specifications
RANDOM_SEED = 42

cm = 1/2.54

# Define font sizes for different elements
TITLE_FONT_SIZE = BASE_FONT_SIZE - 1  # Reduced scenario label size
AXIS_LABEL_FONT_SIZE = BASE_FONT_SIZE - 2
TICK_FONT_SIZE = BASE_FONT_SIZE - 4
LEGEND_FONT_SIZE = BASE_FONT_SIZE - 3  # Reduced colorbar label size
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE - 2

# Scenario names and colors mapping
FRIENDLY_COLORS = {
    'static': '#EE7733',
    'random': '#0077BB',
    'l-sim': '#33BBEE',     # Updated from 'local (similar)'
    'l-opp': '#009988',     # Updated from 'local (opposite)'
    'b-sim': '#CC3311',     # Updated from 'bridge (similar)'
    'b-opp': '#EE3377',     # Updated from 'bridge (opposite)'
    'wtf': '#BBBBBB',
    'node2vec': '#44BB99'
}

FRIENDLY_NAMES = {
    'none_none': 'static',
    'random_none': 'random',
    'biased_same': 'L-sim',  # "Local-similar" shortened to "L-sim"
    'biased_diff': 'L-opp',  # "Local-opposite" shortened to "L-opp"
    'bridge_same': 'B-sim',  # "Bridge-similar" shortened to "B-sim"
    'bridge_diff': 'B-opp',  # "Bridge-opposite" shortened to "B-opp"
    'wtf_none': 'wtf',
    'node2vec_none': 'node2vec'
}


# Line styling (following common publication standards)
ZERO_LINE_STYLE = {
    'linestyle': 'solid',      # Solid line for reference/baseline
    'color': 'black',
    'linewidth': 0.7,
    'alpha': 0.8,
    'zorder': 5            # Above heatmap
}

TRAJECTORY_LINE_STYLE = {
    'linestyle': '--',      # Dotted line for data trajectory
    'color': 'r',          # Red
    'linewidth': 0.8,
    'zorder': 10           # Above zero line
}
# Excluded scenarios (static in this case)
EXCLUDED_SCENARIOS = ['none_none', 'static', 'none']

# ====================== FUNCTIONS ======================

def setup_plotting():
    """Configure plotting style for publication quality."""
    plt.rcParams.update({
        'font.size': BASE_FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.dpi': 300,
        'figure.figsize': (17.8*cm, 10*cm),  # Increased height slightly
        'axes.labelsize': AXIS_LABEL_FONT_SIZE,
        'axes.titlesize': TITLE_FONT_SIZE,
        'xtick.labelsize': TICK_FONT_SIZE,
        'ytick.labelsize': TICK_FONT_SIZE,
        'legend.fontsize': LEGEND_FONT_SIZE,
        'legend.title_fontsize': LEGEND_FONT_SIZE,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'axes.linewidth': 0.8,
    })
    sns.set_theme(font_scale=BASE_FONT_SIZE/12)
    sns.set(style="ticks")
    sns.set(rc={
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'xtick.major.size': 2,
        'ytick.major.size': 2,
        "axes.grid": True,
        "grid.color": 'white',
        'grid.linestyle': 'solid', 
        "lines.linewidth": 0.7,
        "axes.edgecolor": "black",
        "patch.edgecolor": "black",
        "patch.linewidth": 0.2,
        "axes.spines.bottom": True,
        "grid.alpha": 0.8,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.2,
        "xtick.bottom": True,
        "ytick.left": True
    })
    
def get_clean_yticks(rate_min, rate_max, num_ticks=5):
    """Generate clean y-ticks that include zero when in range."""
    # Create evenly spaced ticks
    # ticks = np.linspace(rate_min, rate_max, num_ticks)
    
    # # If zero is in range but not in ticks, replace closest tick with zero
    # if rate_min <= 0 <= rate_max and 0 not in ticks:
    #     idx = np.abs(ticks).argmin()  # Find closest tick to zero
    #     ticks[idx] = 0.0              # Replace with exactly zero
    
    ticks = [-0.3, 0.0, 0.6]
    return ticks

def get_friendly_name(scenario, rewiring):
    """Get user-friendly algorithm name."""
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

def find_all_files():
    """Find all the unified .pkl files with ALL in the name."""
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        return []
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pkl') and 'ALL' in f]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(INPUT_DIR, x)), reverse=True)
    return files

def load_unified_results(filename):
    """Load the unified results from the ALL .pkl file."""
    filepath = os.path.join(INPUT_DIR, filename)
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if 'all_results' not in data:
            print(f"Error: Invalid data format in {filepath} - missing 'all_results' key")
            return None
        
        print(f"Loaded data with param_name: {data.get('param_name', 'unknown')}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_rewiring_modes(all_results):
    """Extract rewiring modes for all scenarios across all topologies."""
    rewiring_modes = {}
    
    for topology, results in all_results.items():
        for scenario in results:
            # Infer rewiring mode from scenario name if it's a combined format
            if '_' in scenario:
                parts = scenario.split('_')
                base_scenario = parts[0]
                mode = parts[1]
                rewiring_modes[scenario] = mode
            elif scenario.lower() in ['none', 'random', 'wtf', 'node2vec', 'static']:
                rewiring_modes[scenario] = 'none'
            # If we can't determine the mode, default to 'none'
            else:
                rewiring_modes[scenario] = 'none'
    
    print(f"Extracted rewiring modes: {rewiring_modes}")
    return rewiring_modes

def create_master_heatmap_grid(data):
    """
    Create a comprehensive grid of heatmaps showing all scenarios across all topologies.
    
    Parameters:
    data - Complete data dictionary from the unified file
    """
    all_results = data['all_results']
    param_name = data.get('param_name', 'Unknown Parameter')
    
    # Get all unique topologies and scenarios
    all_topologies = list(all_results.keys())
    
    # Define the preferred topology order (to keep directed and undirected together)
    preferred_order = ["DPAH", "Twitter", "cl", "FB"]
    all_topologies = sorted(all_topologies, key=lambda t: preferred_order.index(t) if t in preferred_order else 999)
    
    # Create a set of all scenarios across all topologies
    all_scenarios = set()
    for topology, results in all_results.items():
        all_scenarios.update(results.keys())
    
    # Extract rewiring modes for friendly names
    rewiring_modes = extract_rewiring_modes(all_results)
    
    # Handle excluded scenarios
    filtered_scenarios = []
    for s in sorted(all_scenarios):
        should_exclude = False
        if s in EXCLUDED_SCENARIOS:
            should_exclude = True
        # Also check if the friendly name maps to an excluded scenario
        friendly_name = get_friendly_name(s, rewiring_modes.get(s, 'none'))
        if friendly_name in EXCLUDED_SCENARIOS or friendly_name.lower() == 'static':
            should_exclude = True
            
        if not should_exclude:
            filtered_scenarios.append(s)
            
    all_scenarios = filtered_scenarios
    print(f"After filtering excluded scenarios: {all_scenarios}")
    
    # Display what we found
    print(f"Found {len(all_topologies)} topologies and {len(all_scenarios)} scenarios")
    print(f"Topologies: {all_topologies}")
    print(f"Scenarios: {[get_friendly_name(s, rewiring_modes.get(s, 'none')) for s in all_scenarios]}")
    
    # Determine grid layout: topologies as rows, scenarios as columns
    n_rows = len(all_topologies)
    n_cols = len(all_scenarios)

    # Create figure with tighter layout
    plt.figure(figsize=(17.8*cm, 10*cm))
    gs = GridSpec(n_rows, n_cols, figure=plt.gcf(), wspace=0.15, hspace=0.15)  # Reduced spacing
    
    # Find global min/max rates for consistent scale across all plots
    all_rates = []
    for topology, results in all_results.items():
        for scenario in all_scenarios:
            if scenario in results:
                for rates in results[scenario]['distributions'].values():
                    if rates and len(rates) > 0:
                        all_rates.extend(rates[:min(len(rates), 100)])  # Take samples from each
    
    if all_rates:
        rate_min = np.percentile(all_rates, 1)  # 1st percentile
        rate_max = np.percentile(all_rates, 99)  # 99th percentile
    else:
        rate_min, rate_max = 0, 1
    
    print(f"Global rate range: {rate_min:.2e} to {rate_max:.2e}")
    
    # Number of bins for the heatmap
    rate_bins = 20
    
    # Create custom colormap with better low-count visibility
    custom_cmap = sns.color_palette("viridis", as_cmap=True)
    
    # Create a single colorbar on the right side spanning all rows
    # First, get the last plotted image to use for the colorbar
    last_im = None
    
    # Process each topology and scenario
    for row_idx, topology in enumerate(all_topologies):
        topology_results = all_results.get(topology, {})
        topology_friendly = topology.upper()
        
        for col_idx, scenario in enumerate(all_scenarios):
            # Create subplot
            ax = plt.subplot(gs[row_idx, col_idx])
            
            # Handle empty data case - just leave empty white space
            if scenario not in topology_results or not topology_results[scenario].get('distributions'):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)  # Remove the box around the empty subplot
                continue
            
            scenario_data = topology_results[scenario]
            
            # Get friendly name and color
            rewiring = rewiring_modes.get(scenario, 'none')
            friendly_name = get_friendly_name(scenario, rewiring)
            color_key = friendly_name.lower()
            title_color = FRIENDLY_COLORS.get(color_key, 'black')
            
            # Get parameter values
            param_vals = scenario_data.get('param_values', [])
            
            # Create 2D histogram data
            hist_data = np.zeros((len(param_vals), rate_bins))
            rate_bin_edges = np.linspace(rate_min, rate_max, rate_bins + 1)
            
            # Fill histogram
            for j, val in enumerate(param_vals):
                if val in scenario_data['distributions']:
                    rates = scenario_data['distributions'][val]
                    if rates and len(rates) > 0:
                        hist, _ = np.histogram(rates, bins=rate_bin_edges)
                        if np.sum(hist) > 0:
                            # Use sqrt normalization to enhance low count visibility
                            hist = np.sqrt(hist / np.max(hist))
                        hist_data[j, :] = hist
            
            # Plot heatmap
            im = ax.imshow(
                hist_data.T,
                aspect='auto',
                origin='lower',
                extent=[0, len(param_vals)-1, rate_min, rate_max],
                cmap=custom_cmap
            )
            last_im = im  # Keep track of the last image for colorbar
            
            # Add horizontal line at y=0
            #ax.axhline(y=0, color='black', linestyle=':', linewidth=0.5, alpha=0.7)
            
            # Handle ticks and labels based on position
            # Set up x-ticks (show labels only on bottom row)
            x_ticks = np.arange(len(param_vals))
            ax.set_xticks(x_ticks[::max(1, len(param_vals) // 5)])
            
            if row_idx == n_rows - 1:  # Bottom row - show x-tick labels
                ax.set_xticklabels([f'{param_vals[i]:.1f}' for i in x_ticks[::max(1, len(param_vals) // 5)]], 
                                  fontsize=TICK_FONT_SIZE)
            else:
                ax.set_xticklabels([])  # Hide x-tick labels for non-bottom rows
            
            # Set up y-ticks (show labels only on leftmost column)
            
            y_ticks = get_clean_yticks(rate_min, rate_max)
            ax.set_yticks(y_ticks)
            ax.set_ylim(rate_min, rate_max)
            
            if col_idx == 0:  # Leftmost column - show y-tick labels
                ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=TICK_FONT_SIZE)
                
                # Add vertical topology label with adjusted position for tighter spacing
                ax.text(-0.35, 0.5, topology_friendly, 
                        transform=ax.transAxes,
                        rotation=90, 
                        fontsize=AXIS_LABEL_FONT_SIZE + 1, 
                        fontweight='bold',
                        va='center',
                        ha='center')
            else:
                ax.set_yticklabels([])  # Hide y-tick labels for non-leftmost columns
            
            # Add scenario label on top for first row
            if row_idx == 0:
                ax.set_title(friendly_name, color=title_color, 
                           fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=2)
            
             # Add horizontal reference line at y=0
            if rate_min <= 0 <= rate_max:
                ax.axhline(y=0, **ZERO_LINE_STYLE)
            
             # Add median line and absolute median line
            medians = []
            abs_medians = []
            
            for j, val in enumerate(param_vals):
                if val in scenario_data['distributions']:
                    rates = scenario_data['distributions'][val]
                    if rates and len(rates) > 0:
                        # Original median calculation (already in scenario_data['rates'])
                        medians.append((j, scenario_data['rates'][val]))
                        
                        # Calculate median of absolute values
                        abs_rates = [abs(r) for r in rates]
                        abs_median = np.median(abs_rates)
                        abs_medians.append((j, abs_median))
            
            if medians:
                x_med, y_med = zip(*medians)
                ax.plot(x_med, y_med, 'r-', linewidth=0.8)
                
                # Add absolute median line in blue
                x_abs, y_abs = zip(*abs_medians)
                ax.plot(x_abs, y_abs, '--', color = '#cb02f7',  linewidth=0.8)
    
    # Add a single colorbar on the right side of the entire figure
    if last_im:
        fig = plt.gcf()
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label('√(Count)', fontsize=LEGEND_FONT_SIZE)
        cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
        cbar.outline.set_linewidth(0.4)
    
    # Add centered axis labels - adjusted for tighter layout
    fig.text(0.5, 0.04, param_name, ha='center', fontsize=AXIS_LABEL_FONT_SIZE+1, fontweight='bold')
    fig.text(0.06, 0.5, 'Rate (×10³)', va='center', rotation='vertical', 
            fontsize=AXIS_LABEL_FONT_SIZE+1, fontweight='bold')
    
    # Create directory and save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    save_path = f'{OUTPUT_DIR}/convergence_rate_master_grid_{today}'
    
    for ext in ['pdf', 'png']:
        plt.savefig(f"{save_path}.{ext}", bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()
    print(f"Saved master grid heatmap to {save_path}")

def process_unified_file():
    """Process a single unified ALL file to build master grid view."""
    available_files = find_all_files()
    
    if not available_files:
        print("No unified data files found.")
        return
    
    # Print available files
    print("Available unified files:")
    for i, file in enumerate(available_files):
        print(f"{i}: {file}")
    
    # Ask user to select a file
    try:
        file_idx = int(input("Enter the index of the file to use: "))
        if file_idx < 0 or file_idx >= len(available_files):
            print(f"Invalid index. Using the most recent file (0).")
            file_idx = 0
    except ValueError:
        print("Invalid input. Using the most recent file (0).")
        file_idx = 0
    
    selected_file = available_files[file_idx]
    print(f"Selected file: {selected_file}")
    
    # Load unified file data
    data = load_unified_results(selected_file)
    
    if data:
        create_master_heatmap_grid(data)
    else:
        print("No valid data found to plot.")

def main():
    """Main execution function."""
    # Setup plotting style
    setup_plotting()
    
    # Process unified file to create master grid
    process_unified_file()

if __name__ == "__main__":
    main()