
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

# Excluded scenarios (static in this case)
EXCLUDED_SCENARIOS = ['none_none', 'static', 'none']

# ====================== FUNCTIONS ======================

def setup_plotting():
    """Configure plotting style using settings from heatmap_alternate."""
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
        'legend.title_fontsize': LEGEND_FONT_SIZE,
        'xtick.major.width': 0.8,    # Thinner x ticks
        'ytick.major.width': 0.8,   # Thinner y ticks
        'axes.linewidth': 0.8,       # Thinner plot borders
    })
    sns.set_theme(font_scale=BASE_FONT_SIZE/12)
    sns.set(style="ticks")
    sns.set(rc={
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'xtick.major.size': 2,  # Shorter x ticks (default is 3.5)
        'ytick.major.size': 2,  # Shorter y ticks
        "axes.grid": True,
        "grid.color": 'white',
        'grid.linestyle': 'solid', 
        "lines.linewidth": 0.7,      # Thinner lines in general
        "axes.edgecolor": "black",
        "patch.edgecolor": "black",
        "patch.linewidth": 0.2,
        "axes.spines.bottom": True,
        "grid.alpha": 0.8,
        "axes.linewidth": 0.8,       # Thinner plot borders
        "grid.linewidth": 0.2,       # Thinner grid lines
        "xtick.bottom": True,
        "ytick.left": True
    })

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
    
    # Create a set of all scenarios across all topologies
    all_scenarios = set()
    for topology, results in all_results.items():
        all_scenarios.update(results.keys())
    
    # Extract rewiring modes for friendly names
    rewiring_modes = extract_rewiring_modes(all_results)
    
    # Debug info to help diagnose issues
    #print_debug_info(all_results, rewiring_modes)
    
    # Handle excluded scenarios - make sure to check both the scenario name and any standard name mappings
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

    # Create figure with constrained layout
    plt.figure()
    gs = GridSpec(n_rows, n_cols, figure=plt.gcf(), wspace=0.35, hspace=0.3)
    
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
    
    # Process each topology and scenario
    for row_idx, topology in enumerate(all_topologies):
        topology_results = all_results.get(topology, {})
        topology_friendly = topology.upper()
        
        for col_idx, scenario in enumerate(all_scenarios):
            if scenario not in topology_results:
                # Skip if no data for this scenario in this topology
                ax = plt.subplot(gs[row_idx, col_idx])
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=ANNOTATION_FONT_SIZE)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            scenario_data = topology_results[scenario]
            
            # Skip if no distributions in scenario data
            if not scenario_data.get('distributions') or len(scenario_data['distributions']) == 0:
                ax = plt.subplot(gs[row_idx, col_idx])
                ax.text(0.5, 0.5, "No distributions", ha='center', va='center', fontsize=ANNOTATION_FONT_SIZE)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Get friendly name and color
            rewiring = rewiring_modes.get(scenario, 'none')
            friendly_name = get_friendly_name(scenario, rewiring)
            color_key = friendly_name.lower()
            title_color = FRIENDLY_COLORS.get(color_key, 'black')
            
            # Get parameter values
            param_vals = scenario_data.get('param_values', [])
            
            # Create subplot
            ax = plt.subplot(gs[row_idx, col_idx])
            
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
            
            # Add colorbar for the right-most column
            if col_idx == n_cols - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('√(Count)', fontsize=LEGEND_FONT_SIZE)
                cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
                # Make colorbar outline thinner
                cbar.outline.set_linewidth(0.4)
            
            # Labels
            # Add topology label on left side for first column
            if col_idx == 0:
                # Add vertical topology label
                ax.text(-0.85, 0.5, topology_friendly, 
                        transform=ax.transAxes,
                        rotation=90, 
                        fontsize=AXIS_LABEL_FONT_SIZE + 1, 
                        fontweight='bold',
                        va='center',
                        ha='center')
                
                # Add y-axis label
                ax.set_ylabel('Rate (×10³)', fontsize=AXIS_LABEL_FONT_SIZE)
            
            # Add scenario label on top for first row
            if row_idx == 0:
                friendly_name = get_friendly_name(scenario, rewiring_modes.get(scenario, 'none'))
                color_key = friendly_name.lower()
                title_color = FRIENDLY_COLORS.get(color_key, 'black')
                ax.set_title(friendly_name, color=title_color, 
                           fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=2)
            
            # Set x-ticks to parameter values with standardized format
            x_ticks = np.arange(len(param_vals))
            x_tick_labels = [f'{val:.1f}' for val in param_vals]
            tick_spacing = max(1, len(param_vals) // 5)
            ax.set_xticks(x_ticks[::tick_spacing])
            ax.set_xticklabels(x_tick_labels[::tick_spacing], fontsize=TICK_FONT_SIZE)
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
            
            # Bottom row gets parameter name as x-label
            if row_idx == n_rows - 1:
                ax.set_xlabel(param_name, fontsize=AXIS_LABEL_FONT_SIZE)
            
            # Add median line
            medians = []
            for j, val in enumerate(param_vals):
                if val in scenario_data['rates']:
                    medians.append((j, scenario_data['rates'][val]))
            
            if medians:
                x_med, y_med = zip(*medians)
                ax.plot(x_med, y_med, 'r-', linewidth=0.8)
    
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