#!/usr/bin/env python3
"""
Enhanced visualization script for previously processed convergence rate data.
Creates a comprehensive grid display with scenarios on rows and topologies on columns
to facilitate direct comparison of algorithm performance across network structures.
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

# ====================== CONFIGURATION ======================
# Input directory with processed data
INPUT_DIR = "../../Output/ProcessedRates"

# Output directory for plots
OUTPUT_DIR = "../../Figs/ConvergenceRates"

# Plot settings
FONT_SIZE = 14
RANDOM_SEED = 42

# Scenario names and colors mapping (from heatmap_alternate)
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
EXCLUDED_SCENARIOS = ['none_none']

# ====================== FUNCTIONS ======================

def setup_plotting():
    """Configure plotting style using settings from heatmap_alternate."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.dpi': 300
    })
    sns.set_theme(font_scale=FONT_SIZE/12)
    sns.set(style="ticks")
    sns.set(rc={
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        "axes.grid": True,
        "grid.color": 'white',
        'grid.linestyle': 'solid', 
        "axes.edgecolor": "black",
        "patch.edgecolor": "black",
        "patch.linewidth": 0.5,
        "axes.spines.bottom": True,
        "grid.alpha": 0.8,
        "xtick.bottom": True,
        "ytick.left": True
    })

def get_friendly_name(scenario, rewiring):
    """Get user-friendly algorithm name."""
    if scenario is None:
        return "Unknown"
    
    if rewiring is None:
        rewiring = "none"
    
    scenario = str(scenario).lower()
    rewiring = str(rewiring).lower()
    
    key = f"{scenario}_{rewiring}"
    if scenario in ["none", "random", "wtf", "node2vec"]:
        key = f"{scenario}_none"
    
    return FRIENDLY_NAMES.get(key, f"{scenario} ({rewiring})")

def list_available_files():
    """List all available pickle files with processed data."""
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        return []
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pkl') and f.startswith('convergence_rates_')]
    return files

def extract_topology_from_filename(filename):
    """Extract topology from filename."""
    parts = filename.split('_')
    if len(parts) > 2:
        return parts[2].upper()  # Based on naming convention: convergence_rates_TOPOLOGY_...
    return None

def load_results(filename):
    """Load processed results from pickle file."""
    filepath = os.path.join(INPUT_DIR, filename)
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def create_master_heatmap_grid(all_data):
    """
    Create a comprehensive grid of heatmaps showing all scenarios across all topologies.
    Scenarios are on rows, topologies are on columns to facilitate algorithm comparison.
    
    Parameters:
    all_data - Dictionary with topology as key and data as value
    """
    # Get all unique topologies and scenarios
    all_topologies = list(all_data.keys())
    all_scenarios = set()
    all_rewiring_modes = {}
    all_param_names = set()
    
    for topology, data in all_data.items():
        all_param_names.add(data['param_name'])
        for scenario in data['results'].keys():
            # Skip excluded scenarios
            scenario_key = f"{scenario}_{data['rewiring_modes'].get(scenario, 'none')}"
            if scenario_key not in EXCLUDED_SCENARIOS and len(data['results'][scenario]['distributions']) > 0:
                all_scenarios.add(scenario)
                all_rewiring_modes[scenario] = data['rewiring_modes'].get(scenario, 'none')
    
    # Convert scenarios to friendly names for sorting
    friendly_scenario_map = {s: get_friendly_name(s, all_rewiring_modes.get(s, 'none')) for s in all_scenarios}
    all_scenarios = sorted(list(all_scenarios), key=lambda s: friendly_scenario_map[s])
    
    # Handle possible different parameter names
    param_name = list(all_param_names)[0] if len(all_param_names) == 1 else "Parameter"
    
    # Display what we found
    print(f"Found {len(all_topologies)} topologies and {len(all_scenarios)} scenarios")
    print(f"Topologies: {all_topologies}")
    print(f"Scenarios: {[friendly_scenario_map[s] for s in all_scenarios]}")
    
    # Determine grid layout: scenarios as rows, topologies as columns (REVERSED FROM PREVIOUS VERSION)
    n_rows = len(all_scenarios)
    n_cols = len(all_topologies)
    
    # Create figure with constrained layout
    plt.figure(figsize=(n_cols*4, n_rows*3.5))
    gs = GridSpec(n_rows, n_cols, figure=plt.gcf(), wspace=0.25, hspace=0.3)
    
    # Find global min/max rates for consistent scale across all plots
    all_rates = []
    for topology, data in all_data.items():
        for scenario in all_scenarios:
            if scenario in data['results']:
                for rates in data['results'][scenario]['distributions'].values():
                    if len(rates) > 0:
                        # Take samples from each to avoid memory issues
                        sample_size = min(len(rates), 100)
                        all_rates.extend(np.random.choice(rates, sample_size).tolist())
    
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
    
    # Process each scenario and topology (reversed order from previous version)
    for row_idx, scenario in enumerate(all_scenarios):
        friendly_name = friendly_scenario_map[scenario]
        title_color = FRIENDLY_COLORS.get(friendly_name, 'black')
        
        for col_idx, topology in enumerate(all_topologies):
            data = all_data[topology]
            
            # Skip if no data for this scenario-topology combination
            if scenario not in data['results'] or len(data['results'][scenario]['distributions']) == 0:
                ax = plt.subplot(gs[row_idx, col_idx])
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Get data for this scenario
            scenario_data = data['results'][scenario]
            param_vals = scenario_data['param_values']
            
            # Create subplot
            ax = plt.subplot(gs[row_idx, col_idx])
            
            # Create 2D histogram data
            hist_data = np.zeros((len(param_vals), rate_bins))
            rate_bin_edges = np.linspace(rate_min, rate_max, rate_bins + 1)
            
            # Fill histogram
            for j, val in enumerate(param_vals):
                if val in scenario_data['distributions']:
                    rates = scenario_data['distributions'][val]
                    if rates:
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
                cbar.set_label('√(Count)', fontsize=FONT_SIZE-2)
            
            # Labels
            # Add scenario label on left side for first column
            if col_idx == 0:
                # Add vertical scenario label with color
                ax.text(-0.6, 0.5, friendly_name, 
                        transform=ax.transAxes,
                        rotation=90, 
                        fontsize=FONT_SIZE+1, 
                        fontweight='bold',
                        va='center',
                        ha='center',
                        color=title_color)
                
                # Add y-axis label
                ax.set_ylabel('Convergence Rate (×10³)', fontsize=FONT_SIZE-1)
            
            # Add topology label on top for first row
            if row_idx == 0:
                ax.set_title(topology, fontsize=FONT_SIZE+1, fontweight='bold')
            
            # Set x-ticks to parameter values
            x_ticks = np.arange(len(param_vals))
            x_tick_labels = [f'{val:.2f}' for val in param_vals]
            tick_spacing = max(1, len(param_vals) // 5)
            ax.set_xticks(x_ticks[::tick_spacing])
            ax.set_xticklabels(x_tick_labels[::tick_spacing], fontsize=FONT_SIZE-2)
            
            # Bottom row gets parameter name as x-label
            if row_idx == n_rows - 1:
                ax.set_xlabel(param_name, fontsize=FONT_SIZE-1)
            
            # Add median line
            medians = []
            for j, val in enumerate(param_vals):
                if val in scenario_data['rates']:
                    medians.append((j, scenario_data['rates'][val]))
            
            if medians:
                x_med, y_med = zip(*medians)
                ax.plot(x_med, y_med, 'r-', linewidth=1.5)
    
    # Add overall title
    plt.suptitle('Convergence Rate Distributions Across Rewiring Algorithms and Network Topologies', 
                fontsize=FONT_SIZE+4, y=0.995, fontweight='bold')
    
    # Create directory and save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    save_path = f'{OUTPUT_DIR}/convergence_rate_master_grid_{today}'
    
    for ext in ['pdf', 'png']:
        plt.savefig(f"{save_path}.{ext}", bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()
    print(f"Saved master grid heatmap to {save_path}")

def process_all_files():
    """Process all available files to build master grid view."""
    available_files = list_available_files()
    
    if not available_files:
        print("No processed data files found.")
        return
    
    # Load data from all files
    all_data = {}
    
    for file in available_files:
        print(f"Loading file: {file}")
        topology = extract_topology_from_filename(file)
        if not topology:
            print(f"  Warning: Could not extract topology from {file}")
            continue
        
        data = load_results(file)
        if not data or 'results' not in data:
            print(f"  Error: Invalid data format in {file}")
            continue
        
        # Extract rewiring modes
        rewiring_modes = {}
        for scenario in data['results']:
            if scenario.lower() in ['none', 'random', 'wtf', 'node2vec']:
                rewiring_modes[scenario] = 'none'
            elif 'biased' in scenario.lower():
                if 'diff' in file.lower():
                    rewiring_modes[scenario] = 'diff'
                else:
                    rewiring_modes[scenario] = 'same'
            elif 'bridge' in scenario.lower():
                if 'diff' in file.lower():
                    rewiring_modes[scenario] = 'diff'
                else:
                    rewiring_modes[scenario] = 'same'
        
        data['rewiring_modes'] = rewiring_modes
        all_data[topology] = data
    
    # Create master grid
    if all_data:
        create_master_heatmap_grid(all_data)
    else:
        print("No valid data found to plot.")

def main():
    """Main execution function."""
    # Setup plotting style
    setup_plotting()
    
    # Process all files to create master grid
    process_all_files()

if __name__ == "__main__":
    main()