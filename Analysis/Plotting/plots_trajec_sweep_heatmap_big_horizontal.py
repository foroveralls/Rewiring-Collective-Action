#!/usr/bin/env python3
"""
Enhanced visualization script for previously processed convergence rate data.
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

# ====================== CONFIGURATION ======================
# Input directory with processed data
INPUT_DIR = "../../Output/ProcessedRates"

# Output directory for plots
OUTPUT_DIR = "../../Figs/ConvergenceRates"

# Plot settings - ADJUSTED FOR JOURNAL SPECIFICATIONS
BASE_FONT_SIZE = 6  # Reduced from 14 to fit journal specifications
RANDOM_SEED = 42

cm = 1/2.54

# Define font sizes for different elements
TITLE_FONT_SIZE = BASE_FONT_SIZE - 1  # Reduced scenario label size
AXIS_LABEL_FONT_SIZE = BASE_FONT_SIZE - 1
TICK_FONT_SIZE = BASE_FONT_SIZE - 4
LEGEND_FONT_SIZE = BASE_FONT_SIZE - 3  # Reduced colorbar label size
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE - 2

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
        'legend.title_fontsize': LEGEND_FONT_SIZE
    })
    sns.set_theme(font_scale=BASE_FONT_SIZE/12)
    sns.set(style="ticks")
    sns.set(rc={
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        "axes.grid": True,
        "grid.color": 'white',
        'grid.linestyle': 'solid', 
        "lines.linewidth": 0.8,
        "axes.edgecolor": "black",
        "patch.edgecolor": "black",
        "patch.linewidth": 0.2,
        "axes.spines.bottom": True,
        "grid.alpha": 0.8,
        "grid.linewidth": 0.3,
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
    
    Parameters:
    all_data - Dictionary with topology as key and data as value
    """
    # Get all unique topologies and scenarios
    all_topologies = list(all_data.keys())
    all_scenarios = set()
    all_rewiring_modes = {}
    
    for topology, data in all_data.items():
        for scenario in data['results'].keys():
            # Skip excluded scenarios
            scenario_key = f"{scenario}_{data['rewiring_modes'].get(scenario, 'none')}"
            if scenario_key not in EXCLUDED_SCENARIOS and len(data['results'][scenario]['distributions']) > 0:
                all_scenarios.add(scenario)
                all_rewiring_modes[scenario] = data['rewiring_modes'].get(scenario, 'none')
    
    all_scenarios = sorted(list(all_scenarios))
    
    # Display what we found
    print(f"Found {len(all_topologies)} topologies and {len(all_scenarios)} scenarios")
    print(f"Topologies: {all_topologies}")
    print(f"Scenarios: {[get_friendly_name(s, all_rewiring_modes.get(s, 'none')) for s in all_scenarios]}")
    
    # Determine grid layout: topologies as rows, scenarios as columns
    n_rows = len(all_topologies)
    n_cols = len(all_scenarios)

    # Create figure with constrained layout
    plt.figure()
    gs = GridSpec(n_rows, n_cols, figure=plt.gcf(), wspace=0.35, hspace=0.3)  # Increased wspace for better horizontal spacing
    
    # Find global min/max rates for consistent scale across all plots
    all_rates = []
    for topology, data in all_data.items():
        for scenario in all_scenarios:
            if scenario in data['results']:
                for rates in data['results'][scenario]['distributions'].values():
                    if len(rates) > 0:
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
    colors = sns.color_palette("viridis", n_colors=256)
    custom_cmap = sns.color_palette("viridis", as_cmap=True)
    
    # Process each topology and scenario
    for row_idx, topology in enumerate(all_topologies):
        data = all_data[topology]
        param_name = data['param_name']
        topology_friendly = topology.upper()
        
        for col_idx, scenario in enumerate(all_scenarios):
            if scenario not in data['results'] or len(data['results'][scenario]['distributions']) == 0:
                # Skip if no data for this scenario
                ax = plt.subplot(gs[row_idx, col_idx])
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=ANNOTATION_FONT_SIZE)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Get friendly name and color
            rewiring = data['rewiring_modes'].get(scenario, 'none')
            friendly_name = get_friendly_name(scenario, rewiring)
            title_color = FRIENDLY_COLORS.get(friendly_name, 'black')
            
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
                cbar.set_label('√(Count)', fontsize=LEGEND_FONT_SIZE)
                cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
            
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
                ax.set_title(friendly_name, color=title_color, fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=2)  # Reduced padding
            
            # Set x-ticks to parameter values with standardized format
            x_ticks = np.arange(len(param_vals))
            x_tick_labels = [f'{val:.1f}' for val in param_vals]  # Changed to 1 decimal place
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
                ax.plot(x_med, y_med, 'r-', linewidth=1.5)
    
    # Title removed as requested
    
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
# %%
    main()
