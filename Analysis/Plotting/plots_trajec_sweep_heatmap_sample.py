#!/usr/bin/env python3
"""
Visualization script for previously processed convergence rate data.
This script loads saved convergence rate data and generates heatmap visualizations.
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

# ====================== CONFIGURATION ======================
# Specify which topologies to visualize (leave empty to plot all available)
SELECTED_TOPOLOGIES = []  # Empty list means all available topologies will be plotted

# Specify which scenarios to include in visualization (leave empty to plot all available)
SELECTED_SCENARIOS = []  # Empty list means all available scenarios will be plotted

# Input directory with processed data
INPUT_DIR = "../../Output/ProcessedRates"

# Output directory for plots
OUTPUT_DIR = "../../Figs/ConvergenceRates"

# Plot settings
FONT_SIZE = 14
RANDOM_SEED = 42

# Scenario names and colors mapping
FRIENDLY_COLORS = {
    'static': '#EE7733',
    'random': '#0077BB',
    'local (similar)': '#33BBEE',
    'local (opposite)': '#009988',
    'bridge (similar)': '#CC3311',
    'bridge (opposite)': '#EE3377',
    'wtf': '#BBBBBB',
    'node2vec': '#44BB99'
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

# ====================== FUNCTIONS ======================

def setup_plotting():
    """Configure plotting style."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'figure.figsize': (20, 12),
        'figure.dpi': 300
    })
    sns.set_theme(font_scale=FONT_SIZE/12)
    sns.set_style("ticks")

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
        return parts[2]  # Based on naming convention: convergence_rates_TOPOLOGY_...
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

def create_heatmap(data, rewiring_modes):
    """Generate heatmap visualization for convergence rates."""
    topology = data['topology']
    results = data['results']
    positive_only = data.get('positive_only', False)
    
    print(f"Creating heatmap for {topology}")
    
    # Filter scenarios if specified
    if SELECTED_SCENARIOS:
        valid_scenarios = [s for s in results.keys() 
                         if s in SELECTED_SCENARIOS and results[s]['distributions']]
    else:
        valid_scenarios = [s for s in results.keys() 
                         if results[s]['distributions']]  # Only include scenarios with data
    
    if not valid_scenarios:
        print("No valid data to plot")
        return
    
    # Determine grid layout
    n_scenarios = len(valid_scenarios)
    n_cols = min(3, n_scenarios)
    n_rows = (n_scenarios + n_cols - 1) // n_cols
    
    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
    
    # Find global min/max rates for consistent scale
    all_rates = []
    for scenario in valid_scenarios:
        for rates in results[scenario]['distributions'].values():
            if len(all_rates) < 10000:  # Limit sample size for memory
                all_rates.extend(rates[:min(len(rates), 100)])  # Take at most 100 values from each
    
    if all_rates:
        rate_min = np.percentile(all_rates, 1)  # 1st percentile
        rate_max = np.percentile(all_rates, 99)  # 99th percentile
    else:
        rate_min, rate_max = 0, 1
    
    print(f"Rate range: {rate_min:.2f} to {rate_max:.2f}")
    
    # Number of bins for the heatmap
    rate_bins = 20
    
    # Note about which rates are included
    rate_mode_label = "Positive Rates Only" if positive_only else "All Rates"
    
    # Process each scenario
    for i, scenario in enumerate(valid_scenarios):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]
        
        # Get friendly name and color
        friendly_name = get_friendly_name(scenario, rewiring_modes.get(scenario, 'none'))
        title_color = FRIENDLY_COLORS.get(friendly_name, 'black')
        
        # Get data for this scenario
        scenario_data = results[scenario]
        param_vals = scenario_data['param_values']
        
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
                        hist = hist / np.sum(hist)  # Normalize
                    hist_data[j, :] = hist
        
        # Plot heatmap
        im = ax.imshow(
            hist_data.T,
            aspect='auto',
            origin='lower',
            extent=[0, len(param_vals)-1, rate_min, rate_max],
            cmap='viridis'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Count')
        
        # Set title and labels
        ax.set_title(friendly_name, color=title_color, fontsize=FONT_SIZE+2, fontweight='bold')
        ax.set_xlabel(f'{data["param_name"]}', fontsize=FONT_SIZE)
        ax.set_ylabel('Convergence Rate (×10³)', fontsize=FONT_SIZE)
        
        # Set x-ticks to parameter values
        x_ticks = np.arange(len(param_vals))
        x_tick_labels = [f'{val:.2f}' for val in param_vals]
        tick_spacing = max(1, len(param_vals) // 5)
        ax.set_xticks(x_ticks[::tick_spacing])
        ax.set_xticklabels(x_tick_labels[::tick_spacing])
        
        # Add median line
        medians = []
        for j, val in enumerate(param_vals):
            if val in scenario_data['rates']:
                medians.append((j, scenario_data['rates'][val]))
        
        if medians:
            x_med, y_med = zip(*medians)
            ax.plot(x_med, y_med, 'r-', linewidth=2, label='Median')
            ax.legend(loc='best')
    
    # Hide unused subplots
    for i in range(len(valid_scenarios), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axs[row, col].set_visible(False)
    
    # Add overall title with rate mode indication
    plt.suptitle(f'Convergence Rate Distributions - {topology.upper()} ({rate_mode_label})', 
                fontsize=FONT_SIZE+4, y=0.98, fontweight='bold')
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ['pdf', 'png']:
        rate_mode_suffix = "_positive" if positive_only else "_all"
        save_path = f'{OUTPUT_DIR}/convergence_rate_{topology}{rate_mode_suffix}.{ext}'
        plt.show()
        #plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.close()
    print(f"Saved heatmap for {topology}")
    gc.collect()

def get_file_to_process():
    """Get file to process from command line or user input."""
    available_files = list_available_files()
    
    if not available_files:
        print("No processed data files found in the input directory.")
        return None
    
    # If only one topology is selected, try to find matching file
    if len(SELECTED_TOPOLOGIES) == 1:
        matching_files = [f for f in available_files if SELECTED_TOPOLOGIES[0].lower() in f.lower()]
        if matching_files:
            print(f"Found matching file for topology {SELECTED_TOPOLOGIES[0]}: {matching_files[0]}")
            return matching_files[0]
    
    # If multiple files or no matching file, show options
    print("Available data files:")
    for i, file in enumerate(available_files):
        print(f"{i}: {file}")
    
    # If command line argument provided
    if len(sys.argv) > 1:
        try:
            file_index = int(sys.argv[1])
            if 0 <= file_index < len(available_files):
                return available_files[file_index]
        except ValueError:
            # Check if argument matches filename
            if sys.argv[1] in available_files:
                return sys.argv[1]
    
    # Otherwise prompt user
    try:
        user_input = input("Enter the index or name of the file to process: ")
        
        # Try as index
        try:
            file_index = int(user_input)
            if 0 <= file_index < len(available_files):
                return available_files[file_index]
        except ValueError:
            # Try as filename
            if user_input in available_files:
                return user_input
            
            # Try as partial match
            matches = [f for f in available_files if user_input.lower() in f.lower()]
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                print(f"Multiple matches found for '{user_input}':")
                for i, m in enumerate(matches):
                    print(f"{i}: {m}")
                try:
                    match_idx = int(input("Enter the index of the file to use: "))
                    if 0 <= match_idx < len(matches):
                        return matches[match_idx]
                except ValueError:
                    pass
        
        print("Invalid input. Using first file.")
        return available_files[0]
    
    except Exception as e:
        print(f"Error in file selection: {e}")
        if available_files:
            return available_files[0]
        return None

def process_all_files():
    """Process all available files or selected topologies."""
    available_files = list_available_files()
    
    if not available_files:
        print("No processed data files found in the input directory.")
        return
    
    # Filter by selected topologies if specified
    if SELECTED_TOPOLOGIES:
        files_to_process = []
        for topology in SELECTED_TOPOLOGIES:
            matching_files = [f for f in available_files if topology.lower() in f.lower()]
            files_to_process.extend(matching_files)
        
        if not files_to_process:
            print(f"No files found matching selected topologies: {SELECTED_TOPOLOGIES}")
            return
    else:
        files_to_process = available_files
    
    print(f"Processing {len(files_to_process)} files")
    
    for file in files_to_process:
        print(f"\nProcessing file: {file}")
        data = load_results(file)
        
        if data and 'results' in data:
            # Extract rewiring modes from data
            rewiring_modes = {}
            for scenario in data['results']:
                # Use most common rewiring value for each scenario
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
            
            create_heatmap(data, rewiring_modes)
        else:
            print(f"Error: Invalid data format in {file}")

def main():
    """Main execution function."""
    # Setup plotting style
    setup_plotting()
    
    # Process all files or select specific file
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'all':
        process_all_files()
    else:
        filename = get_file_to_process()
        if filename:
            data = load_results(filename)
            if data and 'results' in data:
                # Extract rewiring modes from results
                rewiring_modes = {}
                for scenario in data['results']:
                    if scenario.lower() in ['none', 'random', 'wtf', 'node2vec']:
                        rewiring_modes[scenario] = 'none'
                    else:
                        # Default to 'diff' unless filename indicates 'same'
                        if 'same' in filename.lower():
                            rewiring_modes[scenario] = 'same'
                        else:
                            rewiring_modes[scenario] = 'diff'
                
                create_heatmap(data, rewiring_modes)
            else:
                print(f"Error: Invalid data format in {filename}")

if __name__ == "__main__":
    main()