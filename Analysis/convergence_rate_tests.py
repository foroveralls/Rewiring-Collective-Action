"""
Enhanced Convergence Rate Calculation Test Script
------------------------------------------------
Visualizes and compares different methods for calculating convergence rates.

Features:
- Inflection method visualization with adjustable parameters
- Time-to-threshold method visualization
- Method comparison plots
- Correlation analysis between methods
- Horizontal grouping by topology
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

#-------------------------
# CONFIGURATION SETTINGS (edit these variables to control the script)
#-------------------------

Modes = ['inflection', 'threshold', 'compare', 'correlation', 'parameter_test', 'horizontal_compare']
for i, mode in enumerate(Modes):
    print(f"{i}: {mode}")
    

mode_index = int(input("Enter mode index"))

MODE = Modes[mode_index]

file_list = [f for f in os.listdir("../Output") if f.endswith(".csv") and "default_run_avg" in f]

if not file_list:
    print("No suitable files found in the Output directory.")
    exit()

# Print file list with indices
for i, file in enumerate(file_list):
    print(f"{i}: {file}")

# Get user input for file selection
file_index = int(input("Enter the index of the file you want to plot: "))
 
data_path = os.path.join("../Output", file_list[file_index])

# Path to the data file
DATA_FILE = data_path

# Filter options (set to None to include all)
TOPOLOGY_FILTER = None  # e.g., 'Twitter', 'DPAH', 'cl', 'FB'
SCENARIO_FILTER = None  # e.g., 'random_none', 'biased_same'

# Output path (set to None to just display)
OUTPUT_PATH = f'../Figs/Convergence/Tests/convergence_test_{MODE}.png'

# Inflection method parameters (for parameter testing)
GAUSSIAN_SIGMA = 300  # Smoothing parameter
MIN_INFLECTION_IDX = 5000  # Minimum index to look for inflection
REGRESSION_WINDOW = 15  # Window size for linear regression around inflection

FONT_SIZE = 14
cm = 1/2.54

# Color mapping for scenarios
PLOT_COLORS = {
    'none_none': '#EE7733',    # Orange
    'random_none': '#0077BB',  # Blue
    'biased_same': '#33BBEE',  # Cyan
    'biased_diff': '#009988',  # Teal
    'bridge_same': '#CC3311',  # Red
    'bridge_diff': '#EE3377',  # Magenta
    'wtf_none': '#BBBBBB',     # Grey
    'node2vec_none': '#44BB99' # Blue-green
}

# Friendly names for display
friendly_names = {
    'none_none': 'static',
    'random_none': 'random',
    'biased_same': 'local (similar)',
    'biased_diff': 'local (opposite)',
    'bridge_same': 'bridge (similar)',
    'bridge_diff': 'bridge (opposite)',
    'wtf_none': 'wtf',
    'node2vec_none': 'node2vec'
}

#-------------------------
# FUNCTIONS
#-------------------------

def setup_plotting_style():
    """Set consistent style elements for all plots"""
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.0,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (20*cm, 15*cm), 
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out'
    })

def load_and_prepare_data(data_path, topology_filter=None, scenario_filter=None):
    """Load and prepare data with optional filtering"""
    raw_data = pd.read_csv(data_path)
    
    # Prepare data for analysis
    id_vars = ['t', 'scenario', 'rewiring', 'type']
    raw_data['rewiring'] = raw_data['rewiring'].fillna('none')
    raw_data['scenario'] = raw_data['scenario'].fillna('none')
    
    # Create scenario_grouped column
    raw_data['scenario_grouped'] = raw_data['scenario'].str.cat(
        raw_data['rewiring'], sep='_')
    
    # Melt the dataframe
    melted_data = pd.melt(raw_data, id_vars=id_vars + ['scenario_grouped'], 
                         value_vars=['avg_state', 'std_states'], 
                         var_name='measurement', value_name='value')
    
    # Apply filters if specified
    if topology_filter:
        melted_data = melted_data[melted_data['type'] == topology_filter]
    
    if scenario_filter:
        melted_data = melted_data[melted_data['scenario_grouped'] == scenario_filter]
    
    return melted_data

def find_inflection(seq, min_idx=MIN_INFLECTION_IDX, sigma=GAUSSIAN_SIGMA):
    """Calculate inflection point in trajectory"""
    smooth = gaussian_filter1d(seq, sigma)
    d2 = np.gradient(np.gradient(smooth))
    infls = np.where(np.diff(np.sign(d2)))[0]
    
    # Find first inflection point after min_idx
    inf_ind = None
    for i in infls:
        if i >= min_idx:
            inf_ind = i
            break
    
    if inf_ind is None and len(infls) > 0:
        # If no inflection after min_idx, take the last one
        inf_ind = infls[-1]
    
    return inf_ind, smooth, d2

def estimate_convergence_rate(trajec, loc=None, regwin=REGRESSION_WINDOW):
    """Estimate convergence rate around specified location"""
    x = np.arange(len(trajec) - 1)
    y = trajec
    
    if loc is not None:
        # Ensure we don't go out of bounds
        start_idx = max(0, loc-regwin)
        end_idx = min(len(trajec)-1, loc+regwin+1)
        x = x[start_idx:end_idx]
        y = trajec[start_idx:end_idx]
    
    # Linear regression
    n = np.size(x) 
    mx, my = np.mean(x), np.mean(y) 
    ssxy = np.sum(y*x) - n*my*mx 
    ssxx = np.sum(x*x) - n*mx*mx 
    
    if ssxx == 0:
        return 0, 0, 0
        
    b1 = ssxy / ssxx 
    b0 = my - b1*mx 
    
    rate = -b1/(trajec[loc]-1) if trajec[loc] != 1 else -b1/0.001
    return rate, b0, b1

def calculate_time_to_threshold(times, trajectory, thresholds):
    """Calculate times to reach specific thresholds"""
    times_to_threshold = []
    
    # Find when trajectory crosses each threshold
    for threshold in thresholds:
        # Find the first time the trajectory exceeds the threshold
        threshold_crossed = np.where(trajectory >= threshold)[0]
        if len(threshold_crossed) > 0:
            times_to_threshold.append(times[threshold_crossed[0]])
        else:
            # If threshold never reached, use last time point
            times_to_threshold.append(times[-1])
    
    return times_to_threshold

def demonstrate_inflection_method(data, output_path=None):
    """Visualize the inflection point based method for calculating convergence rates"""
    setup_plotting_style()
    
    # Get unique combinations of topologies and scenarios
    topologies = data['type'].unique()
    scenarios = data['scenario_grouped'].unique()
    
    # Create a grid of plots based on unique values
    n_rows = len(topologies)
    n_cols = len(scenarios) # Limit to 3 columns
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*8, n_rows*6), 
                            sharex=True, sharey=True)
    
    # Make axes 2D if it's 1D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Create a separate figure for legend
    legend_fig = plt.figure(figsize=(10, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis('off')
    
    legend_handles = []
    
    for t_idx, topology in enumerate(topologies):
        for s_idx, scenario in enumerate(scenarios[:n_cols]):  # Limit to n_cols
            ax = axes[t_idx, s_idx]
            
            # Filter data for this topology and scenario
            filtered_data = data[(data['type'] == topology) & 
                                (data['scenario_grouped'] == scenario) &
                                (data['measurement'] == 'avg_state')]
            
            if filtered_data.empty:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                continue
            
            # Get trajectory data
            times = filtered_data['t'].values
            trajectory = filtered_data['value'].values
            
            # Find inflection point
            inf_idx, smoothed, d2 = find_inflection(trajectory, 
                                                 sigma=GAUSSIAN_SIGMA, 
                                                 min_idx=MIN_INFLECTION_IDX)
            if inf_idx is None:
                inf_idx = len(trajectory) // 2  # Default to middle if no inflection found
            
            # Calculate convergence rate
            rate, b0, b1 = estimate_convergence_rate(smoothed, inf_idx, regwin=REGRESSION_WINDOW)
            
            # Original trajectory
            ax.plot(times, trajectory, '-', color='lightgray', label='Original')
            
            # Smoothed trajectory
            ax.plot(times, smoothed, '-', color=PLOT_COLORS.get(scenario, 'blue'),
                  linewidth=2.5, label='Smoothed')
            
            # Mark inflection point
            ax.plot(times[inf_idx], smoothed[inf_idx], 'o', color='red', markersize=8)
            
            # Plot linear fit around inflection point
            fit_start = max(0, inf_idx - REGRESSION_WINDOW)
            fit_end = min(len(times), inf_idx + REGRESSION_WINDOW + 1)
            fit_x = times[fit_start:fit_end]
            fit_y = b0 + b1 * np.arange(fit_start, fit_end)
            ax.plot(fit_x, fit_y, '--', color='black', linewidth=2)
            
            # Add text with calculated rate
            friendly_scenario = friendly_names.get(scenario, scenario)
            ax.text(0.05, 0.95, f"{friendly_scenario} on {topology}\nRate: {rate*1000:.4f}×10³",
                  transform=ax.transAxes, va='top', ha='left', 
                  bbox=dict(facecolor='white', alpha=0.8))
            
            # Configure subplot
            ax.set_title(f"{friendly_scenario} - {topology}")
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add to legend handles for the common legend
            if t_idx == 0 and s_idx == 0:
                legend_handles.extend([
                    plt.Line2D([], [], color='lightgray', label='Original Trajectory'),
                    plt.Line2D([], [], color='blue', linewidth=2.5, label='Smoothed Trajectory'),
                    plt.Line2D([], [], marker='o', color='red', linestyle='None', 
                              markersize=8, label='Inflection Point'),
                    plt.Line2D([], [], color='black', linestyle='--', linewidth=2, 
                              label='Linear Fit for Rate')
                ])
    
    # Add common y-axis label
    fig.text(0.04, 0.5, 'Average State', va='center', rotation='vertical', fontsize=14)
    
    # Add common x-axis label
    fig.text(0.5, 0.04, 'Time [timestep / system size]', ha='center', fontsize=14)
    
    # Add title
    fig.suptitle(f'Inflection Point Method (sigma={GAUSSIAN_SIGMA}, window={REGRESSION_WINDOW})', fontsize=16)
    
    # Add common legend
    legend_ax.legend(handles=legend_handles, loc='center', ncol=4, frameon=True)
    
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def demonstrate_threshold_method(data, output_path=None):
    """Visualize the time-to-threshold method for calculating convergence rates"""
    setup_plotting_style()
    
    # Get unique combinations of topologies and scenarios
    topologies = data['type'].unique()
    scenarios = data['scenario_grouped'].unique()
    
    # Create a grid of plots based on unique values
    n_rows = len(topologies)
    n_cols = min(3, len(scenarios))  # Limit to 3 columns
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*8, n_rows*6), 
                            sharex=True, sharey=True)
    
    # Make axes 2D if it's 1D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Create a separate figure for legend
    legend_fig = plt.figure(figsize=(10, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis('off')
    
    legend_handles = []
    
    for t_idx, topology in enumerate(topologies):
        for s_idx, scenario in enumerate(scenarios[:n_cols]):  # Limit to n_cols
            ax = axes[t_idx, s_idx]
            
            # Filter data for this topology and scenario
            filtered_data = data[(data['type'] == topology) & 
                                (data['scenario_grouped'] == scenario) &
                                (data['measurement'] == 'avg_state')]
            
            if filtered_data.empty:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                continue
            
            # Get trajectory data
            times = filtered_data['t'].values
            trajectory = filtered_data['value'].values
            
            # Apply light smoothing
            smoothed = gaussian_filter1d(trajectory, sigma=100)
            
            # Get starting and final values
            start_val = smoothed[0]
            final_val = smoothed[-1]
            
            # Calculate range of values
            value_range = final_val - start_val
            
            # Define thresholds (25%, 50%, 75% of total change)
            thresholds = [start_val + value_range * fraction for fraction in [0.25, 0.5, 0.75]]
            
            # Calculate times to reach thresholds
            times_to_threshold = calculate_time_to_threshold(times, smoothed, thresholds)
            
            # Calculate rate
            if len(times_to_threshold) > 0:
                mean_time = np.mean(times_to_threshold)
                rate = (value_range / mean_time) * 1000  # Scale for readability
            else:
                rate = 0
            
            # Plot original and smoothed trajectories
            ax.plot(times, trajectory, '-', color='lightgray', label='Original')
            ax.plot(times, smoothed, '-', color=PLOT_COLORS.get(scenario, 'blue'),
                  linewidth=2.5, label='Smoothed')
            
            # Plot thresholds and times
            threshold_colors = ['green', 'red', 'purple']
            for i, (threshold, time_point) in enumerate(zip(thresholds, times_to_threshold)):
                # Horizontal line for threshold
                ax.axhline(y=threshold, color=threshold_colors[i], linestyle='--', alpha=0.7)
                
                # Vertical line for time point
                ax.axvline(x=time_point, color=threshold_colors[i], linestyle=':', alpha=0.7)
                
                # Mark intersection
                ax.plot(time_point, threshold, 'o', color=threshold_colors[i], markersize=8)
            
            # Add text with calculated rate
            friendly_scenario = friendly_names.get(scenario, scenario)
            ax.text(0.05, 0.95, f"{friendly_scenario} on {topology}\nRate: {rate:.4f}×10³",
                  transform=ax.transAxes, va='top', ha='left', 
                  bbox=dict(facecolor='white', alpha=0.8))
            
            # Configure subplot
            ax.set_title(f"{friendly_scenario} - {topology}")
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add to legend handles for the common legend
            if t_idx == 0 and s_idx == 0:
                legend_handles.extend([
                    plt.Line2D([], [], color='lightgray', label='Original Trajectory'),
                    plt.Line2D([], [], color='blue', linewidth=2.5, label='Smoothed Trajectory'),
                    plt.Line2D([], [], color='green', linestyle='--', alpha=0.7, label='25% Threshold'),
                    plt.Line2D([], [], color='red', linestyle='--', alpha=0.7, label='50% Threshold'),
                    plt.Line2D([], [], color='purple', linestyle='--', alpha=0.7, label='75% Threshold'),
                    plt.Line2D([], [], marker='o', color='green', linestyle='None', markersize=8, label='25% Crossing'),
                    plt.Line2D([], [], marker='o', color='red', linestyle='None', markersize=8, label='50% Crossing'),
                    plt.Line2D([], [], marker='o', color='purple', linestyle='None', markersize=8, label='75% Crossing')
                ])
    
    # Add common y-axis label
    fig.text(0.04, 0.5, 'Average State', va='center', rotation='vertical', fontsize=14)
    
    # Add common x-axis label
    fig.text(0.5, 0.04, 'Time [timestep / system size]', ha='center', fontsize=14)
    
    # Add title
    fig.suptitle('Time-to-Threshold Method for Convergence Rate Calculation', fontsize=16)
    
    # Add common legend
    legend_ax.legend(handles=legend_handles, loc='center', ncol=4, frameon=True)
    
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        legend_fig.savefig(output_path.replace('.png', '_legend.png'), 
                         dpi=300, bbox_inches='tight')
    
    plt.show()

def compare_methods(data, output_path=None):
    """Compare inflection-based and time-to-threshold methods side by side"""
    setup_plotting_style()
    
    # Get unique combinations of topologies and scenarios
    topologies = data['type'].unique()
    scenarios = data['scenario_grouped'].unique()
    
    # Limit number of scenarios to display
    max_scenarios = min(4, len(scenarios))
    selected_scenarios = scenarios[:max_scenarios]
    
    # Create a figure with a grid of subplots
    # We need (topologies × scenarios) rows and 2 columns (one for each method)
    fig, axes = plt.subplots(len(topologies) * max_scenarios, 2, 
                           figsize=(12, 6 * len(topologies) * max_scenarios),
                           sharex='col', sharey='row')
    
    # If we only have one topology and one scenario, make sure axes is 2D
    if len(topologies) * max_scenarios == 1:
        axes = axes.reshape(1, 2)
    
    # Iterate through topologies and scenarios
    for idx, (topology, scenario) in enumerate([(t, s) for t in topologies for s in selected_scenarios]):
        # Filter data for this topology and scenario
        filtered_data = data[(data['type'] == topology) & 
                           (data['scenario_grouped'] == scenario) &
                           (data['measurement'] == 'avg_state')]
        
        if filtered_data.empty:
            continue
        
        # Get trajectory data
        times = filtered_data['t'].values
        trajectory = filtered_data['value'].values
        
        # Apply smoothing for both methods
        inflection_smoothed = gaussian_filter1d(trajectory, GAUSSIAN_SIGMA)  # Heavy smoothing for inflection
        threshold_smoothed = gaussian_filter1d(trajectory, 100)   # Light smoothing for threshold
        
        # --- INFLECTION METHOD (LEFT COLUMN) ---
        ax_inf = axes[idx, 0]
        
        # Find inflection point
        inf_idx, _, _ = find_inflection(trajectory, 
                                     sigma=GAUSSIAN_SIGMA, 
                                     min_idx=MIN_INFLECTION_IDX)
        if inf_idx is None:
            inf_idx = len(trajectory) // 2
        
        # Calculate convergence rate using inflection method
        inf_rate, b0, b1 = estimate_convergence_rate(inflection_smoothed, 
                                                 inf_idx, 
                                                 regwin=REGRESSION_WINDOW)
        
        # Plot for inflection method
        ax_inf.plot(times, trajectory, '-', color='lightgray', label='Original')
        ax_inf.plot(times, inflection_smoothed, '-', color=PLOT_COLORS.get(scenario, 'blue'),
                  linewidth=2.5, label='Smoothed')
        
        # Mark inflection point
        ax_inf.plot(times[inf_idx], inflection_smoothed[inf_idx], 'o', color='red', markersize=8)
        
        # Plot linear fit around inflection point
        fit_start = max(0, inf_idx - REGRESSION_WINDOW)
        fit_end = min(len(times), inf_idx + REGRESSION_WINDOW + 1)
        fit_x = times[fit_start:fit_end]
        fit_y = b0 + b1 * np.arange(fit_start, fit_end)
        ax_inf.plot(fit_x, fit_y, '--', color='black', linewidth=2)
        
        # Add title and rate
        friendly_scenario = friendly_names.get(scenario, scenario)
        ax_inf.set_title(f"Inflection Method\n{friendly_scenario} - {topology}")
        ax_inf.text(0.05, 0.95, f"Rate: {inf_rate*1000:.4f}×10³",
                  transform=ax_inf.transAxes, va='top', ha='left', 
                  bbox=dict(facecolor='white', alpha=0.8))
        
        # --- THRESHOLD METHOD (RIGHT COLUMN) ---
        ax_thresh = axes[idx, 1]
        
        # Get starting and final values
        start_val = threshold_smoothed[0]
        final_val = threshold_smoothed[-1]
        
        # Calculate range of values
        value_range = final_val - start_val
        
        # Define thresholds (25%, 50%, 75% of total change)
        thresholds = [start_val + value_range * fraction for fraction in [0.25, 0.5, 0.75]]
        
        # Calculate times to reach thresholds
        times_to_threshold = calculate_time_to_threshold(times, threshold_smoothed, thresholds)
        
        # Calculate rate
        if len(times_to_threshold) > 0:
            mean_time = np.mean(times_to_threshold)
            thresh_rate = (value_range / mean_time) * 1000  # Scale for readability
        else:
            thresh_rate = 0
        
        # Plot for threshold method
        ax_thresh.plot(times, trajectory, '-', color='lightgray', label='Original')
        ax_thresh.plot(times, threshold_smoothed, '-', color=PLOT_COLORS.get(scenario, 'blue'),
                      linewidth=2.5, label='Smoothed')
        
        # Plot thresholds and times
        threshold_colors = ['green', 'red', 'purple']
        for i, (threshold, time_point) in enumerate(zip(thresholds, times_to_threshold)):
            # Horizontal line for threshold
            ax_thresh.axhline(y=threshold, color=threshold_colors[i], linestyle='--', alpha=0.7)
            
            # Vertical line for time point
            ax_thresh.axvline(x=time_point, color=threshold_colors[i], linestyle=':', alpha=0.7)
            
            # Mark intersection
            ax_thresh.plot(time_point, threshold, 'o', color=threshold_colors[i], markersize=8)
        
        # Add title and rate
        ax_thresh.set_title(f"Threshold Method\n{friendly_scenario} - {topology}")
        ax_thresh.text(0.05, 0.95, f"Rate: {thresh_rate:.4f}×10³",
                     transform=ax_thresh.transAxes, va='top', ha='left', 
                     bbox=dict(facecolor='white', alpha=0.8))
        
        # Configure both subplots
        for ax in [ax_inf, ax_thresh]:
            ax.grid(True, alpha=0.3, linestyle='--')

    # Add common y-axis label
    fig.text(0.04, 0.5, 'Average State', va='center', rotation='vertical', fontsize=14)
    
    # Add common x-axis label
    fig.text(0.5, 0.04, 'Time [timestep / system size]', ha='center', fontsize=14)
    
    # Add title
    fig.suptitle(f'Comparison of Convergence Rate Methods\n(Inflection: sigma={GAUSSIAN_SIGMA}, window={REGRESSION_WINDOW})', fontsize=16)
    
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def horizontal_compare_methods(data, output_path=None):
    """Compare methods with horizontal grouping by topology
    
    For each topology, shows scenarios side-by-side with both methods
    """
    setup_plotting_style()
    
    # Get unique combinations of topologies and scenarios
    topologies = data['type'].unique()
    scenarios = data['scenario_grouped'].unique()
    
    # Limit number of scenarios to display
    max_scenarios = min(4, len(scenarios))
    selected_scenarios = scenarios[:max_scenarios]
    
    # Create one subplot per topology
    fig, main_axes = plt.subplots(len(topologies), 1, 
                                 figsize=(16, 6 * len(topologies)),
                                 sharex=True, 
                                 gridspec_kw={'hspace': 0.4})
    
    if len(topologies) == 1:
        main_axes = [main_axes]
    
    for t_idx, topology in enumerate(topologies):
        ax = main_axes[t_idx]
        ax.set_visible(False)  # Hide the main axes
        
        # Create subplots for each scenario within this topology
        gs = GridSpec(2, max_scenarios, figure=fig, 
                     left=0.1, right=0.98, 
                     bottom=0.1 + t_idx * 0.9/len(topologies), 
                     top=0.1 + (t_idx + 0.8) * 0.9/len(topologies))
                     
        # Add a title for this topology
        fig.text(0.5, 0.1 + (t_idx + 0.9) * 0.9/len(topologies), 
                f"Topology: {topology}", 
                ha='center', fontsize=16, fontweight='bold')
    
        for s_idx, scenario in enumerate(selected_scenarios):
            # Filter data for this topology and scenario
            filtered_data = data[(data['type'] == topology) & 
                               (data['scenario_grouped'] == scenario) &
                               (data['measurement'] == 'avg_state')]
            
            if filtered_data.empty:
                continue
            
            # Get trajectory data
            times = filtered_data['t'].values
            trajectory = filtered_data['value'].values
            
            # Use original friendly name
            friendly_scenario = friendly_names.get(scenario, scenario)
            
            # Create subplot for inflection method (top row)
            ax_inf = fig.add_subplot(gs[0, s_idx])
            
            # Apply smoothing for inflection method
            inflection_smoothed = gaussian_filter1d(trajectory, GAUSSIAN_SIGMA)
            
            # Find inflection point
            inf_idx, _, _ = find_inflection(trajectory, sigma=GAUSSIAN_SIGMA, min_idx=MIN_INFLECTION_IDX)
            if inf_idx is None:
                inf_idx = len(trajectory) // 2
            
            # Calculate convergence rate using inflection method
            inf_rate, b0, b1 = estimate_convergence_rate(inflection_smoothed, inf_idx, regwin=REGRESSION_WINDOW)
            
            # Plot for inflection method
            ax_inf.plot(times, trajectory, '-', color='lightgray')
            ax_inf.plot(times, inflection_smoothed, '-', color=PLOT_COLORS.get(scenario, 'blue'), linewidth=2.5)
            
            # Mark inflection point
            ax_inf.plot(times[inf_idx], inflection_smoothed[inf_idx], 'o', color='red', markersize=8)
            
            # Plot linear fit around inflection point
            fit_start = max(0, inf_idx - REGRESSION_WINDOW)
            fit_end = min(len(times), inf_idx + REGRESSION_WINDOW + 1)
            fit_x = times[fit_start:fit_end]
            fit_y = b0 + b1 * np.arange(fit_start, fit_end)
            ax_inf.plot(fit_x, fit_y, '--', color='black', linewidth=2)
            
            # Configure inflection subplot
            ax_inf.set_title(f"Inflection: {friendly_scenario}")
            ax_inf.text(0.05, 0.95, f"Rate: {inf_rate*1000:.4f}×10³",
                      transform=ax_inf.transAxes, va='top', ha='left', 
                      bbox=dict(facecolor='white', alpha=0.8))
            ax_inf.grid(True, alpha=0.3, linestyle='--')
            
            # Create subplot for threshold method (bottom row)
            ax_thresh = fig.add_subplot(gs[1, s_idx])
            
            # Apply light smoothing for threshold method
            threshold_smoothed = gaussian_filter1d(trajectory, 100)
            
            # Get starting and final values
            start_val = threshold_smoothed[0]
            final_val = threshold_smoothed[-1]
            
            # Calculate range of values
            value_range = final_val - start_val
            
            # Define thresholds (25%, 50%, 75% of total change)
            thresholds = [start_val + value_range * fraction for fraction in [0.25, 0.5, 0.75]]
            
            # Calculate times to reach thresholds
            times_to_threshold = calculate_time_to_threshold(times, threshold_smoothed, thresholds)
            
            # Calculate rate
            if len(times_to_threshold) > 0:
                mean_time = np.mean(times_to_threshold)
                thresh_rate = (value_range / mean_time) * 1000  # Scale for readability
            else:
                thresh_rate = 0
            
            # Plot for threshold method
            ax_thresh.plot(times, trajectory, '-', color='lightgray')
            ax_thresh.plot(times, threshold_smoothed, '-', color=PLOT_COLORS.get(scenario, 'blue'), linewidth=2.5)
            
            # Plot thresholds and times
            threshold_colors = ['green', 'red', 'purple']
            for i, (threshold, time_point) in enumerate(zip(thresholds, times_to_threshold)):
                # Horizontal line for threshold
                ax_thresh.axhline(y=threshold, color=threshold_colors[i], linestyle='--', alpha=0.7)
                
                # Vertical line for time point
                ax_thresh.axvline(x=time_point, color=threshold_colors[i], linestyle=':', alpha=0.7)
                
                # Mark intersection
                ax_thresh.plot(time_point, threshold, 'o', color=threshold_colors[i], markersize=8)
            
            # Configure threshold subplot
            ax_thresh.set_title(f"Threshold: {friendly_scenario}")
            ax_thresh.text(0.05, 0.95, f"Rate: {thresh_rate:.4f}×10³",
                         transform=ax_thresh.transAxes, va='top', ha='left', 
                         bbox=dict(facecolor='white', alpha=0.8))
            ax_thresh.grid(True, alpha=0.3, linestyle='--')
            
            # Add y-axis label only to the leftmost subplots
            if s_idx == 0:
                ax_inf.set_ylabel('Average State')
                ax_thresh.set_ylabel('Average State')
            
            # Add x-axis label only to the bottom row
            if t_idx == len(topologies) - 1:
                ax_thresh.set_xlabel('Time [timestep / system size]')
    
    # Add overall title
    fig.suptitle('Comparison of Convergence Rate Calculation Methods by Topology', fontsize=18, y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def get_rates_dataframe(data):
    """Calculate convergence rates using both methods for all scenarios and topologies"""
    rates_data = []
    
    # Group by scenario and topology
    for topology in data['type'].unique():
        for scenario in data['scenario_grouped'].unique():
            # Filter data for this scenario and topology
            filtered_data = data[(data['type'] == topology) & 
                              (data['scenario_grouped'] == scenario) &
                              (data['measurement'] == 'avg_state')]
            
            if filtered_data.empty:
                continue
                
            # Get trajectory data
            times = filtered_data['t'].values
            trajectory = filtered_data['value'].values
            
            # Get friendly name
            friendly_scenario = friendly_names.get(scenario, scenario)
            
            # Calculate inflection method rate
            inflection_smoothed = gaussian_filter1d(trajectory, GAUSSIAN_SIGMA)
            inf_idx, _, _ = find_inflection(trajectory, sigma=GAUSSIAN_SIGMA, min_idx=MIN_INFLECTION_IDX)
            if inf_idx is None:
                inf_idx = len(trajectory) // 2
            
            inf_rate, _, _ = estimate_convergence_rate(inflection_smoothed, inf_idx, regwin=REGRESSION_WINDOW)
            inf_rate *= 1000  # Scale for display
            
            # Calculate threshold method rate
            threshold_smoothed = gaussian_filter1d(trajectory, 100)
            start_val = threshold_smoothed[0]
            final_val = threshold_smoothed[-1]
            value_range = final_val - start_val
            
            thresholds = [start_val + value_range * fraction for fraction in [0.25, 0.5, 0.75]]
            times_to_threshold = calculate_time_to_threshold(times, threshold_smoothed, thresholds)
            
            if len(times_to_threshold) > 0:
                mean_time = np.mean(times_to_threshold)
                thresh_rate = (value_range / mean_time) * 1000  # Scale for readability
            else:
                thresh_rate = 0
                
            # Add to results
            rates_data.append({
                'scenario': friendly_scenario,
                'topology': topology,
                'inflection_rate': inf_rate,
                'threshold_rate': thresh_rate,
                'raw_scenario': scenario
            })
    
    return pd.DataFrame(rates_data)

def plot_rates_correlation(data, output_path=None):
    """Plot a correlation between inflection and threshold methods"""
    setup_plotting_style()
    
    # Calculate rates for all scenarios and topologies
    rates_df = get_rates_dataframe(data)
    
    # Create the correlation plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate correlation
    corr, p_value = pearsonr(rates_df['inflection_rate'], rates_df['threshold_rate'])
    
    # Plot by scenarios
    for scenario in rates_df['raw_scenario'].unique():
        scenario_data = rates_df[rates_df['raw_scenario'] == scenario]
        ax.scatter(
            scenario_data['inflection_rate'],
            scenario_data['threshold_rate'],
            label=friendly_names.get(scenario, scenario),
            color=PLOT_COLORS.get(scenario, 'gray'),
            s=100,
            alpha=0.7,
            edgecolors='black'
        )
    
    # Calculate max value for axis limits
    max_val = max(
        rates_df['inflection_rate'].max(), 
        rates_df['threshold_rate'].max()
    ) * 1.1
    
    # Plot diagonal line y=x
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    # Add correlation text
    ax.text(0.05, 0.95, f"Correlation: {corr:.3f}\np-value: {p_value:.3f}",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Add labels with topology for each point
    for _, row in rates_df.iterrows():
        ax.annotate(
            row['topology'],
            (row['inflection_rate'], row['threshold_rate']),
            xytext=(7, 0),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    # Configure plot
    ax.set_xlabel('Inflection Method Rate (×10³)', fontsize=14)
    ax.set_ylabel('Threshold Method Rate (×10³)', fontsize=14)
    ax.set_title('Correlation Between Convergence Rate Calculation Methods', fontsize=16)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return rates_df

def test_inflection_parameters(data, sigmas=[150, 300], windows=[10, 50], output_path=None):
    """Test different parameter combinations for inflection method across multiple algorithms"""
    # Use a smaller font size for this specific plot
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })
    
    # Get unique combinations of topologies and scenarios
    topologies = data['type'].unique()
    all_scenarios = data['scenario_grouped'].unique()
    
    # Select a single topology for consistency
    selected_topology = topologies[0]  # Use the first available topology
    
    # Pick 4 diverse scenarios if available
    desired_scenarios = ['random_none', 'biased_diff', 'bridge_diff', 'node2vec_none']
    selected_scenarios = []
    
    # First try to find our desired scenarios
    for scenario in desired_scenarios:
        if scenario in all_scenarios:
            selected_scenarios.append(scenario)
    
    # If we don't have 4 desired scenarios, fill with others
    if len(selected_scenarios) < 4:
        for scenario in all_scenarios:
            if scenario not in selected_scenarios:
                selected_scenarios.append(scenario)
                if len(selected_scenarios) == 4:
                    break
    
    # Limit to max 4 scenarios
    selected_scenarios = selected_scenarios[:4]
    n_scenarios = len(selected_scenarios)
    
    # Set up figure dimensions - make it more compact
    fig_width = 12
    fig_height = 5 * n_scenarios
    
    # Create the figure with tighter spacing
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Results storage
    all_results = []
    
    # Calculate total rows and columns
    total_rows = n_scenarios * len(sigmas)
    total_cols = len(windows)
    
    # Process each scenario
    for scenario_idx, scenario in enumerate(selected_scenarios):
        # Create a subtitle for this scenario
        friendly_scenario = friendly_names.get(scenario, scenario)
        fig.text(0.5, 1 - (scenario_idx * 1/n_scenarios) - 0.02, 
                f"{friendly_scenario} on {selected_topology}", 
                ha='center', fontsize=12, fontweight='bold')
        
        # Get data for this scenario
        filtered_data = data[(data['type'] == selected_topology) & 
                           (data['scenario_grouped'] == scenario) &
                           (data['measurement'] == 'avg_state')]
        
        if filtered_data.empty:
            print(f"No data found for {selected_topology} / {scenario}")
            continue
        
        # Get trajectory data
        times = filtered_data['t'].values
        trajectory = filtered_data['value'].values
        
        # Test each parameter combination
        for sigma_idx, sigma in enumerate(sigmas):
            for window_idx, window in enumerate(windows):
                # Calculate subplot index
                row_idx = scenario_idx * len(sigmas) + sigma_idx
                col_idx = window_idx
                subplot_idx = row_idx * total_cols + col_idx + 1  # +1 because subplot indices start at 1
                
                # Create subplot
                ax = fig.add_subplot(total_rows, total_cols, subplot_idx)
                
                # Apply smoothing
                smoothed = gaussian_filter1d(trajectory, sigma)
                
                # Find inflection point
                inf_idx, _, _ = find_inflection(trajectory, sigma=sigma, min_idx=MIN_INFLECTION_IDX)
                if inf_idx is None:
                    inf_idx = len(trajectory) // 2
                
                # Calculate rate
                rate, b0, b1 = estimate_convergence_rate(smoothed, inf_idx, regwin=window)
                
                # Original trajectory
                ax.plot(times, trajectory, '-', color='lightgray')
                
                # Smoothed trajectory
                ax.plot(times, smoothed, '-', color=PLOT_COLORS.get(scenario, 'blue'), linewidth=1.5)
                
                # Mark inflection point
                ax.plot(times[inf_idx], smoothed[inf_idx], 'o', color='red', markersize=4)
                
                # Plot fit
                fit_start = max(0, inf_idx - window)
                fit_end = min(len(times), inf_idx + window + 1)
                fit_x = times[fit_start:fit_end]
                fit_y = b0 + b1 * np.arange(fit_start, fit_end)
                ax.plot(fit_x, fit_y, '--', color='black', linewidth=1)
                
                # Add rate with smaller font
                ax.text(0.05, 0.95, f"Rate: {rate*1000:.3f}×10³",
                      transform=ax.transAxes, va='top', ha='left',
                      bbox=dict(facecolor='white', alpha=0.7),
                      fontsize=8)
                
                # Configure subplot with compact title
                ax.set_title(f"σ={sigma}, w={window}", fontsize=9)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Set tight axis limits to maximize plot area
                min_val = np.min(trajectory)
                max_val = np.max(trajectory)
                padding = (max_val - min_val) * 0.1
                ax.set_ylim(min_val - padding, max_val + padding)
                
                # Remove all tick labels except for leftmost column (y) and bottom row (x)
                if window_idx != 0:  # Not leftmost column
                    ax.set_yticklabels([])
                
                if not (row_idx == total_rows - 1 and scenario_idx == n_scenarios - 1 and sigma_idx == len(sigmas) - 1):
                    # Not the last row in the last scenario group
                    ax.set_xticklabels([])
                
                # Only show y-label for leftmost column
                if window_idx == 0:
                    ax.set_ylabel(f'σ={sigma}', fontsize=9)
                
                # Only show x-label for bottom row of the entire plot
                if row_idx == total_rows - 1 and window_idx == 1:  # Center column of bottom row
                    ax.set_xlabel('Time', fontsize=9)
                
                # Add to results
                all_results.append({
                    'scenario': friendly_scenario,
                    'topology': selected_topology,
                    'sigma': sigma,
                    'window': window,
                    'rate': rate*1000
                })
    
    # Add compact legend
    legend_handles = [
        plt.Line2D([], [], color='lightgray', label='Original'),
        plt.Line2D([], [], color='blue', linewidth=1.5, label='Smoothed'),
        plt.Line2D([], [], marker='o', color='red', linestyle='None', markersize=4, label='Inflection'),
        plt.Line2D([], [], color='black', linestyle='--', linewidth=1, label='Fit')
    ]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              ncol=4, frameon=True, fontsize=9)
    
    # Add overall title
    fig.suptitle('Parameter Testing for Inflection Method', fontsize=14, y=0.99)
    
    # Use tight layout with smaller margins
    plt.tight_layout(rect=[0.5, 0.8, 1, 0.96], pad=0.8, h_pad=0.5, w_pad=0.5)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Reset font sizes to original
    setup_plotting_style()
    
    # Return results as DataFrame
    return pd.DataFrame(all_results)

#-------------------------
# MAIN EXECUTION
#-------------------------

# Setup plotting
setup_plotting_style()

# Load and prepare data
data = load_and_prepare_data(DATA_FILE, TOPOLOGY_FILTER, SCENARIO_FILTER)

# Run the selected mode
if MODE == 'inflection':
    demonstrate_inflection_method(data, OUTPUT_PATH)
elif MODE == 'threshold':
    demonstrate_threshold_method(data, OUTPUT_PATH)
elif MODE == 'compare':
    compare_methods(data, OUTPUT_PATH)
elif MODE == 'horizontal_compare':
    horizontal_compare_methods(data, OUTPUT_PATH)
elif MODE == 'correlation':
    rates_df = plot_rates_correlation(data, OUTPUT_PATH)
    print("Correlation Results:")
    print(rates_df)
elif MODE == 'parameter_test':
    test_inflection_parameters(data, 
                            sigmas=[150,300, 60],
                            windows=[10,15,50], 
                            output_path=OUTPUT_PATH)
else:
    print(f"Invalid mode: {MODE}. Choose from 'inflection', 'threshold', 'compare', 'horizontal_compare', 'correlation', or 'parameter_test'.")

# Print information about the data analyzed
print(f"Data file: {DATA_FILE}")
print(f"Topologies: {', '.join(data['type'].unique())}")
print(f"Scenarios: {', '.join([friendly_names.get(s, s) for s in data['scenario_grouped'].unique()])}")