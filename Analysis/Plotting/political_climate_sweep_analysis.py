#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 18:00:54 2025

@author: jpoveralls
"""

#!/usr/bin/env python3
"""
Analysis script for investigating state transitions in parameter sweep data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filename):
    """Load and return the sweep data."""
    df = pd.read_csv(filename)
    print(f"Loaded data with {len(df)} rows")
    return df

def analyze_transitions(df, param_name='political_climate', max_param=0.15):
    """Analyze the transitions in state values across parameter values."""
    # Filter to parameter values <= max_param
    df = df[df[param_name] <= max_param]
    
    # Get unique parameter values, topologies, and modes+rewirings
    param_values = sorted(df[param_name].unique())
    topologies = sorted(df['topology'].unique())
    
    # Create combinations of mode and rewiring
    df['scenario'] = df['mode'] + '_' + df['rewiring']
    scenarios = df['scenario'].unique()
    
    print(f"Parameter values: {param_values}")
    print(f"Topologies: {topologies}")
    print(f"Scenarios: {scenarios}")
    
    # Analyze each topology and scenario combination
    for topology in topologies:
        for scenario in scenarios:
            # Filter data
            scenario_data = df[(df['topology'] == topology) & (df['scenario'] == scenario)]
            if scenario_data.empty:
                continue
            
            # For each parameter value, analyze the distribution of states
            states_by_param = {}
            for param in param_values:
                param_states = scenario_data[scenario_data[param_name] == param]['state'].values
                states_by_param[param] = param_states
            
            # Print summary statistics
            print(f"\nTopology: {topology}, Scenario: {scenario}")
            for param, states in states_by_param.items():
                if len(states) > 0:
                    print(f"  {param_name}={param:.2f}: mean={np.mean(states):.4f}, std={np.std(states):.4f}, min={np.min(states):.4f}, max={np.max(states):.4f}, count={len(states)}")

def plot_state_distributions(df, param_name='political_climate', max_param=0.15):
    """Plot distributions of state values for each parameter value."""
    # Filter to parameter values <= max_param
    df = df[df[param_name] <= max_param]
    
    # Get unique parameter values 
    param_values = sorted(df[param_name].unique())
    
    # Create a single figure with multiple rows of plots
    plt.figure(figsize=(15, 10))
    
    # Sample some combinations for clarity
    sample_topology = 'DPAH'
    sample_scenarios = ['biased_diff', 'bridge_diff', 'none_none', 'random_none']
    
    for i, scenario in enumerate(sample_scenarios):
        scenario_parts = scenario.split('_')
        mode = scenario_parts[0]
        rewiring = '_'.join(scenario_parts[1:]) if len(scenario_parts) > 1 else 'none'
        
        plt.subplot(len(sample_scenarios), 1, i+1)
        
        # Filter data
        scenario_data = df[(df['topology'] == sample_topology) & 
                           (df['mode'] == mode) & 
                           (df['rewiring'] == rewiring)]
        
        if not scenario_data.empty:
            # Create line plot
            mean_values = [scenario_data[scenario_data[param_name] == val]['state'].mean() 
                           for val in param_values]
            plt.plot(param_values, mean_values, 'o-', label=f'Mean state')
            
            # Add standard deviation
            std_values = [scenario_data[scenario_data[param_name] == val]['state'].std() 
                          for val in param_values]
            plt.fill_between(param_values, 
                             [m-s for m,s in zip(mean_values, std_values)],
                             [m+s for m,s in zip(mean_values, std_values)],
                             alpha=0.2)
            
            # Create scatter plot of all individual points to see actual distribution
            for j, param in enumerate(param_values):
                param_states = scenario_data[scenario_data[param_name] == param]['state'].values
                # Add jitter to x-axis for better visibility
                jittered_x = np.random.normal(param, 0.002, size=len(param_states))
                plt.scatter(jittered_x, param_states, alpha=0.3, s=10)
        
        plt.title(f'{sample_topology}, {scenario}')
        plt.xlabel(param_name)
        plt.ylabel('State')
        plt.ylim(-1.1, 1.1)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('state_distribution_analysis.png', dpi=300)
    plt.show()

def plot_transition_heatmap(df, param_name='political_climate', max_param=0.15):
    """Create heatmaps showing the actual distribution of state values."""
    # Filter to parameter values <= max_param
    df = df[df[param_name] <= max_param]
    
    # Sample some scenarios for clarity
    sample_topology = 'DPAH'
    sample_scenario = 'biased_diff'
    scenario_parts = sample_scenario.split('_')
    mode = scenario_parts[0]
    rewiring = '_'.join(scenario_parts[1:]) if len(scenario_parts) > 1 else 'none'
    
    # Filter data
    scenario_data = df[(df['topology'] == sample_topology) & 
                       (df['mode'] == mode) & 
                       (df['rewiring'] == rewiring)]
    
    if scenario_data.empty:
        print(f"No data for {sample_topology}, {sample_scenario}")
        return
    
    # Get parameter values
    param_values = sorted(scenario_data[param_name].unique())
    
    # Create a 2D histogram
    state_bins = np.linspace(-1, 1, 21)  # 20 bins from -1 to 1
    hist_data = np.zeros((len(param_values), len(state_bins)-1))
    
    for i, param in enumerate(param_values):
        param_states = scenario_data[scenario_data[param_name] == param]['state'].values
        if len(param_states) > 0:
            hist, _ = np.histogram(param_states, bins=state_bins, density=True)
            hist_data[i, :] = hist
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(hist_data.T, aspect='auto', origin='lower', 
               extent=[0, len(param_values)-1, -1, 1], 
               cmap='viridis', interpolation='nearest')
    
    plt.colorbar(label='Normalized Frequency')
    plt.xticks(np.arange(len(param_values)), [f"{v:.2f}" for v in param_values], rotation=45)
    plt.xlabel(param_name)
    plt.ylabel('State Value')
    plt.title(f'Distribution of State Values: {sample_topology}, {sample_scenario}')
    plt.tight_layout()
    plt.savefig('state_distribution_heatmap.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Load data
    filename = "../../Output/heatmap_sweep_sweep_20250328_1808_politicalClimate_gbh.csv"
    df = load_data(filename)
    
    # Analyze transitions
    analyze_transitions(df)
    
    # Plot state distributions
    plot_state_distributions(df)
    
    # Plot transition heatmap
    plot_transition_heatmap(df)