#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:38:59 2025

@author: jpoveralls
"""

import seaborn as sns
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import transforms
import gc



#%%

cm = 1/2.54
# Constants matching the paper style
FONT_SIZE = 14
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

def setup_plotting_style():
    """Configure plotting style to match paper."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.figsize': (17.8*cm, 8.9*cm),
        'figure.dpi': 300
    })
    sns.set_theme(font_scale=FONT_SIZE/12)
    sns.set_style("white")
    sns.set_style("ticks")
    plt.style.use('seaborn-white')



#%%
def get_data_file():
    """Get the data file path from user input."""
    file_list = [f for f in os.listdir("../Output") 
                 if f.endswith(".csv") and "param_sweep_individual" in f]
    
    if not file_list:
        print("No parameter sweep files found in the Output directory.")
        sys.exit(1)
    
    for i, file in enumerate(file_list):
        print(f"{i}: {file}")
    
    file_index = int(input("Enter the index of the file you want to plot: "))
    return os.path.join("../Output", file_list[file_index])

def get_data_info(filepath):
    """Get unique scenarios and topologies without loading entire file."""
    # Read just the header and first row to get column names
    df_peek = pd.read_csv(filepath, nrows=1)
    
    # Read specific columns in chunks to get unique values
    scenarios = set()
    topologies = set()
    
    chunk_size = 100000  # Adjust based on available memory
    cols_needed = ['scenario', 'rewiring', 'type']
    
    for chunk in pd.read_csv(filepath, usecols=cols_needed, chunksize=chunk_size):
        scenarios.update(chunk['scenario'].unique())
        topologies.update(chunk['type'].unique())
    
    return list(scenarios), list(topologies)

def process_chunks_for_final_states(filepath, scenario, topology):
    """Process data in chunks for a specific scenario and topology."""
    # Only read necessary columns
    cols_needed = ['t', 'avg_state', 'model_run', 'scenario', 'type', 'polarisingNode_f']
    chunk_size = 100000  # Adjust based on available memory
    
    # Initialize dictionary to store final states
    final_states_dict = {}
    max_t = -1
    
    # First pass: find maximum t value for this scenario and topology
    print(f"Finding maximum timestep for {scenario} - {topology}...")
    for chunk in pd.read_csv(filepath, usecols=cols_needed, chunksize=chunk_size):
        chunk_filtered = chunk[(chunk['scenario'] == scenario) & 
                             (chunk['type'] == topology)]
        if not chunk_filtered.empty:
            max_t = max(max_t, chunk_filtered['t'].max())
    
    # Second pass: collect final states
    print(f"Collecting final states for {scenario} - {topology}...")
    for chunk in pd.read_csv(filepath, usecols=cols_needed, chunksize=chunk_size):
        chunk_filtered = chunk[(chunk['scenario'] == scenario) & 
                             (chunk['type'] == topology) &
                             (chunk['t'] == max_t)]
        
        for _, row in chunk_filtered.iterrows():
            key = (row['model_run'], row['polarisingNode_f'])
            final_states_dict[key] = row['avg_state']
        
    # Convert to DataFrame
    final_states = pd.DataFrame([
        {'model_run': k[0], 
         'polarisingNode_f': k[1], 
         'avg_state': v} 
        for k, v in final_states_dict.items()
    ])
    
    # Clear memory
    del final_states_dict
    gc.collect()
    
    return final_states

def create_heatmap(final_states, topology, scenario):
    """Create a single heatmap from processed final states."""
    print(f"Creating heatmap for {scenario} - {topology}...")
    
    # Create bins for final states
    final_states['state_bin'] = pd.cut(final_states['avg_state'], 
                                     bins=np.linspace(-1, 1, 21),
                                     labels=np.linspace(-0.95, 0.95, 20))
    
    # Calculate normalized counts
    heatmap_data = final_states.groupby(['polarisingNode_f', 'state_bin'])\
        .size().reset_index(name='count')
    heatmap_data['normalized_count'] = heatmap_data.groupby('polarisingNode_f')['count']\
        .transform(lambda x: x/x.sum())
    
    # Create pivot table
    pivot_data = heatmap_data.pivot_table(
        index='state_bin',
        columns='polarisingNode_f',
        values='normalized_count',
        fill_value=0
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate heatmap
    cmap = sns.diverging_palette(20, 220, as_cmap=True, center="light")
    sns.heatmap(pivot_data,
                cmap=cmap,
                center=0.5,
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Normalized Count'},
                ax=ax)
    
    # Customize axes
    ax.set_xlabel('Polarizing Node Fraction')
    ax.set_ylabel('Final State')
    
    # Format tick labels
    ax.set_yticklabels([f'{float(y):.2f}' for y in pivot_data.index], rotation=0)
    ax.set_xticklabels([f'{float(x):.2f}' for x in pivot_data.columns], rotation=45)
    
    # Add title
    friendly_name = FRIENDLY_NAMES.get(f"{scenario.split('_')[0]}_{scenario.split('_')[1]}", scenario)
    plt.title(f'{topology} - {friendly_name}')
    
    return fig

def main():
    """Main execution function."""
    # Setup
    setup_plotting_style()
    
    # Get data file
    
    data_path = get_data_file()
    
    # Get unique scenarios and topologies
    print("Analyzing file structure...")
    scenarios, topologies = get_data_info(data_path)
    
    # Create output directory
    os.makedirs('../Figs/ParameterSweep', exist_ok=True)
    
    # Process each combination
    for topology in topologies:
        for scenario in scenarios:
            print(f"\nProcessing {topology} - {scenario}")
            
            # Get final states for this combination
            final_states = process_chunks_for_final_states(data_path, scenario, topology)
            
            if not final_states.empty:
                # Create and save heatmap
                fig = create_heatmap(final_states, topology, scenario)
                save_path = f'../Figs/ParameterSweep/sweep_{topology}_{scenario}.pdf'
                fig.savefig(save_path, 
                           bbox_inches='tight',
                           dpi=300,
                           format='pdf',
                           transparent=True)
                plt.close(fig)
                
                # Clear memory
                del final_states
                gc.collect()
            
            print(f"Completed {topology} - {scenario}")

if __name__ == "__main__":
    main()