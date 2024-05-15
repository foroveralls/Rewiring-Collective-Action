# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:56:55 2024

@author: everall
"""

import seaborn as sns
import sys
import os
import re
import numpy as  np
import pandas as pd
import seaborn as sns
from statistics import stdev, median, mean
import matplotlib.pyplot as plt

#%% setting file params

file_list = os.listdir("../Output")
file_extension = ".csv"

#%% Default run - Importing model output

keyword = "heatmap"

# get list of relevant output files
file_list = list(filter(
    lambda x: file_extension in x and keyword in x, os.listdir("../Output")))


#%%

df = pd.read_csv(os.path.join("../Output", file_list[3]))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
# Adjust the "rewiring" column for specific modes
df.loc[df['mode'].isin(['wtf', 'node2vec']), 'rewiring'] = 'empirical'

# Create a new 'scenario' column that combines 'rewiring' and 'mode'
df['scenario'] = df['rewiring'] + ' ' + df['mode']

# Define the value columns and the unique identifiers
value_columns = ['state', 'state_std', 'convergence_speed']
unique_scenarios = df['scenario'].unique()
unique_topologies = df['topology'].unique()

# Set general aesthetic settings for the plots
sns.set(style="white", context="talk")  # Adjust to 'paper' or 'talk' for smaller or larger sizes
plt.rcParams.update({'axes.titlesize': 'large', 'axes.titleweight': 'bold', 'axes.labelsize': 'medium'})

# Loop through each topology to create a separate figure
for topology in unique_topologies:
    fig, axes = plt.subplots(nrows=len(unique_scenarios), ncols=len(value_columns), figsize=(18, 5 * len(unique_scenarios)), sharex=True, sharey=True)

    for i, scenario in enumerate(unique_scenarios):
        scenario_data = df[(df['scenario'] == scenario) & (df['topology'] == topology)]

        for k, col in enumerate(value_columns):
            ax = axes[i, k] if len(unique_scenarios) > 1 else axes[k]
            heatmap_data = scenario_data.pivot_table(index='polarisingNode_f', columns='stubbornness', values=col, aggfunc='mean')
            
            # Check if heatmap_data is empty
            if heatmap_data.empty:
                ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                continue
            
            cbar = k == len(value_columns) - 1  # Add a color bar only to the last column of each row
            sns.heatmap(heatmap_data, ax=ax, cmap='viridis', cbar=cbar, cbar_kws={'label': "" if cbar else None})
            ax.invert_yaxis()  # Invert the y-axis to have 0 at the bottom

            # Setting the correct labels
            if k == 0:  # Only set the y-label for the first column
                ax.set_ylabel('Polarising Node f')
                # Add the scenario label to the left of the y-axis labels
                ax.text(-0.3, 0.5, scenario, transform=ax.transAxes, 
                        ha='right', va='center', fontweight='bold', fontsize='large', rotation=0)
            else:
                ax.set_ylabel('')
            if i == len(unique_scenarios) - 1:  # Only set the x-label for the bottom row
                ax.set_xlabel('Stubbornness')
            else:
                ax.set_xlabel('')
            if i == 0:
                ax.set_title(col, fontsize='large', fontweight='bold')

    # Adding a supra-level title for each figure
    fig.suptitle(f'Topology: {topology}', fontsize=20, fontweight='bold', y=1.01)


    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'../Figs/Heatmaps/heatmap_{topology}.png', bbox_inches='tight', dpi = 300)  # Save each figure to a file
    plt.show()

