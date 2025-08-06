#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 14:13:20 2025

@author: jpoveralls
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read and process data
df = pd.read_csv("out.csv")



# Create the plot
plt.figure(figsize=(12, 8))

# Define colors for each algorithm
algorithms = df['algorithm'].unique()

# Plot each algorithm explicitly with offset for visibility
for i, algorithm in enumerate(algorithms):
    data = df[df['algorithm'] == algorithm]
    # Add tiny offset to identical lines so they're visible
    offset = i * 0.001 if algorithm.startswith('WTF') else 0
    plt.plot(data['time'], data['state'] + offset, 
             label=algorithm, 
             linewidth=3 if algorithm == 'Static' else 2,
             alpha=0.8)
    print(f"Plotted {algorithm}: {len(data)} data points")

plt.xlabel('Time')
plt.ylabel('State')
plt.title('Network Dynamics by Algorithm Type (Aggregated Trajectories)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
#plt.savefig('../Figs/wtf_algorithm_comparison.pdf', dpi=300, bbox_inches='tight')
#plt.savefig('../Figs/wtf_algorithm_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to Figs/wtf_algorithm_comparison.pdf and .png")
print(f"Total trajectories per algorithm:")
