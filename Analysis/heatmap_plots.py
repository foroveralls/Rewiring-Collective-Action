# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:27:39 2022

@author: everall
"""

# ilona presenting first week of november
# 3 November
# %%
import sys
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import stdev, median, mean
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os as os
import inspect
import itertools
import math
import re
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import rcParams 
from matplotlib.colors import LogNorm

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)


# %% import data and joining together

# sys.path.append('C:/Users/lilli/Documents/UNI/USW-VWL/Bachelor thesis/paper/github_paper')
# os.chdir('C:/Users/lilli/Documents/UNI/USW-VWL/Bachelor thesis/paper/github_paper')
# print(os.getcwd())

#change params here
file_extension = "csv"
parameter = "random_stubbornness" #"politicalClimate"


# stipulating regex pattern to get the parameter value from the file name string
def use_regex(input_text):
    pattern = re.compile(r"\d\.\d\d", re.IGNORECASE)

    return pattern.search(input_text)


file_list = os.listdir("../Output")

# get list of relevant output files
file_list = list(filter(
    lambda x: file_extension in x and "heatmap" in x and parameter in x, os.listdir("../Output")))

# otherwise define it manually
# file list =

joined = []

for i in file_list:
    abs_path = os.path.join(currentdir, i)

    vals = pd.read_csv(f"../Output/{i}", delimiter=",")
    joined.append(vals)

runs_array = pd.concat(joined)
runs_array.to_csv("runs_array.csv")
data = runs_array

#%%

# Group by 'value' and 'state' to count occurrences of each state within each bin
state_counts = data.groupby(['value', 'state']).size().reset_index(name='counts')

# Calculate the total counts for each bin
total_counts_per_bin = state_counts.groupby('value')['counts'].sum().reset_index(name='total_counts')

# Merge the total counts with state counts
state_counts = state_counts.merge(total_counts_per_bin, on='value')

# Compute the probability of each state within each bin
state_counts['probability'] = state_counts['counts'] / state_counts['total_counts']



# Using the previously computed state_counts data
# Create the heatmap  using the "inferno" colormap
plt.figure(figsize=(9, 5))
#cbar_ticks = [0.01, 0.02, 0.03]

ax = sns.histplot(data=state_counts, x='value', y='state', weights='probability', 
                  bins=300, cbar=True, cmap="inferno", vmax=0.05)# cbar_kws={"ticks": cbar_ticks})

# Set background color and other properties
ax.set_facecolor('black')
plt.grid(False)
plt.savefig(f"../Figs/heatmap_{parameter}.pdf", bbox_inches='tight', dpi = 300) 
plt.show()

#%%

# Assuming 'data' DataFrame exists

# Group by 'value' and 'state' to count occurrences of each state within each bin
state_counts = data.groupby(['value', 'state']).size().reset_index(name='counts')

# Calculate the total counts for each bin
total_counts_per_bin = state_counts.groupby('value')['counts'].sum().reset_index(name='total_counts')

# Merge the total counts with state counts
state_counts = state_counts.merge(total_counts_per_bin, on='value')

# Compute the probability of each state within each bin
state_counts['probability'] = state_counts['counts'] / state_counts['total_counts']

# Pivot the data for heatmap format
heatmap_data = state_counts.pivot(index='value', columns='state', values='probability')

# Create the heatmap using the "inferno" colormap
#plt.figure(figsize=(12, 6))
cbar_ticks = [0.01, 0.02, 0.03]

sns.heatmap(heatmap_data, cmap="inferno", cbar_kws={"ticks": cbar_ticks})

# Set background color and other properties
plt.title('Heatmap of State Probability by Value with Log-transformed Color Scale')
plt.grid(False)
plt.show()



#%%
#seaborn plot

# sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'white', "axes.grid": False})
# #bi= sns.scatterplot(data = runs_array, x= "value", y = "state")
# #bi.xaxis()
# bi_hist = sns.histplot(data = runs_array, stat = "probability", 
#                        bins=300, cmap="inferno", pthresh = 0.00 , pmax=0.45, x= "value", y = "state",cbar = True,
#                        cbar_kws={'ticks':MaxNLocator(5),"label":"probability"}) #'format':'%.e'})
# plt.ticklabel_format(style='sci', axis='x', scilimits = (0,0))
# plt.title("Stubborness Parameter Sweep")
# plt.savefig(f"./Figs/heatmap_{parameter}.pdf", bbox_inches='tight', dpi = 300) 
# plt.show()




#%% alternative


heatmap, xedges, yedges = np.histogram2d(runs_array.value, runs_array.state ,# tauspace, nsamples * nagents, 
                                                bins=[300,300])

data = heatmap.T

fig = plt.figure(figsize =(4,3))


norm_data = (data -0) / np.sum(data, axis=0)[ np.newaxis,:]
X, Y = np.meshgrid(xedges, yedges)

plt.pcolormesh(X, Y, norm_data, norm=colors.LogNorm(vmin=1./100,vmax=0.1))
plt.colorbar()
plt.ylabel("Final State")
plt.ylim(-1,1.1)

#img = ax.pcolormesh(xedges,yedges , (norm_data)+0.001,  cmap='viridis',norm=colors.LogNorm(vmin=1./100,
                                                                                  #  vmax=0.2)) 



#%%




























