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

# get list of relevant output files
file_list = list(filter(
    lambda x: file_extension in x and "default" in x, os.listdir("../Output")))


#%%
id_vars = ['t', 'scenario', 'rewiring', 'type']
default_run = pd.read_csv(os.path.join("../Output", file_list[0]))
default_run = default_run.drop(default_run.columns[0], axis=1)
default_r_l = pd.melt(default_run, id_vars=id_vars, var_name='measurement', value_name='value')
default_r_l['scenario_grouped'] = default_r_l['scenario'].str.cat(default_r_l['rewiring'], sep='_')
default_r_l = default_r_l.drop(columns=['scenario', 'rewiring'])
default_r_l['value'] = pd.to_numeric(default_r_l['value'], errors='coerce')

data = default_r_l[default_r_l['measurement']=='avg_state']
#data = default_r_l[default_r_l['t'] < 4000]
data = data.drop(data[data['t'] > 2000].index)

#sns.lineplot(data, x='t', y = 'value', hue = 'type')

# Plot the lines on two facets
sns.relplot(
    data=data,
    x="t", y= "value",
    hue="scenario_grouped",col="type",
    kind="line",
    alpha = 0.7,
    height=5, aspect=.75, facet_kws=dict(sharex=False),
)










































#%%

def extract_values(fname):
    # Match the structure and extract the values directly
    match = re.search(r'/(\w+)_linkif_(\w+)_top_(\w+).csv$', fname)
    return match.groups() if match else (None, None, None)

joined = []

for i in file_list:
    vals = pd.read_csv(f"../Output/{i}", delimiter=",")
    extract_values(i)
    joined.append(vals)
    

extract_values

runs_array = pd.concat(joined)
runs_array.to_csv("runs_array.csv")
data = runs_array
