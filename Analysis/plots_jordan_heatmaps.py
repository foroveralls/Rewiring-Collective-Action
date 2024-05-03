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

keyword = "polarising"

# get list of relevant output files
file_list = list(filter(
    lambda x: file_extension in x and keyword in x, os.listdir("../Output")))


#%%
id_vars = ["rewiring", "mode", "topology", "convergence_speed"]
default_run = pd.read_csv(os.path.join("../Output", file_list[0]))
default_run['rewiring'] = default_run['rewiring'].fillna('empirical')
default_r_l = pd.melt(default_run, id_vars=id_vars, var_name='measurement', value_name='value')
default_r_l['scenario_grouped'] = default_r_l['rewiring'].str.cat(default_r_l['mode'], sep='_')
default_r_l = default_r_l.drop(columns=['mode', 'rewiring'])


#%% Colour/Heatmap


df_map = default_r_l
sns.displot(df_map: )












# def extract_values(fname):
#     # Match the structure and extract the values directly
#     match = re.search(r'/(\w+)_linkif_(\w+)_top_(\w+).csv$', fname)
#     return match.groups() if match else (None, None, None)

# joined = []

# for i in file_list:
#     vals = pd.read_csv(f"../Output/{i}", delimiter=",")
#     extract_values(i)
#     joined.append(vals)
    

# extract_values

# runs_array = pd.concat(joined)
# runs_array.to_csv("runs_array.csv")
# data = runs_array
