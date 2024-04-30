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
default_run['scenario'] = default_run['scenario'].fillna('empirical')
default_r_l = pd.melt(default_run, id_vars=id_vars, var_name='measurement', value_name='value')
default_r_l['scenario_grouped'] = default_r_l['scenario'].str.cat(default_r_l['rewiring'], sep='_')
default_r_l = default_r_l.drop(columns=['scenario', 'rewiring'])
default_r_l['value'] = pd.to_numeric(default_r_l['value'], errors='coerce')

data = default_r_l[default_r_l['measurement']=='avg_state']
#data = default_r_l[default_r_l['t'] < 4000]
data = data.drop(data[data['t'] > 1000].index)

# #sns.lineplot(data, x='t', y = 'value', hue = 'type')
# sns.set_style("ticks")
# #sns.set(palette="muted")
# # Plot the lines on two facets
# g = sns.relplot(
#     data=data,
#     x="t", y= "value",
#     hue="scenario_grouped",col="type",
#     kind="line",
#     dashes=False,
#     palette="Set2",
#     alpha = 0.8,
#     markers = True, 
#     height=5, aspect=.75, facet_kws=dict(sharex=False),
# )


# #g.fig.suptitle('Average State Over Time by Scenario and Type', fontsize=16, fontweight='bold', color="#333333")
# g.set_titles("{col_name}", fontweight='bold', color="#333333")
# g.set_axis_labels("Time (t)", "Average State Value")
# g.set(xticks=np.arange(0, 1000, 200))
# #g.get_legend_handles_labels()
# #g.add_legend(title="Scenario & Rewiring")
# #handles, labels = g.get_legend_handles_labels()
# plt.legend(title="Scenario & Rewiring", bbox_to_anchor=(1.05, 1), loc=2)
# plt.subplots_adjust(right=0.75)

# plt.setp(g._legend.get_texts(), fontsize='12')
# plt.setp(g._legend.get_title(), fontsize='14')

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("all_scenarios_compared.png", dpi = 400)
# plt.show()

#%%
#sns.set_style("ticks") 
sns.set(style = "ticks", font_scale=1.5)

g = sns.relplot(
    data=data,
    x="t", y="value",
    hue="scenario_grouped", col="type",
    kind="line",
    dashes=False,
    palette="Set2",
    alpha=0.8,
    markers=True, 
    height=5, aspect=1.1, facet_kws=dict(sharex=False)
)

# Setting titles and labels
g.set_titles("{col_name}", fontweight='bold', color="#333333")
g.set_axis_labels("Time (t)", "Average State Value")
g.set(xticks=np.arange(0, 1001, 200))

# Custom titles for each column
custom_titles = {"FB": "Facebook", "cl": "Clustered Scale-Free", }
for ax, title in zip(g.axes.flat, g.col_names):
    # Set the custom title if defined, otherwise use the default title
    ax.set_title(custom_titles.get(title, title))
# Removing the automatically generated legend
g._legend.remove()

# Extract handles and labels for the legend
handles, labels = g.axes.flat[0].get_legend_handles_labels()

# Creating a new legend closer to the plot
g.fig.legend(handles=handles, labels=labels, title="Scenario & Rewiring", loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)

# Adjusting bottom to move the legend closer to the plot
plt.subplots_adjust(bottom=0.04)  # Smaller value moves the legend closer
#plt.tight_layout(rect=[0, 0, 1, 0.95])
# Save and show the adjusted plot
plt.gcf().set_size_inches(16, 4)
plt.savefig("all_scenarios_compared.pdf", dpi = 600, bbox_inches='tight')


plt.show()






#%%

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
