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


currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)


# %% import data and joining together


#change params here
file_extension = "csv"
parameter = "stubbornness" #"politicalClimate"


# stipulating regex pattern to get the parameter value from the file name string
def use_regex(input_text):
    pattern = re.compile(r"\d\.\d\d", re.IGNORECASE)

    return pattern.search(input_text)


file_list = os.listdir("./Output")

# get list of relevant output files
file_list = list(filter(
    lambda x: file_extension in x and "heatmap" in x and parameter in x, os.listdir("./Output")))

# otherwise define it manually
# file list =

joined = []

for i in file_list:
    abs_path = os.path.join(currentdir, i)

    vals = pd.read_csv(f"./Output/{i}", delimiter=",")
    joined.append(vals)

runs_array = pd.concat(joined)
#%%
#seaborn plot

sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'white', "axes.grid": False})
#bi= sns.scatterplot(data = runs_array, x= "value", y = "state")
#bi.xaxis()
bi_hist = sns.histplot(data = runs_array, stat = "probability", \
                        bins=100, cmap="inferno", \
                       pthresh = 0.00 , pmax=0.4, x= "value", y = "final state",cbar = True,
                       cbar_kws={'ticks':MaxNLocator(5),"label":"probability", 'format':'%.e'})
plt.ticklabel_format(style='sci', axis='x', scilimits = (0,0))
plt.title("Stubborness Parameter Sweep")
plt.savefig(f"./Figs/heatmap_{parameter}.pdf", bbox_inches='tight', dpi = 300) 
plt.show()


#%% alternative


heatmap, xedges, yedges = np.histogram2d(runs_array.value, runs_array.state ,# tauspace, nsamples * nagents, 
                                                bins=[100,100])

data = heatmap.T

fig = plt.figure(figsize =(4,3))


norm_data = (data -0) / np.sum(data, axis=0)[ np.newaxis,:]
X, Y = np.meshgrid(xedges, yedges)

plt.pcolormesh(X, Y, norm_data, norm=colors.LogNorm(vmin=1./100,vmax=0.2))
plt.colorbar()
plt.ylabel("Final State")
plt.ylim(-1,1.1)

#img = ax.pcolormesh(xedges,yedges , (norm_data)+0.001,  cmap='viridis',norm=colors.LogNorm(vmin=1./100,
                                                                                  #  vmax=0.2)) 



#%%

def phase_plot(loc_2,times):
    loc = '/home/yuki//%s/'%loc_2
    name= 'parameter_scan_all_si'
    x = np.load(loc+name)
    x = x.replace([np.inf, -np.inf], np.nan)
    #df = pd.DataFrame(x.as_matrix())
    xm = x.as_matrix()
    xm.shape
    df = pd.concat([ pd.concat([xm[j][i]['s'] for i in range(200)]) for j in range(100)], 
                   axis= 1, keys = times)
    heatmap, xedges, yedges = np.histogram2d(np.repeat(
                                                times,200*100), # tauspace, nsamples * nagents
                                                df.values.T.ravel(), 
                                                bins=[times,100])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    rcParams.update({'font.size': 18})
    plt.clf()
    #plt.imshow(heatmap.T)
    fig,ax = plt.subplots(2,1,sharex=True)
    fig.set_size_inches((10.5,10.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[10./10.5,1/10.5],width_ratios=[10./10.5,.5/10], wspace=0.1)
    ax=plt.subplot(gs[0,0])

    data = heatmap.T
    #norm_data = (data - 0) / np.max(data, axis=0)[np.newaxis,:]
    # im2 = ax.pcolormesh(xedges,yedges , (norm_data)+0.001,  cmap='viridis', norm=colors.LogNorm(vmin=heatmap.min()+1, 
    #                                                                                      vmax=heatmap.max()))
    #im2 = ax.pcolormesh(xedges,yedges , (norm_data)+0.001,  cmap='viridis')
    norm_data = (data -0) / np.sum(data, axis=0)[ np.newaxis,:]
    im2 = ax.pcolormesh(xedges,yedges , (norm_data)+0.001,  cmap='viridis',norm=colors.LogNorm(vmin=1./100,
                                                                                        vmax=0.2))
    df_inc = pd.concat([ pd.concat([xm[j][i]['i'] for i in range(200)]) for j in range(100)], 
                   axis= 1, keys = times)

    ax3=plt.subplot(gs[0,1])
    cb = plt.colorbar(im2,cax=ax3)
    cb.set_label(r'normalized by $\tau$')
    ax.set_ylabel(r'Frequency $s_i$')
    #ax.set_xlabel(r'$\tau$')
    #ax.set_ylim((0,1))
    #ax.set_aspect('equal')
    ax.set_xscale('log')


    loc = '/home/yuki/Dropbox/Unizeugs/Fernuni/BA/code/pysave/experiments/output_data/%s/'%loc_2
    name= 'parameter_scannat_sav'
    x = np.load(loc+name)
    x = x.replace([np.inf, -np.inf], np.nan)
    #df = pd.DataFrame(x.as_matrix())
    xm = x.as_matrix()

    # ks= (df_cap*df).values.reshape(100,200,100)
    # y = ks.sum(1) / df_cap.values.reshape(100,200,100).sum(1)
    #plt.imshow(heatmap.T)
    ax2 = plt.subplot(gs[1,0])
    ax3 = ax2.twinx()
    #fig.set_size_inches((12,10))

    ax2.plot(times,xm.mean(1) ,'k')
    ax2.fill_between(times,xm.mean(1)-xm.std(1),xm.mean(1)+xm.std(1),
                     color='b',alpha=0.4)
    gg=[[gini(df_inc[tau][i:(i+1)*100]) for i in range(200)] for tau in times]
    ax3.plot(times, np.array(gg).mean(1),'k')
    ax3.fill_between(times,np.array(gg).mean(1)-np.array(gg).std(1),np.array(gg).mean(1)+np.array(gg).std(1),
                     color='g',alpha=0.4)

    ax3.set_ylabel(r'Gini')
    ax2.tick_params(axis='y', colors='b')
    ax3.tick_params(axis='y', colors='g')
    ax2.set_ylabel(r'$\tilde{s}$')
    ax2.set_xlabel(r'$\tau$')
    ax2.set_xscale('log')
    ax3.set_ylim((0,0.5))
    #ax2.set_ylim(ymin=0.,ymax = 1.)
    fig.savefig('%s.pdf'%loc_2)

loc = '/home/yuki//'
name= 'parameter_scannat_sav'
x = np.load(loc+name)
x = np.load(loc+name)
x = x.replace([np.inf, -np.inf], np.nan)
#df = pd.DataFrame(x.as_matrix())
xm = x.as_matrix()


locs = ['X3_Ldistphi01_fully_eps01_q_longer']
times = [np.logspace(0,4,100)] #[np.logspace(0,3,100),
for i, loc in enumerate(locs):
    phase_plot(loc, times[i])



































