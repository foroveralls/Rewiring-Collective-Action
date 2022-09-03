# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 08:12:04 2022

@author: lilli
"""

import numpy as np
import pylab
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import glob
import math
from scipy import stats
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
import matplotlib.pylab as pylab
import os
import sys
#import seaborn as sns !! conda install
#import pandas as pd !! conda install

sys.path.append('C:/Users/lilli/Documents/UNI/USW-VWL/Bachelor thesis/paper/github_paper/Output')
os.chdir('C:/Users/lilli/Documents/UNI/USW-VWL/Bachelor thesis/paper/github_paper/Output')
print(os.getcwd())

import os
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    print(f)

params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large',
          'figure.autolayout':True}
pylab.rcParams.update(params)

systemsize = 1089 #1089

def linreg(x, y): ## snippet stolen from geeksforkeeks
    x = np.array(x)
    y = np.array(y)
    
    n = np.size(x) 
     
    mx, my = np.mean(x), np.mean(y) 
  
    ssxy = np.sum(y*x) - n*my*mx 
    ssxx = np.sum(x*x) - n*mx*mx 
  
    b1 = ssxy / ssxx 
    b0 = my - b1*mx 
  
    return(b0, b1) 

randomrewiring05 = np.loadtxt('./random_linkif_NA.csv', delimiter = ',')
norewiring = np.loadtxt('./randomrewiring0.csv', delimiter = ',')
#randomrewiring01 = np.loadtxt('./randomrewiring0.3.csv', delimiter = ',') #I don't have data sets for p=q=0.1 and 0.9, but add the right csv here
#randomrewiring09 = np.loadtxt('./randomrewiring0.7.csv', delimiter = ',')
bridge_differentlink = np.loadtxt('./bridge_linkif_diff.csv', delimiter = ',')
bridge_samelink = np.loadtxt('./bridge_linkif_same.csv', delimiter = ',')
biased_differentlink2 = np.loadtxt('./biased_linkif_diff.csv', delimiter = ',')
biased_samelink2 = np.loadtxt('./biased_linkif_same.csv', delimiter = ',')

statesrandomrewiring05 = randomrewiring05[:,0]
statesstdrandomrewiring05 = randomrewiring05[:,1]
clusterstdrandomrewiring05 = randomrewiring05[:,2]
avgdegreerandomrewiring05 = randomrewiring05[:,3]
stddegreerandomrewiring05 = randomrewiring05[:,4]
maxdegreerandomrewiring05 = randomrewiring05[:,6] 
mindegreerandomrewiring05 = randomrewiring05[:,5]

statesnorewiring = norewiring[:,0]
statesstdnorewiring = norewiring[:,1]
clusterstdnorewiring = norewiring[:,2]
avgdegreenorewiring = norewiring[:,3]
stddegreenorewiring = norewiring[:,4]
maxdegreenorewiring = norewiring[:,6] 
mindegreenorewiring = norewiring[:,5]

# statesrandomrewiring01 = randomrewiring01[:,0]
# statesstdrandomrewiring01 = randomrewiring01[:,1]
# clusterstdrandomrewiring01 = randomrewiring01[:,2]
# avgdegreerandomrewiring01 = randomrewiring01[:,3]
# stddegreerandomrewiring01 = randomrewiring01[:,4]
# maxdegreerandomrewiring01 = randomrewiring01[:,6]
# mindegreerandomrewiring01 = randomrewiring01[:,5]

# statesrandomrewiring09 = randomrewiring09[:,0]
# statesstdrandomrewiring09 = randomrewiring09[:,1]
# clusterstdrandomrewiring09 = randomrewiring09[:,2]
# avgdegreerandomrewiring09 = randomrewiring09[:,3]
# stddegreerandomrewiring09 = randomrewiring09[:,4]
# maxdegreerandomrewiring09 = randomrewiring09[:,6]
# mindegreerandomrewiring09 = randomrewiring09[:,5]

statesbiased_samelink = biased_samelink2[:,0]
statesstdbiased_samelink = biased_samelink2[:,1]
clusterstdbiased_samelink = biased_samelink2[:,2]
avgdegreebiased_samelink = biased_samelink2[:,3]
stddegreebiased_samelink = biased_samelink2[:,4]
maxdegreebiased_samelink = biased_samelink2[:,6]
mindegreebiased_samelink = biased_samelink2[:,5]

statesbiased_differentlink = biased_differentlink2[:,0]
statesstdbiased_differentlink = biased_differentlink2[:,1]
clusterstdbiased_differentlink = biased_differentlink2[:,2]
avgdegreebiased_differentlink = biased_differentlink2[:,3]
stddegreebiased_differentlink = biased_differentlink2[:,4]
maxdegreebiased_differentlink = biased_differentlink2[:,6]
mindegreebiased_differentlink = biased_differentlink2[:,5]

statesbridge_differentlink = bridge_differentlink[:,0]
statesstdbridge_differentlink = bridge_differentlink[:,1]
clusterstdbridge_differentlink = bridge_differentlink[:,2]
avgdegreebridge_differentlink = bridge_differentlink[:,3]
stddegreebridge_differentlink = bridge_differentlink[:,4]
maxdegreebridge_differentlink = bridge_differentlink[:,6]
mindegreebridge_differentlink = bridge_differentlink[:,5]

statesbridge_samelink = bridge_samelink[:,0]
statesstdbridge_samelink = bridge_samelink[:,1]
clusterstdbridge_samelink = bridge_samelink[:,2]
avgdegreebridge_samelink = bridge_samelink[:,3]
stddegreebridge_samelink = bridge_samelink[:,4]
maxdegreebridge_samelink = bridge_samelink[:,6]
mindegreebridge_samelink = bridge_samelink[:,5]

xaxis = np.array(list(range(len(statesrandomrewiring05)))) / systemsize #this is time

#plotting cooperativity and SD cooperativity

#fig1 cooperation + SD random rewiring
#I don't have a full data set for 0.5, but in this figure I guess we would show p=q=0.5
#?? add probability variations right away or do one figure with simple graph and one with more probabilities tested?
figtype, ax =  plt.subplots()
ax.plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'blue', linestyle = '-')
ax.plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'blue', linestyle = '--') 
ax.plot(xaxis,statesnorewiring,label="no rewiring", color = 'orange', linestyle = '-')
ax.plot(xaxis,statesstdnorewiring,label="SD no rewiring", color = 'orange', linestyle = '--') 
ax.set(xlabel='time [timestep / system size]',ylabel='cooperativity')
ax.legend(loc = 'lower right')
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([-1,1])
ax.set_xlim([0,50])
figtype.savefig('1_random_rewiring_05') 

# #fig2 cooperation + SD random rewiring 0.1, 0.5, 0.9
# #?? add probability variations right away or do one figure with simple graph and one with more probabilities tested?
# figtype, ax =  plt.subplots()
# ax.plot(xaxis,statesrandomrewiring05,label= 'p=q=0.3', color = 'blue', linestyle = '-') #0.5
# ax.plot(xaxis,statesstdrandomrewiring05,label="SD p=q=0.3", color = 'blue', linestyle = '--') #0.5
# ax.plot(xaxis,statesrandomrewiring01,label="p=q=0.1", color = 'orange', linestyle = '-') #0.1
# ax.plot(xaxis,statesstdrandomrewiring01,label="SD p=q=0.1", color = 'orange', linestyle = '--') #0.1
# ax.plot(xaxis,statesrandomrewiring09,label="p=q=0.9", color = 'magenta', linestyle = '-') #0.9
# ax.plot(xaxis,statesstdrandomrewiring09,label="SD p=q=0.9", color = 'magenta', linestyle = '--') #0.9
# ax.set(xlabel='time [timestep / system size]',ylabel='cooperativity')
# ax.legend(loc = 'lower right')
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
# ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
# ax.yaxis.grid(True, linestyle='dotted')
# ax.set_ylim([-1,1])
# ax.set_xlim([0,50])
# figtype.savefig('2_random_rewiring_01_05_09')

#fig3 results biased rewiring vs. random rewiring
#?? call it biased_same, or just biased?
figtype, ax =  plt.subplots()
ax.plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'blue', linestyle = '-') #0.3
ax.plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'blue', linestyle = '--') #0.3
ax.plot(xaxis, statesbiased_samelink, label = "biased_same", color = 'orange', linestyle = '-') #if_same
ax.plot(xaxis, statesstdbiased_samelink, label = "SD biased_same", color = 'orange', linestyle = '--') #if_same
ax.set(xlabel='time [timestep / system size]',ylabel='cooperativity')
ax.legend(loc = 'lower right')
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([-1,1])
ax.set_xlim([0,50])
figtype.savefig('3_biased_vs_randomrewiring')

#fig4 results bridge rewiring vs. random rewiring
figtype, ax =  plt.subplots()
ax.plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'blue', linestyle = '-') #0.3
ax.plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'blue', linestyle = '--') #0.3
ax.plot(xaxis, statesbridge_differentlink, label = "bridge_different", color = 'orange', linestyle = '-') #if_different
ax.plot(xaxis, statesstdbridge_differentlink, label = "SD bridge_different", color = 'orange', linestyle = '--') #if_different
ax.set(xlabel='time [timestep / system size]',ylabel='cooperativity')
ax.legend(loc = 'lower right')
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([-1,1])
ax.set_xlim([0,50])
figtype.savefig('4_bridge_vs_randomrewiring')

#fig5 bridge vs biased with rewiring as baseline
figtype, ax = plt.subplots(2,2)
figtype.set_size_inches(14, 10.5)
line_labels = ['random rewiring', 'SD random rewiring', 'bridge_same', 'SD bridge_same', 'biased_same', 'SD biased_same', 'bridge_different', 'SD bridge_different', 'biased_different', 'SD biased_different']

l1=ax[0, 0].plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'orange', linestyle = '-')[0]
l2=ax[0,0].plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'orange', linestyle = '--')[0]
l3=ax[0,0].plot(xaxis, statesbridge_samelink, label = "bridge_same", color = 'magenta', linestyle = '-')[0] #if_same
l4=ax[0,0].plot(xaxis, statesstdbridge_samelink, label = "SD bridge_same", color = 'magenta', linestyle = '--')[0] #if_same
l5=ax[0,0].plot(xaxis, statesbiased_samelink, label = "biased_same", color = 'green', linestyle = '-')[0] #if_same
l6=ax[0,0].plot(xaxis, statesstdbiased_samelink, label = "SD biased_same", color = 'green', linestyle = '--')[0] #if_same

l7=ax[0,1].plot(xaxis, statesbridge_differentlink, label = "bridge_different", color = 'blue', linestyle = '-')[0] #if_different
l8=ax[0,1].plot(xaxis, statesstdbridge_differentlink, label = "SD bridge_different", color = 'blue', linestyle = '--')[0] #if_different
l9=ax[0,1].plot(xaxis, statesbiased_differentlink, label = "biased_different", color = 'turquoise', linestyle = '-')[0] #if_different
l10=ax[0,1].plot(xaxis, statesstdbiased_differentlink, label = "SD biased_different", color = 'turquoise', linestyle = '--')[0] #if_different
ax[0, 1].plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'orange', linestyle = '-')
ax[0,1].plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'orange', linestyle = '--')

ax[1, 0].plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'orange', linestyle = '-')
ax[1,0].plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'orange', linestyle = '--')
ax[1,0].plot(xaxis, statesbiased_samelink, label = "biased_same", color = 'green', linestyle = '-') #if_same
ax[1,0].plot(xaxis, statesstdbiased_samelink, label = "SD biased_same", color = 'green', linestyle = '--') #if_same
ax[1,0].plot(xaxis, statesbiased_differentlink, label = "biased_different", color = 'turquoise', linestyle = '-') #if_different
ax[1,0].plot(xaxis, statesstdbiased_differentlink, label = "SD biased_different", color = 'turquoise', linestyle = '--') #if_different

ax[1, 1].plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'orange', linestyle = '-')
ax[1,1].plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'orange', linestyle = '--')
ax[1,1].plot(xaxis, statesbridge_samelink, label = "bridge_same", color = 'magenta', linestyle = '-') #if_same
ax[1,1].plot(xaxis, statesstdbridge_samelink, label = "SD bridge_same", color = 'magenta', linestyle = '--') #if_same
ax[1,1].plot(xaxis, statesbridge_differentlink, label = "bridge_different", color = 'blue', linestyle = '-') #if_different
ax[1,1].plot(xaxis, statesstdbridge_differentlink, label = "SD bridge_different", color = 'blue', linestyle = '--') #if_different

def text_coords(ax=None,scalex=0.9,scaley=0.9):
  xlims = ax.get_xlim()
  ylims = ax.get_ylim()
  return {'x':scalex*np.diff(xlims)+xlims[0],
        'y':scaley*np.diff(ylims)+ylims[0]}


scalex = [0.02,0.02,0.02,0.02]
scaley = [1.2,1.2,1.2,1.2]
labels = ['a','b','c','d']

for sx,sy,a,l in zip(scalex,scaley,np.ravel(ax),labels):
   a.text(s=l,**text_coords(ax=a,scalex=sx,scaley=sy),  fontsize = 24) #weight='bold',

for ax in ax.flat:
    ax.set_xlabel('time [timestep / system size]', fontsize = 24)
    ax.set_ylabel('cooperativity', fontsize = 24)
    ax.label_outer()
    ax.set_ylim([-1,1])
    ax.set_xlim([0,50])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5)) #ticker sets the little ticks on the axes
    ax.tick_params(axis = 'x', labelsize = 24)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.tick_params(axis = 'y', labelsize = 24)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
    ax.yaxis.grid(True, linestyle='dotted')
    
figtype.legend([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10],     # The line objects
           labels=line_labels,   # The labels for each line
           loc = 'center left', 
           bbox_to_anchor=(1.01,0.7),
           fontsize = 22
           )

figtype.savefig('5_biased_vs_bridgerewiring')

#fig6 final comparison of all algorithms
figtype, ax =  plt.subplots()
ax.plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'orange', linestyle = '-') #0.3
ax.plot(xaxis, statesbiased_samelink, label = "biased_same", color = 'green', linestyle = '-') #if_same
ax.plot(xaxis, statesbiased_differentlink, label = "biased_different", color = 'turquoise', linestyle = '-') #if_different
ax.plot(xaxis, statesbridge_differentlink, label = "bridge_different", color = 'blue', linestyle = '-') #if_different
ax.plot(xaxis, statesbridge_samelink, label = "bridge_same", color = 'magenta', linestyle = '-') #if_same
ax.set(xlabel='time [timestep / system size]',ylabel='cooperativity')
ax.legend(loc = 'lower right')
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([-1,1])
ax.set_xlim([0,50])
figtype.savefig('6_all_rewiring_algorithms')