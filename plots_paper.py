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

sys.path.append('C:/Users/lilli/Documents/UNI/USW-VWL/Bachelor thesis/project-collective-altruism_david/project-collective-altruism/data')
os.chdir('C:/Users/lilli/Documents/UNI/USW-VWL/Bachelor thesis/project-collective-altruism_david/project-collective-altruism/data')
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

randomrewiring03 = np.loadtxt('./randomrewiring0.3.csv', delimiter = ',')
bridge_differentlink = np.loadtxt('./bridge_linkifdifferent1.csv', delimiter = ',')
bridge_samelink = np.loadtxt('./bridge_linkifsame1.csv', delimiter = ',')
biased_differentlink2 = np.loadtxt('./biased_linkifdifferent_2networksteps1.csv', delimiter = ',')
biased_samelink2 = np.loadtxt('./biased_linkifsame_2networksteps1.csv', delimiter = ',')

statesrandomrewiring03 = randomrewiring03[:,0]
statesstdrandomrewiring03 = randomrewiring03[:,1]
clusterstdrandomrewiring03 = randomrewiring03[:,2]
avgdegreerandomrewiring03 = randomrewiring03[:,3]
stddegreerandomrewiring03 = randomrewiring03[:,4]
maxdegreerandomrewiring03 = randomrewiring03[:,6]
mindegreerandomrewiring03 = randomrewiring03[:,5]

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

xaxis = np.array(list(range(len(statesrandomrewiring03)))) / systemsize #this is time

#plotting cooperativity and SD cooperativity

#fig1 cooperation + SD random rewiring
#?? add probability variations right away or do one figure with simple graph and one with more probabilities tested?
figtype, ax =  plt.subplots()
ax.plot(xaxis,statesrandomrewiring03,label="random rewiring", color = 'blue', linestyle = '-') #0.3
ax.plot(xaxis,statesstdrandomrewiring03,label="SD random rewiring", color = 'blue', linestyle = '--') #0.3
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

#fig2 results biased rewiring vs. random rewiring
#?? call it biased_same, or just biased?
figtype, ax =  plt.subplots()
ax.plot(xaxis,statesrandomrewiring03,label="random rewiring", color = 'blue', linestyle = '-') #0.3
ax.plot(xaxis,statesstdrandomrewiring03,label="SD random rewiring", color = 'blue', linestyle = '--') #0.3
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

#fig3 results bridge rewiring vs. random rewiring
figtype, ax =  plt.subplots()
ax.plot(xaxis,statesrandomrewiring03,label="random rewiring", color = 'blue', linestyle = '-') #0.3
ax.plot(xaxis,statesstdrandomrewiring03,label="SD random rewiring", color = 'blue', linestyle = '--') #0.3
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

#fig4 bridge vs biased with rewiring as baseline
figtype, ax = plt.subplots(2,2)
figtype.set_size_inches(14, 10.5)
line_labels = ['random rewiring', 'SD random rewiring', 'bridge_same', 'SD bridge_same', 'biased_same', 'SD biased_same', 'bridge_different', 'SD bridge_different', 'biased_different', 'SD biased_different']

l1=ax[0, 0].plot(xaxis,statesrandomrewiring03,label="random rewiring", color = 'orange', linestyle = '-')[0]
l2=ax[0,0].plot(xaxis,statesstdrandomrewiring03,label="SD random rewiring", color = 'orange', linestyle = '--')[0]
l3=ax[0,0].plot(xaxis, statesbridge_samelink, label = "bridge_same", color = 'magenta', linestyle = '-')[0] #if_same
l4=ax[0,0].plot(xaxis, statesstdbridge_samelink, label = "SD bridge_same", color = 'magenta', linestyle = '--')[0] #if_same
l5=ax[0,0].plot(xaxis, statesbiased_samelink, label = "biased_same", color = 'green', linestyle = '-')[0] #if_same
l6=ax[0,0].plot(xaxis, statesstdbiased_samelink, label = "SD biased_same", color = 'green', linestyle = '--')[0] #if_same

l7=ax[0,1].plot(xaxis, statesbridge_differentlink, label = "bridge_different", color = 'blue', linestyle = '-')[0] #if_different
l8=ax[0,1].plot(xaxis, statesstdbridge_differentlink, label = "SD bridge_different", color = 'blue', linestyle = '--')[0] #if_different
l9=ax[0,1].plot(xaxis, statesbiased_differentlink, label = "biased_different", color = 'turquoise', linestyle = '-')[0] #if_different
l10=ax[0,1].plot(xaxis, statesstdbiased_differentlink, label = "SD biased_different", color = 'turquoise', linestyle = '--')[0] #if_different
ax[0, 1].plot(xaxis,statesrandomrewiring03,label="random rewiring", color = 'orange', linestyle = '-')
ax[0,1].plot(xaxis,statesstdrandomrewiring03,label="SD random rewiring", color = 'orange', linestyle = '--')

ax[1, 0].plot(xaxis,statesrandomrewiring03,label="random rewiring", color = 'orange', linestyle = '-')
ax[1,0].plot(xaxis,statesstdrandomrewiring03,label="SD random rewiring", color = 'orange', linestyle = '--')
ax[1,0].plot(xaxis, statesbiased_samelink, label = "biased_same", color = 'green', linestyle = '-') #if_same
ax[1,0].plot(xaxis, statesstdbiased_samelink, label = "SD biased_same", color = 'green', linestyle = '--') #if_same
ax[1,0].plot(xaxis, statesbiased_differentlink, label = "biased_different", color = 'turquoise', linestyle = '-') #if_different
ax[1,0].plot(xaxis, statesstdbiased_differentlink, label = "SD biased_different", color = 'turquoise', linestyle = '--') #if_different

ax[1, 1].plot(xaxis,statesrandomrewiring03,label="random rewiring", color = 'orange', linestyle = '-')
ax[1,1].plot(xaxis,statesstdrandomrewiring03,label="SD random rewiring", color = 'orange', linestyle = '--')
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



#fig5 final comparison of all graphs
figtype, ax =  plt.subplots()
ax.plot(xaxis,statesrandomrewiring03,label="random rewiring", color = 'orange', linestyle = '-') #0.3
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