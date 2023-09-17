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
import os as os
import sys
import operator
import pandas as pd
import seaborn as sns
from statistics import stdev, median, mean
from scipy.ndimage import gaussian_filter1d
import inspect
import itertools
import math
import re
#import seaborn as sns !! conda install
#import pandas as pd !! conda install

sys.path.append('/Users/lillifrei/Documents/Uni/Collective_Action_Paper/paper/github_paper/Output')
os.chdir('/Users/lillifrei/Documents/Uni/Collective_Action_Paper/paper/github_paper/Output')
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


norewiring = np.loadtxt('./randomrewiring0.csv', delimiter = ',')
randomrewiring01 = np.loadtxt('./random_p_rewiring_0.1.csv', delimiter = ',') 
randomrewiring05 = np.loadtxt('./random_p_rewiring_0.5.csv', delimiter = ',') 
randomrewiring09 = np.loadtxt('./random_p_rewiring_0.9.csv', delimiter = ',') 
bridge_differentlink = np.loadtxt('./bridge_linkif_diff.csv', delimiter = ',')
bridge_samelink = np.loadtxt('./bridge_linkif_same.csv', delimiter = ',')
local_differentlink2 = np.loadtxt('./biased_linkif_diff.csv', delimiter = ',')
local_samelink2 = np.loadtxt('./biased_linkif_same.csv', delimiter = ',')





statesnorewiring = norewiring[:,0]
statesstdnorewiring = norewiring[:,1]
clusterstdnorewiring = norewiring[:,2]
avgdegreenorewiring = norewiring[:,3]
stddegreenorewiring = norewiring[:,4]
maxdegreenorewiring = norewiring[:,6] 
mindegreenorewiring = norewiring[:,5]

statesrandomrewiring01 = randomrewiring01[:,0]
statesstdrandomrewiring01 = randomrewiring01[:,1]
clusterstdrandomrewiring01 = randomrewiring01[:,2]
avgdegreerandomrewiring01 = randomrewiring01[:,3]
stddegreerandomrewiring01 = randomrewiring01[:,4]
maxdegreerandomrewiring01 = randomrewiring01[:,6]
mindegreerandomrewiring01 = randomrewiring01[:,5]

statesrandomrewiring05 = randomrewiring05[:,0]
statesstdrandomrewiring05 = randomrewiring05[:,1]
clusterstdrandomrewiring05 = randomrewiring05[:,2]
avgdegreerandomrewiring05 = randomrewiring05[:,3]
stddegreerandomrewiring05 = randomrewiring05[:,4]
maxdegreerandomrewiring05 = randomrewiring05[:,6] 
mindegreerandomrewiring05 = randomrewiring05[:,5]

statesrandomrewiring09 = randomrewiring09[:,0]
statesstdrandomrewiring09 = randomrewiring09[:,1]
clusterstdrandomrewiring09 = randomrewiring09[:,2]
avgdegreerandomrewiring09 = randomrewiring09[:,3]
stddegreerandomrewiring09 = randomrewiring09[:,4]
maxdegreerandomrewiring09 = randomrewiring09[:,6]
mindegreerandomrewiring09 = randomrewiring09[:,5]


stateslocal_samelink = local_samelink2[:,0]
statesstdlocal_samelink = local_samelink2[:,1]
clusterstdlocal_samelink = local_samelink2[:,2]
avgdegreelocal_samelink = local_samelink2[:,3]
stddegreelocal_samelink = local_samelink2[:,4]
maxdegreelocal_samelink = local_samelink2[:,6]
mindegreelocal_samelink = local_samelink2[:,5]

stateslocal_differentlink = local_differentlink2[:,0]
statesstdlocal_differentlink = local_differentlink2[:,1]
clusterstdlocal_differentlink = local_differentlink2[:,2]
avgdegreelocal_differentlink = local_differentlink2[:,3]
stddegreelocal_differentlink = local_differentlink2[:,4]
maxdegreelocal_differentlink = local_differentlink2[:,6]
mindegreelocal_differentlink = local_differentlink2[:,5]

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


  

# #check
# figtype, ax =  plt.subplots()
# ax.plot(xaxis,statesnorewiring,label="no rewiring", color = 'orange', linestyle = '-')
# ax.plot(xaxis,statesrandomrewiring05,label= 'p=q=0.5', color = 'red', linestyle = '-') #0.5
# #ax.plot(xaxis,statesrandomrewiring01,label="p=q=0.1", color = 'lightsalmon', linestyle = '-') #0.1
# #ax.plot(xaxis,statesrandomrewiring09,label="p=q=0.9", color = 'saddlebrown', linestyle = '-') #0.9
# # ax.plot(xaxis,statesstdrandomrewiring09,label="SD p=q=0.9", color = 'saddlebrown', linestyle = '--') #0.9
# ax.set(xlabel='time [timestep / system size]',ylabel='cooperativity')
# ax.legend(loc = 'lower right', fontsize=12)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
# ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
# ax.yaxis.grid(True, linestyle='dotted')
# ax.set_ylim([-1,1])
# ax.set_xlim([0,50])
# #figtype.savefig('check_old05_new05.pdf')

# # #fig1 cooperation + SD random rewiring
# figtype, ax =  plt.subplots()
# ax.plot(xaxis,statesnorewiring,label="no rewiring", color = 'orange', linestyle = '-')
# ax.plot(xaxis,statesstdnorewiring,label="SD no rewiring", color = 'orange', linestyle = '--') 
# ax.plot(xaxis,statesrandomrewiring05,label="random rewiring p=q=0.5", color = 'blue', linestyle = '-')
# ax.plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring p=q=0.5", color = 'blue', linestyle = '--') 
# ax.set(xlabel='time [timestep / system size]',ylabel='cooperativity')
# ax.legend(loc = 'lower right', fontsize=12)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
# ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
# ax.yaxis.grid(True, linestyle='dotted')
# ax.set_ylim([-1,1])
# ax.set_xlim([0,50])
# #figtype.savefig('1_random_rewiring_05_new.pdf') 

# #fig2 cooperation + SD random rewiring 0.1, 0.5, 0.9
# figtype, ax =  plt.subplots()
# ax.plot(xaxis,statesnorewiring,label="no rewiring", color = 'orange', linestyle = '-')
# ax.plot(xaxis,statesstdnorewiring,label="SD no rewiring", color = 'orange', linestyle = '--')
# ax.plot(xaxis,statesrandomrewiring01,label="p=q=0.1", color = 'lightsalmon', linestyle = '-') #0.1
# ax.plot(xaxis,statesstdrandomrewiring01,label="SD p=q=0.1", color = 'lightsalmon', linestyle = '--') #0.1
# ax.plot(xaxis,statesrandomrewiring05,label= 'p=q=0.5', color = 'red', linestyle = '-') #0.5
# ax.plot(xaxis,statesstdrandomrewiring05,label="SD p=q=0.5", color = 'red', linestyle = '--') #0.5
# ax.plot(xaxis,statesrandomrewiring09,label="p=q=0.9", color = 'saddlebrown', linestyle = '-') #0.9
# ax.plot(xaxis,statesstdrandomrewiring09,label="SD p=q=0.9", color = 'saddlebrown', linestyle = '--') #0.9
# ax.set(xlabel='time [timestep / system size]',ylabel='cooperativity')
# ax.legend(loc = 'lower right', fontsize=12)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
# ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
# ax.yaxis.grid(True, linestyle='dotted')
# ax.set_ylim([-1,1])
# ax.set_xlim([0,50])
# #figtype.savefig('2_random_rewiring_01_05_09.pdf')

#Fig1a static random
ax = sns.lineplot()
ax.plot(xaxis,statesnorewiring,label="static network", color = 'orange', linestyle = '-')#no rewiring
ax.plot(xaxis,statesstdnorewiring,label='_nolegend_', color = 'orange', linestyle = '--')#no rewiring
ax.plot(xaxis,statesrandomrewiring01,label="0.1", color = 'deepskyblue', linestyle = '-') #p=q=0.1
ax.plot(xaxis,statesstdrandomrewiring01,label="_nolegend_", color = 'deepskyblue', linestyle = '--') #p=q=0.1
ax.plot(xaxis,statesrandomrewiring05,label= '0.5', color = 'limegreen',linestyle = '-') #p=q=0.5
ax.plot(xaxis,statesstdrandomrewiring05,label="_nolegend_", color = 'limegreen', linestyle = '--') #p=q=0.5
ax.plot(xaxis,statesrandomrewiring09,label="0.9", color = 'violet', linestyle = '-') #p=q=0.9
ax.plot(xaxis,statesstdrandomrewiring09,label="_nolegend_",color = 'violet', linestyle = '--') #p=q=0.9
ax.set_ylabel('cooperativity')
ax.set_xlabel("time [timestep / system size]")
#ax.set_legend(title=f'{parameter}')
#ax.set_title(f"trajectories_{scenario}_{parameter}")
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#add label names according to values (see "param") and variable name (see Methods paper)
ax.legend(loc = 'lower right', fontsize=12)# labels = ['static network', '0.1', '0.5', '0.9'])
ax.set_xlim([0,50])
# ax.set_ylim([-1,3])
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
plt.savefig('1_static_random.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#Fig1b static local
ax = sns.lineplot()
ax.plot(xaxis,statesnorewiring,label="static network", color = 'orange', linestyle = '-')#no rewiring
ax.plot(xaxis,statesstdnorewiring,label='_nolegend_', color = 'orange', linestyle = '--')#no rewiring
ax.plot(xaxis, stateslocal_samelink, label = "local(similar)", color = 'limegreen', linestyle = '-') #if_same
ax.plot(xaxis, statesstdlocal_samelink, label = "_nolegend", color = 'limegreen', linestyle = '--') #if_same
ax.set_ylabel('cooperativity')
ax.set_xlabel("time [timestep / system size]")
#ax.set_legend(title=f'{parameter}')
#ax.set_title(f"trajectories_{scenario}_{parameter}")
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#add label names according to values (see "param") and variable name (see Methods paper)
ax.legend(loc = 'lower right', fontsize=12)# labels = ['static network', '0.1', '0.5', '0.9'])
ax.set_xlim([0,50])
# ax.set_ylim([-1,3])
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
plt.savefig('1_static_local.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#Fig1c static bridge
ax = sns.lineplot()
ax.plot(xaxis,statesnorewiring,label="static network", color = 'orange', linestyle = '-')#no rewiring
ax.plot(xaxis,statesstdnorewiring,label='_nolegend_', color = 'orange', linestyle = '--')#no rewiring
ax.plot(xaxis, statesbridge_differentlink, label = "bridge(opposite)", color = 'violet', linestyle = '-') #if_same
ax.plot(xaxis, statesstdbridge_differentlink, label = "_nolegend", color = 'violet', linestyle = '--') #if_same
ax.set_ylabel('cooperativity')
ax.set_xlabel("time [timestep / system size]")
#ax.set_legend(title=f'{parameter}')
#ax.set_title(f"trajectories_{scenario}_{parameter}")
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#add label names according to values (see "param") and variable name (see Methods paper)
ax.legend(loc = 'lower right', fontsize=12)# labels = ['static network', '0.1', '0.5', '0.9'])
ax.set_xlim([0,50])
# ax.set_ylim([-1,3])
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
plt.savefig('1_static_bridge.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#Fig 2a bridge same different
ax = sns.lineplot()
ax.plot(xaxis,statesnorewiring,label="static network", color = 'orange', linestyle = '-')#no rewiring
ax.plot(xaxis,statesstdnorewiring,label='_nolegend_', color = 'orange', linestyle = '--')#no rewiring
ax.plot(xaxis, statesbridge_differentlink, label = "bridge(opposite)", color = 'violet', linestyle = '-') #if_different
ax.plot(xaxis, statesstdbridge_differentlink, label = "_nolegend", color = 'violet', linestyle = '--') #if_different
ax.plot(xaxis, statesbridge_samelink, label = "bridge(similar)", color = 'deepskyblue', linestyle = '-') #if_same
ax.plot(xaxis, statesstdbridge_samelink, label = "_nolegend", color = 'deepskyblue', linestyle = '--') #if_same
ax.set_ylabel('cooperativity')
ax.set_xlabel("time [timestep / system size]")
#ax.set_legend(title=f'{parameter}')
#ax.set_title(f"trajectories_{scenario}_{parameter}")
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#add label names according to values (see "param") and variable name (see Methods paper)
ax.legend(loc = 'lower right', fontsize=12)# labels = ['static network', '0.1', '0.5', '0.9'])
ax.set_xlim([0,50])
# ax.set_ylim([-1,3])
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
plt.savefig('2_bridge_same_diff.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#Fig 2b local same different
ax = sns.lineplot()
ax.plot(xaxis,statesnorewiring,label="static network", color = 'orange', linestyle = '-')#no rewiring
ax.plot(xaxis,statesstdnorewiring,label='_nolegend_', color = 'orange', linestyle = '--')#no rewiring
ax.plot(xaxis, stateslocal_samelink, label = "local(similar)", color = 'limegreen', linestyle = '-') #if_same
ax.plot(xaxis, statesstdlocal_samelink, label = "_nolegend", color = 'limegreen', linestyle = '--') #if_same
ax.plot(xaxis, stateslocal_differentlink, label = "local(opposite)", color = 'blue', linestyle = '-') #if_different
ax.plot(xaxis, statesstdlocal_differentlink, label = "_nolegend", color = 'blue', linestyle = '--') #if_different
ax.set_ylabel('cooperativity')
ax.set_xlabel("time [timestep / system size]")
#ax.set_legend(title=f'{parameter}')
#ax.set_title(f"trajectories_{scenario}_{parameter}")
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#add label names according to values (see "param") and variable name (see Methods paper)
ax.legend(loc = 'lower right', fontsize=12)# labels = ['static network', '0.1', '0.5', '0.9'])
ax.set_xlim([0,50])
# ax.set_ylim([-1,3])
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
plt.savefig('2_local_same_diff.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#Fig 2c same local bridge
ax = sns.lineplot()
ax.plot(xaxis,statesnorewiring,label="static network", color = 'orange', linestyle = '-')#no rewiring
ax.plot(xaxis,statesstdnorewiring,label='_nolegend_', color = 'orange', linestyle = '--')#no rewiring
ax.plot(xaxis, stateslocal_samelink, label = "local(similar)", color = 'limegreen', linestyle = '-') #if_same
ax.plot(xaxis, statesstdlocal_samelink, label = "_nolegend", color = 'limegreen', linestyle = '--') #if_same
ax.plot(xaxis, statesbridge_samelink, label = "bridge(similar)", color = 'deepskyblue', linestyle = '-') #if_different
ax.plot(xaxis, statesstdbridge_samelink, label = "_nolegend", color = 'deepskyblue', linestyle = '--') #if_different
ax.set_ylabel('cooperativity')
ax.set_xlabel("time [timestep / system size]")
#ax.set_legend(title=f'{parameter}')
#ax.set_title(f"trajectories_{scenario}_{parameter}")
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#add label names according to values (see "param") and variable name (see Methods paper)
ax.legend(loc = 'lower right', fontsize=12)# labels = ['static network', '0.1', '0.5', '0.9'])
ax.set_xlim([0,50])
# ax.set_ylim([-1,3])
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
plt.savefig('2_same_local_bridge.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#Fig 2d different local bridge
ax = sns.lineplot()
ax.plot(xaxis,statesnorewiring,label="static network", color = 'orange', linestyle = '-')#no rewiring
ax.plot(xaxis,statesstdnorewiring,label='_nolegend_', color = 'orange', linestyle = '--')#no rewiring
ax.plot(xaxis, stateslocal_differentlink, label = "local(opposite)", color = 'blue', linestyle = '-') #if_different
ax.plot(xaxis, statesstdlocal_differentlink, label = "_nolegend", color = 'blue', linestyle = '--') #if_different
ax.plot(xaxis, statesbridge_differentlink, label = "bridge(opposite)", color = 'violet', linestyle = '-') #if_different
ax.plot(xaxis, statesstdbridge_differentlink, label = "_nolegend", color = 'violet', linestyle = '--') #if_different
ax.set_ylabel('cooperativity')
ax.set_xlabel("time [timestep / system size]")
#ax.set_legend(title=f'{parameter}')
#ax.set_title(f"trajectories_{scenario}_{parameter}")
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#add label names according to values (see "param") and variable name (see Methods paper)
ax.legend(loc = 'lower right', fontsize=12)# labels = ['static network', '0.1', '0.5', '0.9'])
ax.set_xlim([0,50])
# ax.set_ylim([-1,3])
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
plt.savefig('2_different_local_bridge.pdf', bbox_inches='tight', dpi = 300)
plt.show()

#Fig 3 all rewiring algorithms
ax = sns.lineplot()
ax.plot(xaxis,statesnorewiring,label="static network", color = 'black', linestyle = '-')#no rewiring
ax.plot(xaxis,statesstdnorewiring,label='_nolegend_', color = 'black', linestyle = '--')#no rewiring
ax.plot(xaxis,statesrandomrewiring05,label= 'random', color = 'orange',linestyle = '-') #p=q=0.5
ax.plot(xaxis,statesstdrandomrewiring05,label="_nolegend_", color = 'orange', linestyle = '--') #p=q=0.5
ax.plot(xaxis, stateslocal_samelink, label = "local(similar)", color = 'limegreen', linestyle = '-') #if_same
ax.plot(xaxis, statesstdlocal_samelink, label = "_nolegend", color = 'limegreen', linestyle = '--') #if_same
ax.plot(xaxis, stateslocal_differentlink, label = "local(opposite)", color = 'blue', linestyle = '-') #if_different
ax.plot(xaxis, statesstdlocal_differentlink, label = "_nolegend", color = 'blue', linestyle = '--') #if_different
ax.plot(xaxis, statesbridge_differentlink, label = "bridge(opposite)", color = 'violet', linestyle = '-') #if_different
ax.plot(xaxis, statesstdbridge_differentlink, label = "_nolegend", color = 'violet', linestyle = '--') #if_different
ax.plot(xaxis, statesbridge_samelink, label = "bridge(similar)", color = 'red', linestyle = '-') #if_same
ax.plot(xaxis, statesstdbridge_samelink, label = "_nolegend", color = 'red', linestyle = '--') #if_same
ax.set_ylabel('cooperativity')
ax.set_xlabel("time [timestep / system size]")
#ax.set_legend(title=f'{parameter}')
#ax.set_title(f"trajectories_{scenario}_{parameter}")
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#add label names according to values (see "param") and variable name (see Methods paper)
ax.legend(loc = 'lower right', fontsize=12)# labels = ['static network', '0.1', '0.5', '0.9'])
ax.set_xlim([0,50])
# ax.set_ylim([-1,3])
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
plt.savefig('3_all_compared.pdf', bbox_inches='tight', dpi = 300)
plt.show()

# #fig3 results local rewiring vs. random rewiring
# #?? call it local(same), or just local?
# figtype, ax =  plt.subplots()
# # ax.plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'red', linestyle = '-') #0.5
# # ax.plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'red', linestyle = '--') #0.5
# ax.plot(xaxis,statesnorewiring,label="no rewiring", color = 'orange', linestyle = '-')
# ax.plot(xaxis,statesstdnorewiring,label="SD no rewiring", color = 'orange', linestyle = '--') 
# ax.plot(xaxis, stateslocal_samelink, label = "local(same)", color = 'green', linestyle = '-') #if_same
# ax.plot(xaxis, statesstdlocal_samelink, label = "SD local(same)", color = 'green', linestyle = '--') #if_same
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
# #figtype.savefig('3_local_vs_randomrewiring.pdf')

# #fig4 results bridge rewiring vs. random rewiring
# figtype, ax =  plt.subplots()
# # ax.plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'red', linestyle = '-') #0.5
# # ax.plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'red', linestyle = '--') #0.5
# ax.plot(xaxis,statesnorewiring,label="no rewiring", color = 'orange', linestyle = '-')
# ax.plot(xaxis,statesstdnorewiring,label="SD no rewiring", color = 'orange', linestyle = '--') 
# ax.plot(xaxis, statesbridge_differentlink, label = "bridge(different)", color = 'blue', linestyle = '-') #if_different
# ax.plot(xaxis, statesstdbridge_differentlink, label = "SD bridge(different)", color = 'blue', linestyle = '--') #if_different
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
# #figtype.savefig('4_bridge_vs_randomrewiring.pdf')

# #fig5 bridge vs local with rewiring as baseline
# #! change random rewiring to no rewiring? or additionally?
# figtype, ax = plt.subplots(2,2)
# figtype.set_size_inches(14, 10.5)
# line_labels = ['random rewiring', 'SD random rewiring', 'bridge_same', 'SD bridge_same', 'local_same', 'SD local_same', 'bridge_different', 'SD bridge_different', 'local_different', 'SD local_different']

# # l1=ax[0, 0].plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'red', linestyle = '-')[0]
# # l2=ax[0,0].plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'red', linestyle = '--')[0]
# ax[0,0].plot(xaxis,statesnorewiring,label="no rewiring", color = 'orange', linestyle = '-')
# ax[0,0].plot(xaxis,statesstdnorewiring,label="SD no rewiring", color = 'orange', linestyle = '--')
# l3=ax[0,0].plot(xaxis, statesbridge_samelink, label = "bridge(same)", color = 'magenta', linestyle = '-')[0] #if_same
# l4=ax[0,0].plot(xaxis, statesstdbridge_samelink, label = "SD bridge(same)", color = 'magenta', linestyle = '--')[0] #if_same
# l5=ax[0,0].plot(xaxis, stateslocal_samelink, label = "local(same)", color = 'green', linestyle = '-')[0] #if_same
# l6=ax[0,0].plot(xaxis, statesstdlocal_samelink, label = "SD local(same)", color = 'green', linestyle = '--')[0] #if_same
# ax[0,0].legend(loc = 'lower right')

# # ax[0, 1].plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'red', linestyle = '-')
# # ax[0,1].plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'red', linestyle = '--')
# ax[0,1].plot(xaxis,statesnorewiring,label="no rewiring", color = 'orange', linestyle = '-')
# ax[0,1].plot(xaxis,statesstdnorewiring,label="SD no rewiring", color = 'orange', linestyle = '--')
# l7=ax[0,1].plot(xaxis, statesbridge_differentlink, label = "bridge(different)", color = 'blue', linestyle = '-')[0] #if_different
# l8=ax[0,1].plot(xaxis, statesstdbridge_differentlink, label = "SD bridge(different)", color = 'blue', linestyle = '--')[0] #if_different
# l9=ax[0,1].plot(xaxis, stateslocal_differentlink, label = "local(different)", color = 'turquoise', linestyle = '-')[0] #if_different
# l10=ax[0,1].plot(xaxis, statesstdlocal_differentlink, label = "SD local(different)", color = 'turquoise', linestyle = '--')[0] #if_different
# ax[0,1].legend(loc = 'lower right')

# # ax[1, 0].plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'red', linestyle = '-')
# # ax[1,0].plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'red', linestyle = '--')
# ax[1,0].plot(xaxis,statesnorewiring,label="no rewiring", color = 'orange', linestyle = '-')
# ax[1,0].plot(xaxis,statesstdnorewiring,label="SD no rewiring", color = 'orange', linestyle = '--')
# ax[1,0].plot(xaxis, stateslocal_samelink, label = "local(same)", color = 'green', linestyle = '-') #if_same
# ax[1,0].plot(xaxis, statesstdlocal_samelink, label = "SD local(same)", color = 'green', linestyle = '--') #if_same
# ax[1,0].plot(xaxis, stateslocal_differentlink, label = "local(different)", color = 'turquoise', linestyle = '-') #if_different
# ax[1,0].plot(xaxis, statesstdlocal_differentlink, label = "SD local(different)", color = 'turquoise', linestyle = '--') #if_different
# ax[1,0].legend(loc = 'lower right')

# # ax[1, 1].plot(xaxis,statesrandomrewiring05,label="random rewiring", color = 'red', linestyle = '-')
# # ax[1,1].plot(xaxis,statesstdrandomrewiring05,label="SD random rewiring", color = 'red', linestyle = '--')
# ax[1,1].plot(xaxis,statesnorewiring,label="no rewiring", color = 'orange', linestyle = '-')
# ax[1,1].plot(xaxis,statesstdnorewiring,label="SD no rewiring", color = 'orange', linestyle = '--')
# ax[1,1].plot(xaxis, statesbridge_samelink, label = "bridge(same)", color = 'magenta', linestyle = '-') #if_same
# ax[1,1].plot(xaxis, statesstdbridge_samelink, label = "SD bridge(same)", color = 'magenta', linestyle = '--') #if_same
# ax[1,1].plot(xaxis, statesbridge_differentlink, label = "bridge(different)", color = 'blue', linestyle = '-') #if_different
# ax[1,1].plot(xaxis, statesstdbridge_differentlink, label = "SD bridge(different)", color = 'blue', linestyle = '--') #if_different
# ax[1,1].legend(loc = 'lower right')

# def text_coords(ax=None,scalex=0.9,scaley=0.9):
#   xlims = ax.get_xlim()
#   ylims = ax.get_ylim()
#   return {'x':scalex*np.diff(xlims)+xlims[0],
#         'y':scaley*np.diff(ylims)+ylims[0]}


# scalex = [0.02,0.02,0.02,0.02]
# scaley = [1.07,1.07,1.07,1.07]
# labels = ['a) local(same) vs bridge(same)','b) local(different) vs bridge(different)','c) local(same) vs local(different)','d) bridge(same) vs bridge(different)']

# for sx,sy,a,l in zip(scalex,scaley,np.ravel(ax),labels):
#     a.text(s=l,**text_coords(ax=a,scalex=sx,scaley=sy),  fontsize = 22) #weight='bold',

# for ax in ax.flat:
#     ax.set_xlabel('time [timestep / system size]', fontsize = 20)
#     ax.set_ylabel('cooperativity', fontsize = 20)
#     ax.label_outer()
#     ax.set_ylim([-1,1])
#     ax.set_xlim([0,50])
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5)) #ticker sets the little ticks on the axes
#     ax.tick_params(axis = 'x', labelsize = 20)
#     ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
#     ax.tick_params(axis = 'y', labelsize = 20)
#     ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
#     ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
#     ax.yaxis.grid(True, linestyle='dotted')
    
# # figtype.legend([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10],     # The line objects
# #            labels=line_labels,   # The labels for each line
# #            loc = 'center left', 
# #            bbox_to_anchor=(1.01,0.7),
# #            fontsize = 20
# #            )

# #figtype.savefig('5_local_vs_bridgerewiring.pdf')

# #fig6 final comparison of all algorithms
# figtype, ax =  plt.subplots()
# ax.plot(xaxis,statesnorewiring,label="no rewiring", color = 'orange', linestyle = '-')
# ax.plot(xaxis,statesrandomrewiring05,label="random rewiring p=q=0.5", color = 'red', linestyle = '-') #0.3
# ax.plot(xaxis, stateslocal_samelink, label = "local(same)", color = 'green', linestyle = '-') #if_same
# ax.plot(xaxis, stateslocal_differentlink, label = "local(different)", color = 'turquoise', linestyle = '-') #if_different
# ax.plot(xaxis, statesbridge_differentlink, label = "bridge(different)", color = 'blue', linestyle = '-') #if_different
# ax.plot(xaxis, statesbridge_samelink, label = "bridge(same)", color = 'magenta', linestyle = '-') #if_same
# ax.set(xlabel='time [timestep / system size]',ylabel='cooperativity')
# ax.legend(loc = 'lower right', fontsize=12)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
# ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
# ax.yaxis.grid(True, linestyle='dotted')
# ax.set_ylim([-1,1])
# ax.set_xlim([0,50])
# #figtype.savefig('6_all_rewiring_algorithms.pdf')