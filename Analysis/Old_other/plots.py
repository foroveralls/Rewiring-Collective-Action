# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 08:30:38 2021

@author: lilli
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import math
from scipy import stats
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
import matplotlib.pylab as pylab
import os
import sys

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

randomrewiring05 = np.loadtxt('../randomrewiring0.5_nomin_nomax.csv', delimiter = ',')
randomrewiring0 = np.loadtxt('../randomrewiring0.csv', delimiter = ',')
randomrewiring03 = np.loadtxt('../randomrewiring0.3.csv', delimiter = ',')
randomrewiring07 = np.loadtxt('../randomrewiring0.7.csv', delimiter = ',')
randomrewiring1 = np.loadtxt('../randomrewiring1.csv', delimiter = ',')
bridge_differentlink = np.loadtxt('../bridge_linkifdifferent1.csv', delimiter = ',')
bridge_samelink = np.loadtxt('../bridge_linkifsame1.csv', delimiter = ',')
biased_differentlink = np.loadtxt('../biased_linkifdifferent1.csv', delimiter = ',') 
biased_samelink = np.loadtxt('../biased_linkifsame1.csv', delimiter = ',') 
biased_differentlink2 = np.loadtxt('../biased_linkifdifferent_2networksteps1.csv', delimiter = ',')
biased_samelink2 = np.loadtxt('../biased_linkifsame_2networksteps1.csv', delimiter = ',')

statesrandomrewiring05 = randomrewiring05[:,0]
statesstdrandomrewiring05 = randomrewiring05[:,1]
clusterstdrandomrewiring05 = randomrewiring05[:,2]
avgdegreerandomrewiring05 = randomrewiring05[:,3]
stddegreerandomrewiring05 = randomrewiring05[:,4]
#maxdegreerandomrewiring05 = randomrewiring05[:,6]
#mindegreerandomrewiring05 = randomrewiring05[:,5]

statesrandomrewiring0 = randomrewiring0[:,0]
statesstdrandomrewiring0 = randomrewiring0[:,1]
clusterstdrandomrewiring0 = randomrewiring0[:,2]
avgdegreerandomrewiring0 = randomrewiring0[:,3]
stddegreerandomrewiring0 = randomrewiring0[:,4]
maxdegreerandomrewiring0 = randomrewiring0[:,6]
mindegreerandomrewiring0 = randomrewiring0[:,5]

statesrandomrewiring03 = randomrewiring03[:,0]
statesstdrandomrewiring03 = randomrewiring03[:,1]
clusterstdrandomrewiring03 = randomrewiring03[:,2]
avgdegreerandomrewiring03 = randomrewiring03[:,3]
stddegreerandomrewiring03 = randomrewiring03[:,4]
maxdegreerandomrewiring03 = randomrewiring03[:,6]
mindegreerandomrewiring03 = randomrewiring03[:,5]

statesrandomrewiring07 = randomrewiring07[:,0]
statesstdrandomrewiring07 = randomrewiring07[:,1]
clusterstdrandomrewiring07 = randomrewiring07[:,2]
avgdegreerandomrewiring07 = randomrewiring07[:,3]
stddegreerandomrewiring07 = randomrewiring07[:,4]
maxdegreerandomrewiring07 = randomrewiring07[:,6]
mindegreerandomrewiring07 = randomrewiring07[:,5]

statesrandomrewiring1 = randomrewiring1[:,0]
statesstdrandomrewiring1 = randomrewiring1[:,1]
clusterstdrandomrewiring1 = randomrewiring1[:,2]
avgdegreerandomrewiring1 = randomrewiring1[:,3]
stddegreerandomrewiring1 = randomrewiring1[:,4]
maxdegreerandomrewiring1 = randomrewiring1[:,6]
mindegreerandomrewiring1 = randomrewiring1[:,5]


statesbridge_differentlink = bridge_differentlink[:,0]
statesstdbridge_differentlink = bridge_differentlink[:,1]
clusterstdbridge_differentlink = bridge_differentlink[:,2]
avgdegreebridge_differentlink = bridge_differentlink[:,3]
stddegreebridge_differentlink = bridge_differentlink[:,4]
maxdegreebridge_differentlink = bridge_differentlink[:,6]
mindegreebridge_differentlink = bridge_differentlink[:,5]

statesbiased_samelink = biased_samelink[:,0]
statesstdbiased_samelink = biased_samelink[:,1]
clusterstdbiased_samelink = biased_samelink[:,2]
avgdegreebiased_samelink = biased_samelink[:,3]
stddegreebiased_samelink = biased_samelink[:,4]
maxdegreebiased_samelink = biased_samelink[:,6]
mindegreebiased_samelink = biased_samelink[:,5]

statesbridge_samelink = bridge_samelink[:,0]
statesstdbridge_samelink = bridge_samelink[:,1]
clusterstdbridge_samelink = bridge_samelink[:,2]
avgdegreebridge_samelink = bridge_samelink[:,3]
stddegreebridge_samelink = bridge_samelink[:,4]
maxdegreebridge_samelink = bridge_samelink[:,6]
mindegreebridge_samelink = bridge_samelink[:,5]

statesbiased_differentlink = biased_differentlink[:,0]
statesstdbiased_differentlink = biased_differentlink[:,1]
clusterstdbiased_differentlink = biased_differentlink[:,2]
avgdegreebiased_differentlink = biased_differentlink[:,3]
stddegreebiased_differentlink = biased_differentlink[:,4]
maxdegreebiased_differentlink = biased_differentlink[:,6]
mindegreebiased_differentlink = biased_differentlink[:,5]

statesbiased_differentlink2 = biased_differentlink2[:,0]
statesstdbiased_differentlink2 = biased_differentlink2[:,1]
clusterstdbiased_differentlink2 = biased_differentlink2[:,2]
avgdegreebiased_differentlink2 = biased_differentlink2[:,3]
stddegreebiased_differentlink2 = biased_differentlink2[:,4]
maxdegreebiased_differentlink2 = biased_differentlink2[:,6]
mindegreebiased_differentlink2 = biased_differentlink2[:,5]

statesbiased_samelink2 = biased_samelink2[:,0]
statesstdbiased_samelink2 = biased_samelink2[:,1]
clusterstdbiased_samelink2 = biased_samelink2[:,2]
avgdegreebiased_samelink2 = biased_samelink2[:,3]
stddegreebiased_samelink2 = biased_samelink2[:,4]
maxdegreebiased_samelink2 = biased_samelink2[:,6]
mindegreebiased_samelink2 = biased_samelink2[:,5]

xaxis = np.array(list(range(len(statesrandomrewiring05)))) / systemsize #this is time

#plotting cooperativity and SD cooperativity
figtype, ax =  plt.subplots()
ax.plot(xaxis,statesrandomrewiring0,label="no rewiring")#, color = 'grey', linestyle = '-')
#ax.plot(xaxis,statesstdrandomrewiring0,label="SD no rewiring", color = 'orange', linestyle = '--')
ax.plot(xaxis,statesrandomrewiring03,label="random rewiring")#, color = 'green', linestyle = '-') #0.3
#ax.plot(xaxis,statesstdrandomrewiring03,label="SD random rewiring", color = 'green', linestyle = '--') #0.3
#ax.plot(xaxis,statesrandomrewiring05,label="p=q=0.5")
#ax.plot(xaxis,statesstdrandomrewiring05,label="std random rewiring 0.5")
#ax.plot(xaxis,statesrandomrewiring07,label="p=q=0.7")
#ax.plot(xaxis,statesstdrandomrewiring07,label="std random rewiring 0.7")
#ax.plot(xaxis,statesrandomrewiring1,label="p=q=1")
#ax.plot(xaxis,statesstdrandomrewiring1,label="std random rewiring 1")
#ax.plot(xaxis, statesbiased_samelink, label = "biased_if_same", color = 'orange', linestyle = '-') #OLD 3 network steps!
#ax.plot(xaxis, statesstdbiased_samelink, label = "SD biased_if_same", color = 'orange', linestyle = '--') #OLD 3 network steps!
ax.plot(xaxis, statesbiased_samelink2, label = "biased_if_same")#, color = 'orange', linestyle = '-') #if_same
#ax.plot(xaxis, statesstdbiased_samelink2, label = "SD biased_if_same", color = 'orange', linestyle = '--') #if_same
#ax.plot(xaxis, statesbiased_differentlink, label = "biased_if_different") #OLD! with 3networksteps
ax.plot(xaxis, statesbiased_differentlink2, label = "biased_if_different")#, color = 'red', linestyle = '-' )
#ax.plot(xaxis, statesstdbiased_differentlink2, label = "SD biased_if_different", color = 'red', linestyle = '--')
ax.plot(xaxis, statesbridge_differentlink, label = "bridge_if_different")#, color = 'green', linestyle = '-') #if_different
#ax.plot(xaxis, statesstdbridge_differentlink, label = "SD bridge_if_different", color = 'green', linestyle = '--') #if_different
ax.plot(xaxis, statesbridge_samelink, label = "bridge_if_same")#, color = 'blue', linestyle = '-')
#ax.plot(xaxis, statesstdbridge_samelink, label = "SD bridge_if_same", color = 'blue', linestyle = '--')
ax.set(xlabel='time [timestep / system size]',ylabel='cooperativity')
#ax.legend(loc = 'lower right')
#ax.legend(loc = 'center left', bbox_to_anchor=(1.04,0.5)) #loc = 'lower right',
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([-1,1])
ax.set_xlim([0,50])


figtype.savefig('6_cooperativity_biased_same+different')

#plotting std states
figtype, ax =  plt.subplots()
ax.plot(xaxis, statesstdbridge_differentlink, label = "bridgerewiring_if_different")
#ax.plot(xaxis, statesstdbiased_samelink, label = "biasedrewiring_if_same")
#ax.plot(xaxis, statesstdbridge_samelink, label = "bridgerewiring_if_same")
ax.plot(xaxis, statesstdbiased_differentlink, label = "biasedrewiring_if_different")
ax.plot(xaxis, statesstdrandomrewiring0, label = "no rewiring")
ax.set(xlabel='time [timestep / system size]',ylabel='cooperativity SD')
ax.legend(loc = 'lower right')
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([-1,1])
ax.set_xlim([0,50])


#figtype.savefig('cooperativitystd_bridge+biased_ifdifferent')

# #plotting degree average and standart distribution 
figtype, ax =  plt.subplots()
# ax.plot(xaxis,avgdegreerandomrewiring05,label="avg degree random rewiring")
# ax.plot(xaxis,stddegreerandomrewiring05,label="std degree random rewiring")
# ax.plot(xaxis,avgdegreerandomrewiring0,label="avg degree no rewiring")
# ax.plot(xaxis,stddegreerandomrewiring0,label="std degree no rewiring")
# ax.set(xlabel='time [timestep / system size]',ylabel='degree')
# ax.legend(loc = 'upper right')
# ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) #ticker sets the little ticks on the axes
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
# ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
# ax.yaxis.grid(True, linestyle='dotted')
# ax.set_ylim([0,40])
# ax.set_xlim([0,50])

#figtype.savefig('avg_std_degree_bridgeprotection_smallnetwork2nd.png')

# #plotting degree average and standart distribution 
figtype, ax =  plt.subplots()
ax.plot(xaxis,avgdegreerandomrewiring0,label="AVGd no rewiring", color='orange', linestyle='-')
ax.plot(xaxis,stddegreerandomrewiring0,label="SDd no rewiring", color='orange', linestyle='--')
# ax.plot(xaxis,avgdegreerandomrewiring03,label="avg degree 0.3")
# ax.plot(xaxis,stddegreerandomrewiring03,label="std degree 0.3")
# ax.plot(xaxis,avgdegreerandomrewiring05,label="AVGd random rewiring", color='green', linestyle='-')
# ax.plot(xaxis,stddegreerandomrewiring05,label="SDd random rewiring", color='green', linestyle='--')
# ax.plot(xaxis,avgdegreerandomrewiring07,label="avg degree 0.7")
# ax.plot(xaxis,stddegreerandomrewiring07,label="std degree 0.7")
# ax.plot(xaxis,avgdegreerandomrewiring1,label="avg degree 1")
# ax.plot(xaxis,stddegreerandomrewiring1,label="std degree 1")
ax.plot(xaxis,avgdegreebridge_differentlink,label="AVGd bridge_if_different", color='green', linestyle='-')
ax.plot(xaxis,stddegreebridge_differentlink,label="SDd bridge_if_different", color='green', linestyle='--')
ax.plot(xaxis,avgdegreebridge_samelink,label="AVGd bridge_if_same", color='blue', linestyle='-')
ax.plot(xaxis,stddegreebridge_samelink,label="SDd bridge_if_same", color='blue', linestyle='--')
# ax.plot(xaxis,avgdegreebiased_differentlink,label="AVGd biased_if_different", color='red', linestyle='-')
# ax.plot(xaxis,stddegreebiased_differentlink,label="SDd biased_if_different", color='red', linestyle='--')
# ax.plot(xaxis,avgdegreebiased_samelink,label="AVGd biased_if_same", color='orange', linestyle='-')
# ax.plot(xaxis,stddegreebiased_samelink,label="SDd biased_if_same", color='orange', linestyle='--')
ax.set(xlabel='time [timestep / system size]',ylabel='(SD) degree')
#ax.legend(loc = 'upper right')
#ax.legend(loc = 'center left', bbox_to_anchor=(1.04,0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([0,40])
ax.set_xlim([0,50])

#figtype.savefig('avg+stdDEGREE_nolegend_norewiring_bridge')

# # #plotting maximum degree
figtype, ax =  plt.subplots()
#ax.plot(xaxis,maxdegreerandomrewiring05,label="max degree 0.5") #!no min and max for 0.5
# #ax.plot(xaxis,maxdegreerandomrewiring03,label="max degree 0.3")
# #ax.plot(xaxis,maxdegreerandomrewiring07,label="max degree 0.7")
#ax.plot(xaxis,maxdegreerandomrewiring1,label="max degree 1")
ax.plot(xaxis,maxdegreerandomrewiring0, label="max degree no rewiring")
ax.plot(xaxis, maxdegreebridge_differentlink, label='max degree bridge different')
ax.plot(xaxis, maxdegreebridge_samelink, label='max degree bridge same')
#ax.plot(xaxis, maxdegreebiased_differentlink, label='max degree biased different')
#ax.plot(xaxis, maxdegreebiased_samelink, label='max degree biased same')
ax.set(xlabel='time [timestep / system size]',ylabel='degree')
ax.legend(loc = 'upper right')
ax.yaxis.set_major_locator(ticker.MultipleLocator(50)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([0,250])
ax.set_xlim([0,50])

#figtype.savefig('maxdegree_bridge_norewiring')

# #plotting minimum degree
figtype, ax =  plt.subplots()
#ax.plot(xaxis,mindegreerandomrewiring05,label="min degree 0.5")
# # ax.plot(xaxis,mindegreerandomrewiring03,label="min degree 0.3")
# # ax.plot(xaxis,mindegreerandomrewiring07,label="min degree 0.7")
#ax.plot(xaxis,mindegreerandomrewiring1,label="min degree 1")
ax.plot(xaxis,mindegreerandomrewiring0, label="min degree no rewiring")
ax.plot(xaxis, mindegreebridge_differentlink, label='min degree bridge different')
ax.plot(xaxis, mindegreebridge_samelink, label='min degree bridge same')
#ax.plot(xaxis, mindegreebiased_differentlink, label='min degree biased different')
#ax.plot(xaxis, mindegreebiased_samelink, label='min degree biased same')
ax.set(xlabel='time [timestep / system size]',ylabel='degree')
ax.legend(loc = 'upper right')
ax.yaxis.set_major_locator(ticker.MultipleLocator(2)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([0,10])
ax.set_xlim([0,50])

#figtype.savefig('mindegree_bridge_norewiring')

#plotting std clusters
figtype, ax =  plt.subplots()
ax.plot(xaxis,clusterstdrandomrewiring05,label="cluster std random0.5")
# ax.plot(xaxis,mindegreerandomrewiring03,label="min degree 0.3")
# ax.plot(xaxis,mindegreerandomrewiring07,label="min degree 0.7")
ax.plot(xaxis,clusterstdrandomrewiring0, label="cluster std no rewiring")
ax.plot(xaxis, clusterstdbridge_differentlink, label='cluster std bridge different')
ax.plot(xaxis, clusterstdbridge_samelink, label='cluster std bridge same')
#ax.plot(xaxis, clusterstdbiased_differentlink, label='cluster std biased different')
#ax.plot(xaxis, clusterstdbiased_samelink, label='cluster std biased same')
ax.set(xlabel='time [timestep / system size]',ylabel='clusterstd')
ax.legend(loc = 'upper right')
ax.yaxis.set_major_locator(ticker.MultipleLocator(2)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([0,5])
ax.set_xlim([0,50])

#figtype.savefig('clusterstd_bridge_norewiring_random')

#trying to plot log of avg cooperativity
figtype, ax =  plt.subplots()
ax.plot(xaxis,np.ln(statesrandomrewiring03),label="random rewiring")
ax.set(xlabel='time [timestep / system size]',ylabel='l cooperativity')
ax.legend(loc = 'upper right')
ax.yaxis.set_major_locator(ticker.MultipleLocator(2)) #ticker sets the little ticks on the axes
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5))
ax.xaxis.grid(True, linestyle='dotted') #makes dotted lines in the background
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([-1,1])
ax.set_xlim([0,50])