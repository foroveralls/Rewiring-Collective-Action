
#
# Some notes on implementation for better or worse
# 
# Original program written by Sigrid Bratsberg, credit where credit is due! 
# 
# Switching from printing to a single file to making multiple out outs for a given
# set of input values is enabled by decommenting the newvar lines
# 
# TODO
# Someone should move all the parameters to the main program
#
#paramater anaylsis for rewiring values
#import pandas
import os
import sys

#sys.path.append('C:\/Users\everall\Documents\Python\Projects\It-s_how_we_talk_that_matters')
#os.chdir('C:\/Users\everall\Documents\Python\Projects\It-s_how_we_talk_that_matters')

from multiprocessing import Pool
import random
import matplotlib
import matplotlib.pyplot as plt
#from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from copy import deepcopy
#import seaborn as sns
#import pygraphviz as pgv
from statistics import stdev, mean
import imageio
from scipy.stats import truncnorm
from itertools import repeat
import time
import multiprocessing
from pathlib import Path
import dill
import models_checks
from itertools import product 




#random.seed(1574705741)    ## if you need to set a specific random seed put it here
#np.random.seed(1574705741)


if  __name__ ==  '__main__': 

    #Constants and Variables
    numberOfSimulations = 80
    numberOfProcessors =  int(multiprocessing.cpu_count()*0.6) # CPUs to use for parallelization

    start = time.time()
    pool=Pool(processes = numberOfProcessors) #initializing pool
    
    # ----------PATH TO SAVE FIGURES AND DATA-----------

    pathFig = '/Figs'
    pathData = '/Output'
    
    modelargs= models_checks.getargs()  # requires models.py to be imported
    runs = 2   ## has to be even for multiple runs also n is actually n-1 because I'm lazy

    ### comment out all below for single run
    
    ### linear grid 
    #grid = [3]

    ### log grid, only valid on range [-1,1] atm

    #var = "newPoliticalClimate"

    #steps = int(runs/2)
    #start = modelargs[var]
    #endup = 0.10
    #enddw = -0.45
    #logendup = np.log(endup+(1.0-start))
    #logenddw = np.log(enddw+(1.0-start))
    #stepup = logendup / steps
    #stepdw = logenddw / steps

    #gridup = np.array([])
    #griddw = np.array([])

    #for k in range (steps) :
    #    pt = np.exp(stepup*k)
    #    gridup = np.append(gridup,pt)
    #
    #for k in range (steps) :
    #    pt = np.exp(stepdw*k)
    #    griddw = np.append(griddw,pt)

    #gridup = gridup - (1.0-start)
    #griddw = griddw - (1.0-start)

    #griddw = griddw[1:]
    #griddw = np.flip(griddw)

    #grid = np.append(griddw,gridup)

    #print (grid)

    #for run in range(runs-1) :
        
    scenario_list = ["biased", "bridge"]
    rewiring_list = ["diff", "same"]
    
    combined_list = list(product(scenario_list, rewiring_list))
    #combined_list.append(("random", "NA"))
    
    for i, v in combined_list:
        #print(i, v)
        
        print("Started iteration: ", f"{i}_{v}")

        argList = []
        
        ## You can specify simulation parameters here. If they are not set here, they will default to some values set in models.py
        argList.append({"rewiringAlgorithm": i, "rewiringMode": v})
        #argList.append({"influencers" : 0, "type" : "cl"})
       
        #print (argList)

        for j in range(len(argList)):
            sim = pool.starmap(models_checks.simulate, zip(range(numberOfSimulations), repeat(argList[j])))
            #print(sim[0])
            #print(sim[0].algo)
            
            fname = f'./Output/{i}_linkif_{v}.csv'
            models_checks.saveavgdata(sim, fname)

    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Runtime was complete: {mins:5.0f} mins {sec}s\n')
