
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
from datetime import date
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
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
from itertools import product, starmap
import numpy as np
import argparse

#%%

# parser = argparse.ArgumentParser()
# parser.add_argument("parameter")
# parser.add_argument("parameter")

# import argparse
# args = parser.parse_args()


#%%
if __name__ == '__main__':
        
    #random.seed(1574705741)    ## if you need to set a specific random seed put it here
    #np.random.seed(1574705741)
    def param_sweep(parameter, param_vals):
        #Constants and Variables
        numberOfSimulations = 100
        numberOfProcessors =  multiprocessing.cpu_count() #int(multiprocessing.cpu_count()*0.6) # CPUs to use for parallelization
        
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
            
        scenario_list = ["bridge"]#["random"]
        combined_list = list(product(scenario_list, param_vals))    
        
        end_states = []

        for i, v in combined_list:
            #print(i, v)
            
            print("Started iteration: ", f"{i}_0.5_{parameter}_{v}")
        
            argList = []
            
            ## You can specify simulation parameters here. If they are not set here, they will default to some values set in models.py
            argList.append({"rewiringAlgorithm": i, parameter:v, "breaklinkprob": 0.5,
                            "establishlinkprob": 0.5})
           
            #print (argList)
            
            for j in range(len(argList)):
                sim = pool.starmap(models_checks.simulate, zip(range(numberOfSimulations), repeat(argList[j])))
                #print(sim[0].algo, sim[0].probs)
                            
                [end_states.append([y.states[-1], parameter, i, v]) for y in sim]
                
                
        end = time.time()
        mins = (end - start) / 60
        sec = (end - start) % 60
        print(f'Runtime was complete: {mins:5.0f} mins {sec}s\n')
        return end_states
     
    #%% running sweep

    
    parameters = ["stubbornness"] 
    param_vals = [np.linspace(0.01, 1.00, 50)] #[np.linspace(0.1,1,10),
    
    
    today = date.today()
    date = today.strftime("%b_%d_%Y")
    
    output = []
    for i, j in zip(parameters, param_vals):    
        test = param_sweep(i, j)
        df = pd.DataFrame(test, columns = ["state", "parameter", "scenario", "value"])
        output.append(df)
        
    runs_array = pd.concat(output)            
    
    fname = f'./Output/heatmap_sweep_{date}_{scenario_list[0]}_{"_".join(str(x) for x in parameters)}.csv'
    runs_array.to_csv(fname, index=False)
    
       



