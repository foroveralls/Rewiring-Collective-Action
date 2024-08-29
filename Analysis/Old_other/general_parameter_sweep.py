
#
9# Some notes on implementation for better or worse
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
import models_checks_updated
from itertools import product, starmap
import numpy as np
from datetime import date
import pandas as pd
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
    def param_sweep(parameter, param_val):
        #Constants and Variables
        numberOfSimulations = 8
        numberOfProcessors =  int(multiprocessing.cpu_count()*0.6) # CPUs to use for parallelization
         
        start = time.time()
        pool=Pool(processes = numberOfProcessors) #initializing pool
         
         # ----------PATH TO SAVE FIGURES AND DATA-----------
         
        rewiring_list_h = ["diff", "same"]
        directed_topology_list = ["FB", "DPAH"]  
        undirected_topology_list = ["cl"]  
        
        # Create combined list for scenarios "biased" and "bridge" with "diff" and "same"
        # These can be on both directed and undirected networks
        combined_list1 = [(scenario, rewiring, topology)
                      for scenario in ["biased", "bridge"]
                      for rewiring in rewiring_list_h
                      for topology in directed_topology_list + undirected_topology_list]
       
        # Add combinations for "None" scenario with "node2vec" on all topologies
        # "node2vec" works on both directed and undirected
        combined_list2 = [("None", "node2vec", topology) for topology in directed_topology_list + undirected_topology_list]
        
        # Add combinations for "None" scenario with "wtf" only on directed topologies
        combined_list3 = [("None", "wtf", topology) for topology in directed_topology_list]
        
        # Combine all lists
        combined_list = combined_list1 + combined_list2 + combined_list3
        
       
        end_states = []
        for i, v, k in combined_list:
            #print(i, v)
             
             print("Started iteration: ", f"{i}_{v}_{k}")
             
             argList = []
             top_file = "twitter_graph_N_40.gpickle" if k in "twitter" else "twitter_graph_N_40.gpickle"
             ## You can specify simulation parameters here. If they are not set here, they will default to some values set in models.py
             argList.append({"rewiringAlgorithm": i, "rewiringMode": v, "type": k,
                             "top_file": top_file, parameter:param_val})
             #argList.append({"influencers" : 0, "type" : "cl"})
            
             #print (argList)
             
             for j in range(len(argList)):
                   sim = pool.starmap(models_checks_updated.simulate, zip(range(numberOfSimulations), repeat(argList[j])))
               
                   #print(sim[0]. __class__. __name__)
                   #print(sim[0].algo)
                   
                   #fname = f'../Output/{i}_linkif_{v}_top_{j}.csv'
                   #out_list.append(models_checks.saveavgdata(sim, fname, args = argList[0]))
                   [end_states.append([y.states[-1], y.statesds[-1], parameter, param_val, i, v, k, 0]) for y in sim]
           
         
        end = time.time()
        mins = (end - start) / 60
        sec = (end - start) % 60
        print(f'Runtime is complete: {mins:5.0f} mins {sec}s\n')
         
        return end_states
    
        
    
    #%% running sweep
    
    
    parameters = ["polarisingNode_f", "stubbornness"] 
    param_vals = np.linspace(0, 1, 10)
    
  
    today = date.today()
    date = today.strftime("%b_%d_%Y")
    
    # Generate all combinations of parameters and param_vals
    param_combinations = product(parameters, param_vals)
    
    output = []
    
    for param, val in param_combinations:
        run = param_sweep(param, val)  # Assuming param_sweep function exists
        df = pd.DataFrame(run, columns=["state", "state_std", "parameter_val", "rewiring", "mode", "topology", "convergence_speed"])
        output.append(df)

        
    runs_array = pd.concat(output)            
    
    fname = f'../Output/heatmap_sweep_{date}_{"_".join(str(x) for x in parameters)}.csv'
    runs_array.to_csv(fname, index=False)
    
       





