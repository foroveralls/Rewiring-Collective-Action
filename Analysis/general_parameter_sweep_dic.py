
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
import models_checks_updated as models_checks
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
    
    def param_sweep(params):
        # Constants and Variables
        numberOfSimulations = 8
        numberOfProcessors = int(multiprocessing.cpu_count() * 0.6)  # CPUs to use for parallelization
        pool = Pool(processes=numberOfProcessors)  # Initializing pool
        start = time.time()
        
        models_checks.nwsize = 40
        
        # Combinations of different simulation conditions
        rewiring_list_h = ["diff", "same"]
        directed_topology_list = ["FB", "DPAH"]
        undirected_topology_list = ["cl"]

        combined_list1 = [(scenario, rewiring, topology)
                          for scenario in ["biased", "bridge"]
                          for rewiring in rewiring_list_h
                          for topology in directed_topology_list + undirected_topology_list]

        combined_list2 = [("None", "node2vec", topology) for topology in directed_topology_list + undirected_topology_list]
        combined_list3 = [("None", "wtf", topology) for topology in directed_topology_list]
        combined_list = combined_list1 #+ combined_list2 + combined_list3

        end_states = []
        for i, v, k in combined_list:
            models_checks.nwsize = 102
            #print("Started iteration: ", f"{i}_{v}_{k}")
            argList = [{"rewiringAlgorithm": i, "rewiringMode": v, "type": k,
                        "top_file": "twitter_102.gpickle", "plot": False, "timesteps":2000}]
            # Update simulation parameters for each set
            for arg in argList:
                print(arg)
                arg.update(params)  # Merge parameter dictionary into args

            for j in range(len(argList)):
                sim = pool.starmap(models_checks.simulate, zip(range(numberOfSimulations), repeat(argList[j])))
                [end_states.append([y.states[-1], y.statesds[-1]] + list(params.values()) + [i, v, k, 0]) for y in sim]

        end = time.time()
        mins = (end - start) / 60
        sec = (end - start) % 60
        print(f'Runtime is complete: {mins:.0f} mins {sec:.0f}s\n')
        return end_states

    
        
    
    #%% running sweep
    
   # Running sweep
    parameter_names = ["polarisingNode_f", "stubbornness"]
    parameters = {name: np.linspace(0, 1, 7) for name in parameter_names}
    param_product = [dict(zip(parameters.keys(), x)) for x in product(*parameters.values())]

    output = []
    i, sweep_length = 0, len(param_product)
    for params in param_product:
        models_checks.nwsize = 102
        print(f"'Sweep {i}/{sweep_length}")
        run = param_sweep(params)
        df = pd.DataFrame(run, columns=["state", "state_std"] + list(parameters.keys()) + ["rewiring", "mode", "topology", "convergence_speed"])
        output.append(df)
        i += 1 

    runs_array = pd.concat(output)
    today = date.today()
    date_str = today.strftime("%b_%d_%Y")
    fname = f'../Output/heatmap_sweep_{date_str}_{"_".join(parameters.keys())}.csv'
    runs_array.to_csv(fname, index=False)





