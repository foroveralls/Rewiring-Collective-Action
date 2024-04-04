
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
import pandas as pd 
from scipy.stats import truncnorm
from itertools import repeat
import time
import multiprocessing
from pathlib import Path
import dill
import models_checks
import numpy as np 
import pickle 

from itertools import product 




#random.seed(1574705741)    ## if you need to set a specific random seed put it here
#np.random.seed(1574705741)


if  __name__ ==  '__main__': 

    #Constants and Variables
    numberOfSimulations = 8
    numberOfProcessors =  int(multiprocessing.cpu_count()*0.6) # CPUs to use for parallelization

    start = time.time()
    pool=Pool(processes = numberOfProcessors) #initializing pool
    
    # ----------PATH TO SAVE FIGURES AND DATA-----------

    pathFig = '/Figs'
    pathData = '/Output'
    
    modelargs= models_checks.getargs()  # requires models.py to be imported
    runs = 2   ## has to be even for multiple runs also n is actually n-1 because I'm lazy

   
    scenario_list = ["biased", "bridge"]
    rewiring_list = ["diff", "same"]
    topology_list = ["FB", "cl", "DPAH"]
    topology_files = ["twitter_graph_N_40.gpickle"]
    
    combined_list = list(product(scenario_list, rewiring_list,  topology_list))
    #combined_list.append(("random", "NA"))
    
    out_list = []
    for i, v, k in combined_list:
        #print(i, v)
        
        print("Started iteration: ", f"{i}_{v}_{k}")

        argList = []
        
        ## You can specify simulation parameters here. If they are not set here, they will default to some values set in models.py
        argList.append({"rewiringAlgorithm": i, "rewiringMode": v, "type": k,
                        "top_file": topology_files[0]})
        #argList.append({"influencers" : 0, "type" : "cl"})
       
        #print (argList)
        
        for j in range(len(argList)):
            sim = pool.starmap(models_checks.simulate, zip(range(numberOfSimulations), repeat(argList[j])))
        
            #print(sim[0]. __class__. __name__)
            #print(sim[0].algo)
            
            fname = f'../Output/{i}_linkif_{v}_top_{j}.csv'
            out_list.append(models_checks.saveavgdata(sim, fname, args = argList[0]))

    
    
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Runtime was complete: {mins:5.0f} mins {sec}s\n')
    

#%% post processing
    

    columns = ['avg_state', 'std_states','avgdegree','degreeSD','mindegree','maxdegree','scenario','rewiring','type']
    
    stacked_array = np.vstack(out_list)
    out_list_df = pd.DataFrame(stacked_array)
    out_list_df['t'] = np.tile(np.arange(stacked_array.shape[0] // len(out_list)), len(out_list))    

    out_list_df.columns = columns + ['t']
    # Reorder columns to have 't' as the first column
    out_list_df = out_list_df[['t'] + columns]
    
    out_list_df.to_csv('../Output/default_run_all_1.csv')
