
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
import models_checks_updated as models_checks
import numpy as np 
import pickle 
import concurrent
from datetime import date

from itertools import product 




#random.seed(1574705741)    ## if you need to set a specific random seed put it here
#np.random.seed(1574705741)

date = date.today()
lock = None

#This is magic to present data racing I don't know why it works
def init(lock_):
    models_checks.init_lock(lock_)

if  __name__ ==  '__main__': 

    #Constants and Variables
    numberOfSimulations = 20 
    numberOfProcessors =  int(0.8*multiprocessing.cpu_count()) # CPUs to use for parallelization

    start = time.time()
    lock = multiprocessing.Lock()
    
    pool = multiprocessing.Pool(processes=numberOfProcessors, initializer=init, initargs=(lock,))
    # ----------PATH TO SAVE FIGURES AND DATA-----------

    pathFig = '/Figs'
    pathData = '/Output'
    
    modelargs= models_checks.getargs()  # requires models.py to be imported
    #runs = 4   ## has to be even for multiple runs also n is actually n-1 because I'm lazy


    
    rewiring_list_h = ["diff", "same"]
    directed_topology_list = []#"DPAH"] #"Twitter"]  
    undirected_topology_list = ["cl"] #"FB"]  
    
    # Create combined list for scenarios "biased" and "bridge" with "diff" and "same"
    # These can be on both directed and undirected networks
    combined_list1 = [(scenario, rewiring, topology)
                      for scenario in ["biased", "bridge"]
                      for rewiring in rewiring_list_h
                      for topology in directed_topology_list + undirected_topology_list]
    
    # Add combinations for "None" scenario with "node2vec" on all topologies
    # "node2vec" works on both directed and undirected
    combined_list2 = [("node2vec", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    
    # Add combinations for "None" scenario with "wtf" only on directed topologies
    combined_list3 = [("wtf","None", topology) for topology in directed_topology_list]
    
    # Combine all lists
    combined_list = combined_list1 + [("random", "None", "cl")] + [("None", "None", "cl")] #+ combined_list2 + combined_list3
        
    

    out_list = []
    for i, v, k in combined_list:
      
     
       
        print("Started iteration: ", f"{i}_{v}_{k}")

        argList = []
        if k in "Twitter":
            top_file = "twitter_102.gpickle"
            nwsize = 102
            
        elif k in "FB":
            top_file = "fb_150.gpickle"
            nwsize = 150
        
        else:
            top_file = None
            nwsize = 1000
        
        ## You can specify simulation parameters here. If they are not set here, they will default to some values set in models.py
        argList.append({"rewiringAlgorithm": i, "nwsize": nwsize, "rewiringMode": v, "type": k,
                        "top_file": top_file, "polarisingNode_f": 0, "timesteps": 50000 , "plot": False})
       
        
        #print (argList)
        
        for j in range(len(argList)):
            
            start_1  = time.time()
            sim = pool.starmap(models_checks.simulate, zip(range(numberOfSimulations), repeat(argList[j])))#, repeat(lock)))
        
            assert argList[0]["rewiringAlgorithm"] == str(sim[0].algo), "Inconsistent values"
            # #print(sim[0]. __class__. __name__)
            # #print(sim[0].algo) #sim[0].steps)
            
            fname = f'../Output/{i}_linkif_{v}_top_{j}.csv'
        
            out_list.append(models_checks.saveavgdata(sim, fname, args = argList[0]))
            end_1 = time.time()
            mins = (end_1 - start_1) / 60
            sec = (end_1 - start_1) % 60
            print(f'algorithim run is complete: {mins:5.0f} mins {sec}s\n')
            
    pool.close()
    pool.join()
    
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Runtime is complete: {mins/60}) hours, {mins:5.0f} mins {sec}s\n')
    

#%% post processing
    

    columns = ['avg_state', 'std_states','avgdegree','degreeSD','mindegree','maxdegree','scenario','rewiring','type']
    
    stacked_array = np.vstack(out_list)
    out_list_df = pd.DataFrame(stacked_array)
    out_list_df['t'] = np.tile(np.arange(stacked_array.shape[0] // len(out_list)), len(out_list))    

    out_list_df.columns = columns + ['t']
    # Reorder columns to have 't' as the first column
    out_list_df = out_list_df[['t'] + columns]
    
    #out_list_df.to_csv(f'../Output/default_run_all_new_N_{nwsize}.csv')
    try:
        out_list_df.to_csv(f'../Output/default_run_all_new_N_{nwsize}_{date}.csv')
    except NameError as e:
        # Handle the case where nwsize does not exist
        print(f"Error: {e}. It seems 'nwsize' does not exist.")
        out_list_df.to_csv('../Output/default_run_all_N_default.csv')
    except SyntaxError as e:
        # Handle any potential syntax errors
        print(f"Syntax Error: {e}")
    except Exception as e:
        # Handle any other types of errors
        print(f"An unexpected error occurred: {e}")