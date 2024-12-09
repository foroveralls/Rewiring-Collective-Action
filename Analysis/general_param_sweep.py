# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:31:29 2024

@author: Jordan
"""

import os
import pandas as pd 
from itertools import repeat, product
import time
import multiprocessing
import models_checks
import numpy as np 
import glob
from datetime import date


def init(lock_):
    models_checks.init_lock(lock_)

if __name__ == '__main__':
    # Constants and Variables
    numberOfSimulations = 8
    numberOfProcessors = int(0.8 * multiprocessing.cpu_count())
    lock = multiprocessing.Lock()
    
    pool = multiprocessing.Pool(processes=numberOfProcessors, initializer=init, initargs=(lock,))
    start = time.time()
    
    # Network configuration
    rewiring_list_h = ["diff", "same"]
    directed_topology_list = ["DPAH"]
    undirected_topology_list = ["cl"]
    
    # Create combined scenarios list
    combined_list1 = [(scenario, rewiring, topology)
                      for scenario in ["biased", "bridge"]
                      for rewiring in rewiring_list_h
                      for topology in directed_topology_list + undirected_topology_list]
    
    combined_list2 = [("node2vec", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    combined_list3 = [("None", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    combined_list4 = [("wtf", "None", topology) for topology in directed_topology_list]
    combined_list = combined_list1 + combined_list2 + combined_list3 + combined_list4

    combined_list = [("biased", "diff", "cl"), ("bridge", "diff", "cl") ]
    # Parameter sweep configuration
    parameter_names = ["polarisingNode_f", "stubbornness"]
    parameters = {
        "polarisingNode_f": np.linspace(0, 1, 7),
        "stubbornness": np.linspace(0, 1, 7)
    }
    param_product = [dict(zip(parameters.keys(), x)) for x in product(*parameters.values())]

    # Initialize list to store results
    results = []
    
    # Main sweep loop
    for params in param_product:
        print(f"Started parameter combination: {params}")
        
        for algo, mode, topology in combined_list:
            print(f"Running scenario: {algo}_{mode}_{topology}")
            
            # Set network size based on topology
            if topology == "Twitter":
                top_file = "twitter_graph_N_789.gpickle"
                nwsize = 789
            elif topology == "FB":
                top_file = "FB_graph_N_786.gpickle"
                nwsize = 786
            else:
                top_file = None
                nwsize = 800

            # Prepare simulation arguments
            sim_args = {
                "rewiringAlgorithm": algo,
                "nwsize": nwsize,
                "rewiringMode": mode,
                "type": topology,
                "top_file": top_file,
                "timesteps": 10000,
                "plot": False,
                **params
            }

            # Run simulations
            sims = pool.starmap(models_checks.simulate, 
                              zip(range(numberOfSimulations), repeat(sim_args)))
            
            # Extract final states and standard deviations
            final_states = [sim.states[-1] for sim in sims]
            mean_final_state = np.mean(final_states)
            std_final_state = np.std(final_states)
            
            # Store results
            results.append({
                'state': mean_final_state,
                'state_std': std_final_state,
                'polarisingNode_f': params['polarisingNode_f'],
                'stubbornness': params['stubbornness'],
                'rewiring': mode,  # This matches the plotting script's expectations
                'mode': algo,      # This matches the plotting script's expectations
                'topology': topology
            })

    pool.close()
    pool.join()

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Save results
    today = date.today()
    date_str = today.strftime("%b_%d_%Y")
    fname = f'../Output/heatmap_sweep_{date_str}_{"_".join(parameters.keys())}.csv'
    results_df.to_csv(fname, index=False)

    # Clean up any temporary files
    for file in glob.glob("*embeddings*"):
        os.remove(file)

    # Print total runtime
    end = time.time()
    total_hours = (end - start) / 3600
    remaining_mins = ((end - start) % 3600) / 60
    print(f'Total runtime: {int(total_hours)} hours, {int(remaining_mins)} minutes')