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
    numberOfSimulations = 30  # Increased for statistical significance
    numberOfProcessors = int(0.8 * multiprocessing.cpu_count())
    lock = multiprocessing.Lock()
    
    pool = multiprocessing.Pool(processes=numberOfProcessors, initializer=init, initargs=(lock,))
    start = time.time()
    
    # Network configuration
    rewiring_list_h = ["diff", "same"]
    directed_topology_list = ["DPAH", "Twitter"]
    undirected_topology_list = ["cl", "FB"]
    
    # Create combined scenarios list
    combined_list1 = [(scenario, rewiring, topology)
                      for scenario in ["biased", "bridge"]
                      for rewiring in rewiring_list_h
                      for topology in directed_topology_list + undirected_topology_list]
    
    combined_list2 = [("node2vec", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    combined_list3 = [("None", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    combined_list4 = [("wtf", "None", topology) for topology in directed_topology_list]
    combined_list_rand = [("random", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    
    combined_list = combined_list1 + combined_list_rand + combined_list2 + combined_list3 + combined_list4

    # Parameter sweep configuration
    parameter_names = ["polarisingNode_f", "stubbornness"]
    parameters = {
        "polarisingNode_f": np.linspace(0, 1, 10)
    }
    param_product = [dict(zip(parameters.keys(), x)) for x in product(*parameters.values())]

    # Initialize list to store results
    out_list = []
    
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
                "timesteps": 40000,
                "plot": False,
                **params  # Include sweep parameters
            }

            # Run simulations and time them
            start_1 = time.time()
            sim = pool.starmap(models_checks.simulate, 
                             zip(range(numberOfSimulations), repeat(sim_args)))
            
            # Verify consistency
            assert sim_args["rewiringAlgorithm"] == str(sim[0].algo), "Inconsistent values"
            
            # Generate filename with parameter values
            param_str = "_".join([f"{k}_{v:.2f}" for k, v in params.items()])
            fname = f'../Output/{algo}_linkif_{mode}_top_{topology}_{param_str}.csv'
            print("Starting save")
            
            # Save trajectory data
            out_list.append(models_checks.saveavgdata(sim, fname, args=sim_args))
            
            # Print timing information
            end_1 = time.time()
            mins = (end_1 - start_1) / 60
            sec = (end_1 - start_1) % 60
            print(f'Algorithm run complete: {mins:5.0f} mins {round(sec)}s\n')

    pool.close()
    pool.join()

    def process_outputs(out_list, nwsize, params):
        # Unpack the tuples in out_list
        avg_dfs, individual_dfs = zip(*out_list)
        
        # Process averaged data
        combined_avg_df = pd.concat(avg_dfs, ignore_index=True)
        
        # Process individual data
        combined_individual_df = pd.concat(individual_dfs, ignore_index=True)
        
        # Optimize memory usage
        combined_avg_df = combined_avg_df.astype({
            't': 'int32',
            'avg_state': 'float32',
            'std_states': 'float32',
            'avgdegree': 'float32',
            'degreeSD': 'float32',
            'mindegree': 'float32',
            'maxdegree': 'float32',
            'scenario': 'category',
            'rewiring': 'category',
            'type': 'category'
        })
        
        combined_individual_df = combined_individual_df.astype({
            't': 'int32',
            'model_run': 'int32',
            'scenario': 'category',
            'rewiring': 'category',
            'type': 'category'
        })
        
        # Save the combined data with parameter information in filename
        today = date.today()
        param_str = "_".join([f"{k}_{min(v)}_{max(v)}_{len(v)}" for k, v in params.items()])
        avg_output_file = f'../Output/param_sweep_avg_N_{nwsize}_{param_str}_{today}.csv'
        individual_output_file = f'../Output/param_sweep_individual_N_{nwsize}_{param_str}_{today}.csv'
        
        combined_avg_df.to_csv(avg_output_file, index=False)
        combined_individual_df.to_csv(individual_output_file, index=False)
        
        print(f"Averaged output saved to {avg_output_file}")
        print(f"Individual output saved to {individual_output_file}")
        
        return combined_avg_df, combined_individual_df

    # Process and save all outputs
    processed_avg_df, processed_individual_df = process_outputs(out_list, nwsize, parameters)
    
    # Clean up temporary files
    for file in glob.glob("*embeddings*"):
        os.remove(file)

    # Print total runtime
    end = time.time()
    total_hours = (end - start) / 3600
    remaining_mins = ((end - start) % 3600) / 60
    print(f'Total runtime: {int(total_hours)} hours, {int(remaining_mins)} minutes')