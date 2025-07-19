import os
import pandas as pd 
from itertools import repeat, product
import time
import multiprocessing
import models_checks
import numpy as np 
import glob
from sweep_utils import get_sweep_id, save_sweep_config
from datetime import date

def init(lock_):
    models_checks.init_lock(lock_)

def get_adaptive_timesteps(algo, topology, mode="None", base=45000):
    factors = {
        ("DPAH", "biased", "same"): 2.0, ("DPAH", "biased", "diff"): 1.4,
        ("DPAH", "bridge", "same"): 1.7, ("DPAH", "bridge", "diff"): 1.4,
        ("Twitter", "biased", "same"): 1.8, ("Twitter", "biased", "diff"): 1.8,
        ("Twitter", "bridge", "same"): 1.8, ("Twitter", "bridge", "diff"): 1.8,
        ("DPAH", "random"): 0.9, ("Twitter", "random"): 0.9,
        ("DPAH", "node2vec"): 1.4, ("Twitter", "node2vec"): 1.0,
        ("DPAH", "wtf"): 0.8, ("Twitter", "wtf"): 1.2,
        ("DPAH", "None"): 1.1, ("Twitter", "None"): 1.0,
        ("cl", "biased", "same"): 1.75, ("cl", "biased", "diff"): 1.9,
        ("cl", "bridge", "same"): 1.1, ("cl", "bridge", "diff"): 1.9,
        ("FB", "biased", "same"): 1.35, ("FB", "biased", "diff"): 1.9,
        ("FB", "bridge", "same"): 1.35, ("FB", "bridge", "diff"): 1.9,
        ("cl", "random"): 0.8, ("FB", "random"): 0.8,
        ("cl", "node2vec"): 0.8, ("FB", "node2vec"): 0.9,
        ("cl", "None"): 0.8, ("FB", "None"): 0.8
    }
    key = (topology, algo, mode) if mode != "None" else (topology, algo)
    factor = factors.get(key, factors.get((topology, algo), 1.0))
    return int(base*factor)

def get_optimal_process_count():
    total_cpus = multiprocessing.cpu_count()
    reserved_cpus = max(2, int(0.25 * total_cpus))
    return max(1, int(0.70 * (total_cpus - reserved_cpus)))

if __name__ == '__main__':
    numberOfSimulations = 30
    numberOfProcessors = get_optimal_process_count()
    lock = multiprocessing.Lock()
    
    pool = multiprocessing.Pool(processes=numberOfProcessors, initializer=init, initargs=(lock,))
    start = time.time()
    
    # Network configuration
    rewiring_list_h = ["diff", "same"]
    directed_topology_list = ["DPAH", "Twitter"]
    undirected_topology_list = ["cl", "FB"]
    
    combined_list1 = [(scenario, rewiring, topology)
                      for scenario in ["biased", "bridge"]
                      for rewiring in rewiring_list_h
                      for topology in directed_topology_list + undirected_topology_list]
    
    combined_list2 = [("node2vec", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    combined_list3 = [("None", "None", topology) for topology in directed_topology_list + undirected_topology_list]
    combined_list4 = [("wtf", "None", topology) for topology in directed_topology_list]
    combined_list_rand = [("random", "None", topology) for topology in directed_topology_list + undirected_topology_list]

    combined_list = combined_list1 + combined_list_rand + combined_list2 + combined_list3 + combined_list4 
    combined_list = [("biased", "diff", "cl"), ("bridge", "diff", "cl") ]
    
    # Parameter sweep configuration
    parameter_names = ["stubbornness", "polarisingNode_f"]
    parameters = {
        "stubbornness": np.linspace(0, 1, 10),
        "polarisingNode_f": np.linspace(0, 1, 10)
    }
    param_product = [dict(zip(parameters.keys(), x)) for x in product(*parameters.values())]

    results = []
    sweep_id = get_sweep_id(parameter_names)
    base_args = models_checks.getargs()
    scenario_times = {}  # Track timing for each scenario

    # Print starting information
    total_params = len(param_product)
    total_scenarios = len(combined_list)
    total_sims = total_params * total_scenarios * numberOfSimulations
    
    print(f"=== PARAMETER SWEEP STARTING ===")
    print(f"Sweep ID: {sweep_id}")
    print(f"Date: {date.today()}")
    print(f"Parameter combinations: {total_params}")
    print(f"Scenarios per combo: {total_scenarios}")
    print(f"Simulations per scenario: {numberOfSimulations}")
    print(f"Total simulations: {total_sims}")
    print(f"Processors: {numberOfProcessors}")
    print("=" * 35)

    save_sweep_config(
        sweep_id=sweep_id,
        parameter_names=parameter_names,
        parameters=parameters,
        combined_list=combined_list,
        num_simulations=numberOfSimulations,
        base_args=base_args
    )

    # Main sweep loop
    for param_idx, params in enumerate(param_product):
        param_start = time.time()
        print(f"[{param_idx+1}/{total_params}] Parameter combo: {params}")
        
        for algo, mode, topology in combined_list:
            scenario_key = (algo, mode, topology)
            scenario_start = time.time()
            print(f"  Running: {algo}_{mode}_{topology}")
            
            if topology == "Twitter":
                top_file = "twitter_graph_N_789.gpickle"
                nwsize = 789
            elif topology == "FB":
                top_file = "FB_graph_N_786.gpickle"
                nwsize = 786
            else:
                top_file = None
                nwsize = 800
                
            adaptive_timesteps = get_adaptive_timesteps(algo, topology)
            
            sim_args = {
                "rewiringAlgorithm": algo,
                "nwsize": nwsize,
                "rewiringMode": mode,
                "type": topology,
                "top_file": top_file,
                "timesteps": adaptive_timesteps, 
                "plot": False,
                "seed": 42,
                **params
            }
            
            complete_args = {**base_args, **sim_args}
            sims = pool.starmap(models_checks.simulate, 
                              zip(range(numberOfSimulations), repeat(complete_args)))
            
            # Track timing for first param combo
            if param_idx == 0:
                scenario_times[scenario_key] = (time.time() - scenario_start) / numberOfSimulations
            
            algos = [str(m.algo) for m in sims]
            if len(set(algos)) > 1:
                raise ValueError(f"Mixed algorithms in batch: {set(algos)}") 
                 
            for sim in sims:
                results.append({
                    'state': sim.states[-1],
                    'state_std': sim.statesds[-1],
                    'stubbornness': params['stubbornness'],
                    'polarisingNode_f': params['polarisingNode_f'],
                    'rewiring': mode,
                    'mode': algo,
                    'topology': topology,
                })
        
        param_elapsed = (time.time() - param_start) / 3600
        total_elapsed = (time.time() - start) / 3600
        
        # Calculate ETA using scenario timings (available after first param combo)
        if scenario_times:
            remaining_params = total_params - param_idx - 1
            remaining_time = sum(scenario_times.values()) * remaining_params * numberOfSimulations / 3600
            eta_str = f"ETA: {remaining_time:.2f}h"
        else:
            eta_str = "ETA: calculating..."
            
        print(f"  Param combo complete: {param_elapsed:.2f}h | Total: {total_elapsed:.2f}h | {eta_str}")
        
        # Show estimated total time after first combo
        if param_idx == 0 and scenario_times:
            estimated_total = sum(scenario_times.values()) * total_params * numberOfSimulations / 3600
            print(f"  Estimated total time: {estimated_total:.1f}h ")
           
           
    pool.close()
    pool.join()

    # Save results
    results_df = pd.DataFrame(results)
    fname = f'../Output/heatmap_sweep_{sweep_id}.csv'
    results_df.to_csv(fname, index=False)

    # Clean up
    for file in glob.glob("*embeddings*"):
        os.remove(file)

    total_hours = (time.time() - start) / 3600
    print(f"\n=== SWEEP COMPLETE ===")
    print(f"Total runtime: {total_hours:.2f} hours")
    print(f"Results saved: {fname}")
    print(f"Total simulations: {len(results_df)}")