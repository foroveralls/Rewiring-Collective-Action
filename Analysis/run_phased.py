#!/usr/bin/env python3
"""
Phased run script with isolated biased/bridge subvariants - executes algorithms sequentially to avoid race conditions
Compatible with existing plotting scripts like plots_lines.py
"""

import os
import pandas as pd 
from itertools import repeat
import time
import multiprocessing
import models_checks 
import numpy as np 
from datetime import date
import glob

def init(lock_):
    models_checks.init_lock(lock_)

def get_optimal_process_count():
    total_cpus = multiprocessing.cpu_count()
    reserved_cpus = max(2, int(0.25 * total_cpus))
    return max(1, int(0.75 * (total_cpus - reserved_cpus)))

def group_scenarios_by_algorithm(combined_list):
    """Group scenarios by algorithm for phased execution - isolates biased/bridge subvariants"""
    algo_groups = {}
    for scenario, rewiring, topology in combined_list:
        # Create separate groups for biased/bridge subvariants
        if scenario in ["biased", "bridge"] and rewiring in ["same", "diff"]:
            key = f"{scenario}_{rewiring}"
        else:
            key = scenario
            
        if key not in algo_groups:
            algo_groups[key] = []
        algo_groups[key].append((scenario, rewiring, topology))
    return algo_groups

def run_algorithm_phase(algo_scenarios, numberOfSimulations, base_args):
    """Run all scenarios for a single algorithm - one topology at a time"""
    numberOfProcessors = get_optimal_process_count()
    
    # Fix: Get algorithm name correctly
    algo_name = algo_scenarios[0][0].upper()
    print(f"\n=== PHASE: {algo_name} ===")
    print(f"Scenarios: {len(algo_scenarios)}")
    print(f"Sims per scenario: {numberOfSimulations}")
    print(f"Processors: {numberOfProcessors}")
    
    phase_results = []
    
    # Group by topology within this algorithm
    topology_groups = {}
    for scenario, rewiring, topology in algo_scenarios:  # Fix: Unpack explicitly
        if topology not in topology_groups:
            topology_groups[topology] = []
        topology_groups[topology].append((scenario, rewiring, topology))
    
    # Debug: Print topology groups
    for topo, scenarios in topology_groups.items():
        print(f"  DEBUG: {topo} has {len(scenarios)} scenarios: {[(s[0], s[1]) for s in scenarios]}")
    
    # Run each topology sequentially
    for topology, topo_scenarios in topology_groups.items():
        print(f"\n  Topology: {topology} ({len(topo_scenarios)} scenarios)")
        
        # Create fresh pool for each topology
        lock = multiprocessing.Lock()
        with multiprocessing.Pool(processes=numberOfProcessors, initializer=init, initargs=(lock,)) as pool:
            
            for scenario, rewiring, topology_name in topo_scenarios:  # Fix: Unpack explicitly
                start_scenario = time.time()
                print(f"    Running: {scenario}_{rewiring}_{topology_name}")
                
                # Network config (unchanged)
                if topology_name == "Twitter":
                    top_file, nwsize = "twitter_graph_N_789.gpickle", 789
                elif topology_name == "FB":
                    top_file, nwsize = "FB_graph_N_786.gpickle", 786
                else:
                    top_file, nwsize = None, 800
                
                sim_args = {
                    "rewiringAlgorithm": scenario, "nwsize": nwsize, "rewiringMode": rewiring, 
                    "type": topology_name, "top_file": top_file, "polarisingNode_f": 0.10, 
                    "timesteps": 60000, "plot": False
                }
                
                complete_args = {**base_args, **sim_args}
                
                # Run simulations for this scenario
                sims = pool.starmap(models_checks.simulate, 
                                   zip(range(numberOfSimulations), repeat(complete_args)))
                
                # Verify consistency (unchanged)
                algos = [str(m.algo) for m in sims]
                if len(set(algos)) > 1:
                    raise ValueError(f"Mixed algorithms in {scenario}_{rewiring}_{topology_name}: {set(algos)}")
                
                assert sim_args["rewiringAlgorithm"] == str(sims[0].algo), "Inconsistent algo"
                
                # Save and collect results (unchanged)
                fname = f'../Output/{scenario}_linkif_{rewiring}_top_{topology_name}.csv'
                result = models_checks.saveavgdata(sims, fname, args=sim_args)
                phase_results.append(result)
                
                elapsed = (time.time() - start_scenario) / 60
                print(f"      Completed in {elapsed:.1f} min")
            
            # Explicit pool cleanup for this topology
            pool.close()
            pool.join()
        
        # Force cleanup between topologies
        import gc
        gc.collect()
        time.sleep(0.1)
    
    return phase_results

def main():
    start = time.time()
    numberOfSimulations = 90
    
    print("=== PHASED EXECUTION RUN (ISOLATED SUBVARIANTS) ===")
    print(f"Date: {date.today()}")
    print(f"Simulations per scenario: {numberOfSimulations}")
    
    # Define all scenarios
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
    
    # Group scenarios by algorithm (now isolates subvariants)
    algo_groups = group_scenarios_by_algorithm(combined_list)
    base_args = models_checks.getargs()
    
    print(f"Algorithms to run: {list(algo_groups.keys())}")
    print(f"Total scenarios: {sum(len(scenarios) for scenarios in algo_groups.values())}")
    
    # Execute each algorithm phase sequentially
    all_results = []
    for algo, scenarios in algo_groups.items():
        phase_start = time.time()
        phase_results = run_algorithm_phase(scenarios, numberOfSimulations, base_args)
        all_results.extend(phase_results)
        
        phase_elapsed = (time.time() - phase_start) / 60
        print(f"Phase {algo} completed in {phase_elapsed:.1f} min")
    
    # Process and save combined outputs (same as original run.py)
    def process_outputs(out_list, nwsize):
        avg_dfs, individual_dfs = zip(*out_list)
        
        combined_avg_df = pd.concat(avg_dfs, ignore_index=True)
        combined_individual_df = pd.concat(individual_dfs, ignore_index=True)
        
        # Optimize data types
        combined_avg_df = combined_avg_df.astype({
            't': 'int32', 'avg_state': 'float32', 'std_states': 'float32',
            'scenario': 'category', 'rewiring': 'category', 'type': 'category'
        })
        
        combined_individual_df = combined_individual_df.astype({
            't': 'int32', 'model_run': 'int32', 'scenario': 'category', 
            'rewiring': 'category', 'type': 'category'
        })
        
        # Save files (compatible with plots_lines.py)
        avg_file = f'../Output/phased_isolated_run_avg_N_{nwsize}_n_{numberOfSimulations}_pNf_{base_args["polarisingNode_f"]}_pc_{models_checks.politicalClimate}_{date.today()}.csv'
        individual_file = f'../Output/phased_isolated_run_individual_N_{nwsize}_n_{numberOfSimulations}_pNf_{base_args["polarisingNode_f"]}_pc_{models_checks.politicalClimate}_{date.today()}.csv'
        
        combined_avg_df.to_csv(avg_file, index=False)
        combined_individual_df.to_csv(individual_file, index=False)
        
        print(f"Averaged output: {avg_file}")
        print(f"Individual output: {individual_file}")
        
        return combined_avg_df, combined_individual_df
    
    # Process all results
    nwsize = 800  # Default, gets overridden by empirical networks
    processed_avg_df, processed_individual_df = process_outputs(all_results, nwsize)
    
    # Cleanup
    for file in glob.glob("*embeddings*"):
        os.remove(file)
    
    # Summary
    total_hours = (time.time() - start) / 3600
    total_sims = len(combined_list) * numberOfSimulations
    
    print(f"\n=== EXECUTION COMPLETE ===")
    print(f"Total runtime: {total_hours:.2f} hours")
    print(f"Total simulations: {total_sims}")
    print(f"Algorithms executed: {list(algo_groups.keys())}")
    print(f"Average time per sim: {(total_hours * 3600) / total_sims:.1f} seconds")

if __name__ == "__main__":
    main()