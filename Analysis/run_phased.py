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
import glob, gzip, pickle
from sweep_utils import get_sweep_id

def init(lock_):
    models_checks.init_lock(lock_)

# Per-condition timestep override for one-off horizon-corrected reruns.
# bridge/diff/FB needs 270k (confirmed by the convergence diagnostic), not the
# fixed 160k used elsewhere -- see
# claude_stuff/Infrastructure/convergence_diagnostic_criteria_2026-07-03.md section 8c/9.
TIMESTEPS_OVERRIDE = {
    ("bridge", "diff", "FB"): 270000,
}

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
    
    algo_name = algo_scenarios[0][0].upper()
    print(f"\n=== PHASE: {algo_name} ===")
    print(f"Scenarios: {len(algo_scenarios)}")
    print(f"Sims per scenario: {numberOfSimulations}")
    print(f"Processors: {numberOfProcessors}")
    
    phase_results = []
    
    # Group by topology within this algorithm
    topology_groups = {}
    for scenario, rewiring, topology in algo_scenarios:
        if topology not in topology_groups:
            topology_groups[topology] = []
        topology_groups[topology].append((scenario, rewiring, topology))
    
    # Run each topology sequentially
    for topology, topo_scenarios in topology_groups.items():
        print(f"\n  Topology: {topology} ({len(topo_scenarios)} scenarios)")
        
        lock = multiprocessing.Lock()
        with multiprocessing.Pool(processes=numberOfProcessors, initializer=init, initargs=(lock,)) as pool:
            
            for scenario, rewiring, topology_name in topo_scenarios:
                start_scenario = time.time()
                print(f"    Running: {scenario}_{rewiring}_{topology_name}")
                
                # Network config
                if topology_name == "Twitter":
                    top_file, nwsize = "twitter_graph_N_789.gpickle", 789
                elif topology_name == "FB":
                    top_file, nwsize = "FB_graph_N_786.gpickle", 786
                else:
                    top_file, nwsize = None, 800
                
                timesteps = TIMESTEPS_OVERRIDE.get((scenario, rewiring, topology_name), 160000)
                sim_args = {
                    "rewiringAlgorithm": scenario, "nwsize": nwsize, "rewiringMode": rewiring,
                    "type": topology_name, "top_file": top_file, "polarisingNode_f": 0.10,
                    "timesteps": timesteps, "plot": False, "save_snapshots": True
                }
                
                complete_args = {**base_args, **sim_args}
                
                # Run simulations
                results = pool.starmap(models_checks.simulate, 
                                     zip(range(numberOfSimulations), repeat(complete_args)))
                
                # Separate models and snapshots
                sims = [r[0] if isinstance(r, tuple) else r for r in results]
                scenario_snapshots = {i: r[1] for i, r in enumerate(results) if isinstance(r, tuple)}
                
                # Verify consistency
                algos = [str(m.algo) for m in sims]
                if len(set(algos)) > 1:
                    raise ValueError(f"Mixed algorithms in {scenario}_{rewiring}_{topology_name}: {set(algos)}")
                assert sim_args["rewiringAlgorithm"] == str(sims[0].algo), "Inconsistent algo"
                
                # Save snapshots if collected
                if scenario_snapshots:
                    fname = f'../Output/snapshots_{scenario}_{rewiring}_{topology_name}.pkl.gz'
                    snapshot_data = {
                        'metadata': {'algo': scenario, 'mode': rewiring, 'topology': topology_name, 
                                   'params': sim_args},
                        'snapshots': scenario_snapshots
                    }
                    with gzip.open(fname, 'wb') as f:
                        pickle.dump(snapshot_data, f)
                    print(f"      Saved {len(scenario_snapshots)} snapshot sets to {fname}")
                
                # Save trajectory results
                fname = f'../Output/{scenario}_linkif_{rewiring}_top_{topology_name}.csv'
                result = models_checks.saveavgdata(sims, fname, args=sim_args)
                phase_results.append(result)
                
                elapsed = (time.time() - start_scenario) / 60
                print(f"      Completed in {elapsed:.1f} min")
            
            pool.close()
            pool.join()
        
        import gc
        gc.collect()
        time.sleep(0.1)
    
    return phase_results

def main():
    start = time.time()
    numberOfSimulations = 90
    sweep_id = get_sweep_id(["phased_run"])
    
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
    # One-off rerun: bridge/diff/FB needs the corrected 270k horizon (see
    # TIMESTEPS_OVERRIDE above); every other condition is unaffected by the
    # convergence diagnostic and stays at its existing 160k campaign result.
    # Restore the full combined_list above to regenerate the whole campaign.
    combined_list = [("bridge", "diff", "FB")]
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
        avg_file = f'../Output/default_run_avg_N_{nwsize}_n_{numberOfSimulations}_pNf_{base_args["polarisingNode_f"]}_pc_{models_checks.politicalClimate}_{sweep_id}_{date.today()}.csv'
        individual_file = f'../Output/default_run_individual_N_{nwsize}_n_{numberOfSimulations}_pNf_{base_args["polarisingNode_f"]}_pc_{models_checks.politicalClimate}_{sweep_id}_{date.today()}.csv'
        
        combined_avg_df.to_csv(avg_file, index=False)
        combined_individual_df.to_csv(individual_file, index=False)
        
        print(f"Averaged output: {avg_file}")
        print(f"Individual output: {individual_file}")
        
        return combined_avg_df, combined_individual_df
    
    snapshot_files = glob.glob("../Output/snapshots_*.pkl.gz")
    if snapshot_files:
        print(f"\nConsolidating {len(snapshot_files)} snapshot files...")
        master_snapshots = {}
        
        for f in snapshot_files:
            with gzip.open(f, 'rb') as gz:
                data = pickle.load(gz)
                scenario_key = f"{data['metadata']['algo']}_{data['metadata']['mode']}_{data['metadata']['topology']}"
                master_snapshots[scenario_key] = data
            os.remove(f)  # Delete individual file
        
        # Save consolidated file
        master_file = f'../Output/all_snapshots_{sweep_id}_{date.today()}.pkl.gz'
        with gzip.open(master_file, 'wb') as f:
            pickle.dump(master_snapshots, f)
        
        print(f"Consolidated snapshots saved to {master_file}")
    
    
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