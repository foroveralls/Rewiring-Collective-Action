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
import gc

def init(lock_):
    models_checks.init_lock(lock_)

def get_adaptive_timesteps(algo, topology, mode="None", base=45000):
    factors = {
        ("DPAH", "biased", "same"): 3.0, ("DPAH", "biased", "diff"): 1.4,
        ("DPAH", "bridge", "same"): 2.0, ("DPAH", "bridge", "diff"): 1.4,
        ("Twitter", "biased", "same"): 3.0, ("Twitter", "biased", "diff"): 3.0,
        ("Twitter", "bridge", "same"): 3.0, ("Twitter", "bridge", "diff"): 3.0,
        ("DPAH", "random"): 1, ("Twitter", "random"): 0.9,
        ("DPAH", "node2vec"): 1.2, ("Twitter", "node2vec"): 1.4,
        ("DPAH", "wtf"): 2.0, ("Twitter", "wtf"): 1.3,
        ("DPAH", "None"): 1.1, ("Twitter", "None"): 1.3,
        ("cl", "biased", "same"): 2.0, ("cl", "biased", "diff"): 3.0,
        ("cl", "bridge", "same"): 2.0, ("cl", "bridge", "diff"): 3.0,
        ("FB", "biased", "same"): 3.0, ("FB", "biased", "diff"): 3.0,
        ("FB", "bridge", "same"): 3.0, ("FB", "bridge", "diff"): 3.0,
        ("cl", "random"): 0.9, ("FB", "random"): 0.9,
        ("cl", "node2vec"): 0.9, ("FB", "node2vec"): 0.9,
        ("cl", "None"): 0.9, ("FB", "None"): 0.8
    }
    key = (topology, algo, mode) if mode != "None" else (topology, algo)
    factor = factors.get(key, factors.get((topology, algo), 1.0))
    return int(base*factor)

def get_optimal_process_count():
    total_cpus = multiprocessing.cpu_count()
    reserved_cpus = max(2, int(0.25 * total_cpus))
    return max(1, int(0.75 * (total_cpus - reserved_cpus)))

def group_scenarios_by_algorithm(combined_list):
    """Group scenarios by algorithm for phased execution - isolates biased/bridge subvariants"""
    algo_groups = {}
    for scenario, rewiring, topology in combined_list:
        if scenario in ["biased", "bridge"] and rewiring in ["same", "diff"]:
            key = f"{scenario}_{rewiring}"
        else:
            key = scenario

        if key not in algo_groups:
            algo_groups[key] = []
        algo_groups[key].append((scenario, rewiring, topology))
    return algo_groups

def run_algorithm_phase(algo_scenarios, numberOfSimulations, base_args, params):
    """Run all scenarios for a single algorithm phase"""
    numberOfProcessors = get_optimal_process_count()

    phase_results = []
    topology_groups = {}
    for scenario in algo_scenarios:
        topology = scenario[2]
        if topology not in topology_groups:
            topology_groups[topology] = []
        topology_groups[topology].append(scenario)

    for topology, topo_scenarios in topology_groups.items():
        lock = multiprocessing.Lock()
        with multiprocessing.Pool(processes=numberOfProcessors, initializer=init, initargs=(lock,)) as pool:

            for algo, mode, topo in topo_scenarios:
                if topo == "Twitter":
                    top_file, nwsize = "twitter_graph_N_789.gpickle", 789
                elif topo == "FB":
                    top_file, nwsize = "FB_graph_N_786.gpickle", 786
                else:
                    top_file, nwsize = None, 800

                adaptive_timesteps = get_adaptive_timesteps(algo, topo)

                sim_args = {
                    "rewiringAlgorithm": algo, "nwsize": nwsize, "rewiringMode": mode,
                    "type": topo, "top_file": top_file, "timesteps": adaptive_timesteps,
                    "plot": False, "seed": 42, **params
                }

                complete_args = {**base_args, **sim_args}
                sims = pool.starmap(models_checks.simulate,
                                  zip(range(numberOfSimulations), repeat(complete_args)))

                algos = [str(m.algo) for m in sims]
                if len(set(algos)) > 1:
                    raise ValueError(f"Mixed algorithms in batch: {set(algos)}")

                for sim in sims:
                    phase_results.append({
                        'state': sim.states[-1],
                        'state_std': sim.statesds[-1],
                        'stubbornness': params['stubbornness'],
                        'polarisingNode_f': params['polarisingNode_f'],
                        'rewiring': mode,
                        'mode': algo,
                        'topology': topo,
                    })

            pool.close()
            pool.join()

        gc.collect()
        time.sleep(0.1)

    return phase_results

if __name__ == '__main__':
    numberOfSimulations = 30
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

    # Parameter sweep configuration
    parameter_names = ["stubbornness", "polarisingNode_f"]
    parameters = {
        "stubbornness": np.linspace(0, 1, 10),
        "polarisingNode_f": np.linspace(0, 1, 10)
    }
    param_product = [dict(zip(parameters.keys(), x)) for x in product(*parameters.values())]

    # Group scenarios by algorithm for phased execution
    algo_groups = group_scenarios_by_algorithm(combined_list)

    results = []
    sweep_id = get_sweep_id(parameter_names)
    base_args = models_checks.getargs()

    total_params = len(param_product)
    total_scenarios = len(combined_list)
    total_sims = total_params * total_scenarios * numberOfSimulations

    print(f"=== PHASED PARAMETER SWEEP STARTING ===")
    print(f"Sweep ID: {sweep_id}")
    print(f"Date: {date.today()}")
    print(f"Parameter combinations: {total_params}")
    print(f"Algorithm groups: {list(algo_groups.keys())}")
    print(f"Total simulations: {total_sims}")
    print("=" * 40)

    save_sweep_config(
        sweep_id=sweep_id,
        parameter_names=parameter_names,
        parameters=parameters,
        combined_list=combined_list,
        num_simulations=numberOfSimulations,
        base_args=base_args
    )

    # Main sweep loop with phased execution
    for param_idx, params in enumerate(param_product):
        param_start = time.time()
        print(f"[{param_idx+1}/{total_params}] Parameter combo: {params}")

        # Run each algorithm group sequentially
        for algo_name, algo_scenarios in algo_groups.items():
            print(f"  Phase: {algo_name} ({len(algo_scenarios)} scenarios)")
            phase_results = run_algorithm_phase(algo_scenarios, numberOfSimulations, base_args, params)
            results.extend(phase_results)

        param_elapsed = (time.time() - param_start) / 3600
        total_elapsed = (time.time() - start) / 3600
        remaining_params = total_params - param_idx - 1
        eta = param_elapsed * remaining_params

        print(f"  Param combo complete: {param_elapsed:.2f}h | Total: {total_elapsed:.2f}h | ETA: {eta:.2f}h")

    # Save results
    results_df = pd.DataFrame(results)
    fname = f'../Output/heatmap_sweep_phased_{sweep_id}.csv'
    results_df.to_csv(fname, index=False)

    # Clean up
    for file in glob.glob("*embeddings*"):
        os.remove(file)

    total_hours = (time.time() - start) / 3600
    print(f"\n=== PHASED SWEEP COMPLETE ===")
    print(f"Total runtime: {total_hours:.2f} hours")
    print(f"Results saved: {fname}")
    print(f"Total simulations: {len(results_df)}")
    print(f"Algorithm groups executed: {list(algo_groups.keys())}")
