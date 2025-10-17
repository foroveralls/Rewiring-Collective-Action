"""
Detailed trace of rewiring execution to find where it fails
"""
import sys
sys.path.append('../..')
sys.path.append('..')

import numpy as np
import random
import networkx as nx
from scipy.stats import truncnorm
import models_checks
from copy import deepcopy

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def trace_single_rewiring():
    """Trace a single rewiring attempt step-by-step"""

    print("="*60)
    print("TRACING SINGLE L-SIM REWIRING ATTEMPT")
    print("="*60)

    # Base parameters
    base_params = {
        "type": "DPAH",
        "nwsize": 100,
        "degree": 5,
        "polarisingNode_f": 0.10,
        "timesteps": 10,
        "plot": False,
        "friendship": 0.5,
        "friendshipSD": 0.19,
        "skew": -0.25,
        "initSD": 0.15,
        "stubbornness": 0.6,
        "politicalClimate": 0.05,
        "newPoliticalClimate": 0.05,
        "randomness": 0.10,
        "continuous": True,
        "clustering": 0.5,
        "defectorUtility": 0.0,
        "wtf_freq": 10,
        "breaklinkprob": 1,
        "establishlinkprob": 0.5,
        "seed": 42,
        "f_all": 0.5,
        "save_snapshots": False,
        "rewiringAlgorithm": "biased",
        "rewiringMode": "same",
    }

    np.random.seed(42)
    random.seed(42)

    friendshipWeightGenerator = get_truncated_normal(base_params["friendship"], base_params["friendshipSD"], 0, 1)
    initialStateGenerator = get_truncated_normal(base_params["skew"], base_params["initSD"], -1, 1)

    model = models_checks.DPAHModel(base_params["nwsize"], base_params["degree"],
                                    skew=base_params["skew"],
                                    friendshipWeightGenerator=friendshipWeightGenerator,
                                    initialStateGenerator=initialStateGenerator,
                                    args=base_params)

    # Pick a node with decent connectivity
    test_node = 40
    print(f"\nTesting node: {test_node}")

    # Step 1: Get current neighbors
    neighbors = list(model.graph.adj[test_node].keys())
    print(f"1. Current neighbors: {len(neighbors)} - {neighbors}")

    # Step 2: Get node opinion
    node = model.graph.nodes[test_node]['agent']
    node_opinion = node.state
    print(f"2. Node opinion: {node_opinion:.3f} (cooperator={node_opinion >= 0})")

    # Step 3: Find 2-step neighbors
    total_neighbours = model.find_2_steps(test_node)
    print(f"3. 2-step neighbors: {len(total_neighbours)}")

    # Step 4: Get non-neighbors
    non_neighbors = [k for k in nx.non_neighbors(model.graph, test_node)]
    print(f"4. Non-neighbors: {len(non_neighbors)}")

    # Step 5: Find potential friends
    res = [*set(total_neighbours)]
    potential_friends = set(non_neighbors).intersection(set(res))
    print(f"5. Potential friends (2-step ∩ non-neighbors): {len(potential_friends)}")

    if len(potential_friends) == 0:
        print("   -> No potential friends! Exiting.")
        return

    # Step 6: Select using preferential attachment
    establishlinkNeighborIndex = model.select_kmax_new(potential_friends)
    establishlinkNeighbor = model.graph.nodes[establishlinkNeighborIndex]['agent']
    print(f"6. Selected neighbor: {establishlinkNeighborIndex}")
    print(f"   Selected opinion: {establishlinkNeighbor.state:.3f}")

    # Step 7: Check node state
    node_state = model.check_node_state(node, establishlinkNeighbor)
    print(f"7. Node state check: '{node_state}'")

    # Step 8: Check rewiring mode
    rewiring_mode = model.local_args["rewiringMode"]
    print(f"8. Rewiring mode: '{rewiring_mode}'")

    # Step 9: Check condition
    print(f"9. Checking condition:")
    print(f"   node_state='{node_state}', rewiring_mode='{rewiring_mode}'")
    print(f"   'same' in 'same': {'same' in 'same'}")
    print(f"   node_state in 'same': {node_state in 'same'}")
    print(f"   rewiring_mode in 'same': {rewiring_mode in 'same'}")

    condition_met = (node_state in "same" and rewiring_mode in "same")
    print(f"   -> Condition met: {condition_met}")

    if not condition_met:
        print("   -> Condition NOT met! This is why rewiring fails.")
        print(f"   -> node_state='{node_state}', rewiring_mode='{rewiring_mode}'")
        print(f"   -> For L-sim, we need node_state='same' AND rewiring_mode='same'")
        return

    # Step 10: Check if edge already exists
    edge_exists = model.graph.has_edge(test_node, establishlinkNeighborIndex)
    print(f"10. Edge already exists: {edge_exists}")

    if edge_exists:
        print("    -> Edge already exists! Cannot rewire.")
        return

    # Step 11: Random probability check
    print(f"11. Checking establishlinkprob:")
    # Check what establishlinkprob value is being used
    import models_checks as mc
    global_prob = mc.establishlinkprob
    local_prob = model.local_args.get("establishlinkprob", "NOT SET")
    print(f"    Global establishlinkprob: {global_prob}")
    print(f"    model.local_args['establishlinkprob']: {local_prob}")
    print(f"    -> rewire() uses GLOBAL variable!")

    # Manually attempt rewiring multiple times to see probability
    successes = 0
    trials = 100
    for _ in range(trials):
        if random.random() < global_prob:
            successes += 1

    print(f"    Simulated {trials} trials: {successes} successes ({100*successes/trials:.0f}%)")

    # Step 12: Actually call rewire once
    print(f"\n12. Calling model.rewire({test_node}, {establishlinkNeighborIndex})...")
    rewired = model.rewire(test_node, establishlinkNeighborIndex)
    print(f"    -> rewired: {rewired}")

    if rewired:
        print(f"    -> SUCCESS! Edge added.")
        # Check if link should be broken
        print(f"\n13. Checking breaklinkprob:")
        breaklinkprob = model.local_args.get("breaklinkprob", "NOT SET")
        print(f"    model.local_args['breaklinkprob']: {breaklinkprob}")

        if random.random() < breaklinkprob:
            current_neighbours = list(model.graph.adj[test_node].keys())
            old_neighbours = [n for n in current_neighbours if n != establishlinkNeighborIndex]
            print(f"    -> Will break link from {len(old_neighbours)} old neighbors")
        else:
            print(f"    -> Will NOT break link (probability check failed)")
    else:
        print(f"    -> FAILED! Edge not added (probability check failed).")

    print("\n" + "="*60)
    print("TRACE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    trace_single_rewiring()
