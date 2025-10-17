"""
Debug why L-sim rewiring isn't working
"""
import sys
sys.path.append('../..')
sys.path.append('..')

import numpy as np
import random
import networkx as nx
from scipy.stats import truncnorm
import models_checks

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def debug_rewiring():
    """Debug why rewiring isn't happening"""

    print("="*60)
    print("DEBUGGING L-SIM REWIRING")
    print("="*60)

    # Base parameters
    base_params = {
        "type": "DPAH",
        "nwsize": 100,
        "degree": 5,
        "polarisingNode_f": 0.10,
        "timesteps": 100,
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

    print("\n1. NETWORK STRUCTURE")
    degrees = [model.graph.degree(i) for i in range(base_params["nwsize"])]
    print(f"   Mean degree: {np.mean(degrees):.2f}")
    print(f"   Min degree: {np.min(degrees)}")
    print(f"   Max degree: {np.max(degrees)}")
    print(f"   Degree distribution: min={np.min(degrees)}, 25%={np.percentile(degrees, 25):.0f}, "
          f"50%={np.percentile(degrees, 50):.0f}, 75%={np.percentile(degrees, 75):.0f}, max={np.max(degrees)}")

    # Check a few nodes in detail
    print("\n2. DETAILED NODE ANALYSIS")
    test_nodes = [0, 10, 20, 30, 40]

    for node_idx in test_nodes:
        print(f"\n   Node {node_idx}:")

        # Current neighbors
        neighbors = list(model.graph.adj[node_idx].keys())
        print(f"     Current neighbors: {len(neighbors)} - {neighbors[:5]}...")

        # Node opinion
        node_opinion = model.graph.nodes[node_idx]['agent'].state
        print(f"     Node opinion: {node_opinion:.3f}")

        # 2-step neighbors
        total_neighbours = model.find_2_steps(node_idx)
        print(f"     2-step neighbors: {len(total_neighbours)}")

        # Non-neighbors
        non_neighbors = [k for k in nx.non_neighbors(model.graph, node_idx)]
        print(f"     Non-neighbors: {len(non_neighbors)}")

        # Potential friends (2-step AND non-neighbors)
        res = [*set(total_neighbours)]
        potential_friends = set(non_neighbors).intersection(set(res))
        print(f"     Potential friends (2-step ∩ non-neighbors): {len(potential_friends)}")

        if len(potential_friends) > 0:
            # Check opinions of potential friends
            same_opinion = []
            diff_opinion = []
            for pf in list(potential_friends)[:10]:  # Check first 10
                pf_opinion = model.graph.nodes[pf]['agent'].state
                if (node_opinion >= 0 and pf_opinion >= 0) or (node_opinion < 0 and pf_opinion < 0):
                    same_opinion.append(pf)
                else:
                    diff_opinion.append(pf)

            print(f"       Same opinion: {len(same_opinion)} / Different opinion: {len(diff_opinion)}")

    # Test actual rewiring call
    print("\n3. TESTING ACTUAL REWIRING FUNCTION")
    test_node = 20  # Pick a node with reasonable degree

    print(f"   Testing node {test_node}")
    print(f"   Neighbors before: {len(list(model.graph.adj[test_node].keys()))}")

    # Manually call rewiring function multiple times
    rewiring_attempts = 0
    successful_rewires = 0

    for i in range(50):
        # Pick random node
        node = random.randint(0, 99)
        neighbors_before = len(list(model.graph.adj[node].keys()))

        # Call rewiring
        model.biasedrewiring(node)

        neighbors_after = len(list(model.graph.adj[node].keys()))
        rewiring_attempts += 1

        if neighbors_before != neighbors_after:
            successful_rewires += 1

    print(f"   Rewiring attempts: {rewiring_attempts}")
    print(f"   Successful rewires: {successful_rewires}")
    print(f"   Success rate: {100*successful_rewires/rewiring_attempts:.1f}%")

    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    debug_rewiring()
