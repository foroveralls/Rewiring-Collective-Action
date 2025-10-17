"""
Test rewiring statistics over many attempts
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

def test_rewiring_statistics():
    """Test rewiring over many attempts"""

    print("="*60)
    print("L-SIM REWIRING STATISTICS")
    print("="*60)

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

    print("\nInitial network:")
    print(f"  Nodes: {model.graph.number_of_nodes()}")
    print(f"  Edges: {model.graph.number_of_edges()}")
    print(f"  Mean degree: {2*model.graph.number_of_edges()/model.graph.number_of_nodes():.2f}")

    # Track edges before
    edges_before = set(model.graph.edges())

    # Run many interaction+rewiring cycles
    attempts = 0
    edge_adds = 0
    edge_removes = 0
    no_potential_friends = 0
    condition_not_met = 0
    edge_exists_already = 0
    prob_failed = 0

    for i in range(1000):
        # Pick random node
        node_idx = random.randint(0, 99)

        # Get initial info
        neighbors_before = set(model.graph.adj[node_idx].keys())
        degree_before = len(neighbors_before)

        # Find potential friends
        total_neighbours = model.find_2_steps(node_idx)
        non_neighbors = [k for k in nx.non_neighbors(model.graph, node_idx)]
        res = [*set(total_neighbours)]
        potential_friends = set(non_neighbors).intersection(set(res))

        if len(potential_friends) == 0:
            no_potential_friends += 1
            continue

        attempts += 1

        # Select neighbor
        establishlinkNeighborIndex = model.select_kmax_new(potential_friends)
        node = model.graph.nodes[node_idx]['agent']
        establishlinkNeighbor = model.graph.nodes[establishlinkNeighborIndex]['agent']

        # Check state
        node_state = model.check_node_state(node, establishlinkNeighbor)
        rewiring_mode = model.local_args["rewiringMode"]

        if not (node_state in "same" and rewiring_mode in "same"):
            condition_not_met += 1
            continue

        # Check if edge exists
        if model.graph.has_edge(node_idx, establishlinkNeighborIndex):
            edge_exists_already += 1
            continue

        # Try to rewire
        rewired = model.rewire(node_idx, establishlinkNeighborIndex)

        if rewired:
            edge_adds += 1
            # Try to break link
            if random.random() < model.local_args["breaklinkprob"]:
                current_neighbours = list(model.graph.adj[node_idx].keys())
                old_neighbours = [n for n in current_neighbours if n != establishlinkNeighborIndex]
                if len(old_neighbours) >= 1:
                    breaklinkNeighbourIndex = old_neighbours[random.randint(0, len(old_neighbours)-1)]
                    model.graph.remove_edge(node_idx, breaklinkNeighbourIndex)
                    edge_removes += 1
        else:
            prob_failed += 1

    edges_after = set(model.graph.edges())
    new_edges = edges_after - edges_before
    removed_edges = edges_before - edges_after

    print(f"\nAfter 1000 interaction attempts:")
    print(f"  Valid rewiring attempts: {attempts}")
    print(f"    - No potential friends: {no_potential_friends}")
    print(f"    - Condition not met (different opinion): {condition_not_met}")
    print(f"    - Edge already exists: {edge_exists_already}")
    print(f"    - Probability check failed: {prob_failed}")
    print(f"  Successful edge additions: {edge_adds}")
    print(f"  Edge removals: {edge_removes}")
    print(f"  Net edges changed: {len(new_edges)} new, {len(removed_edges)} removed")
    print(f"  Final edges: {model.graph.number_of_edges()}")
    print(f"  Final mean degree: {2*model.graph.number_of_edges()/model.graph.number_of_nodes():.2f}")

    # Calculate expected success rate
    if attempts > 0:
        expected_additions = attempts * 0.5  # establishlinkprob = 0.5
        print(f"\nExpected edge additions (at 50% prob): {expected_additions:.0f}")
        print(f"Actual edge additions: {edge_adds}")
        print(f"Success rate: {100*edge_adds/attempts:.1f}% (expected ~50%)")

    print("\n" + "="*60)
    print("STATISTICS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_rewiring_statistics()
