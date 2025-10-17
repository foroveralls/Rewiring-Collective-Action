"""
Diagnostic script to verify L-sim parameters and opinion states
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

def verify_lsim_setup():
    """Verify L-sim parameters are correctly applied"""

    print("="*60)
    print("L-SIM PARAMETER VERIFICATION")
    print("="*60)

    # Base parameters matching transformation_grid_plot.py
    base_params = {
        "type": "DPAH",
        "nwsize": 100,
        "degree": 5,
        "polarisingNode_f": 0.10,
        "timesteps": 1000,  # Short for testing
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

    # 1. Verify parameter passing
    print("\n1. CHECKING PARAMETER SETUP")
    print(f"   Rewiring Algorithm: {base_params['rewiringAlgorithm']}")
    print(f"   Rewiring Mode: {base_params['rewiringMode']}")
    print(f"   Initial Skew: {base_params['skew']}")
    print(f"   Network Size: {base_params['nwsize']}")
    print(f"   Degree: {base_params['degree']}")

    # 2. Create a model and verify algorithm assignment
    print("\n2. CREATING MODEL AND VERIFYING ALGORITHM")
    np.random.seed(42)
    random.seed(42)

    friendshipWeightGenerator = get_truncated_normal(base_params["friendship"], base_params["friendshipSD"], 0, 1)
    initialStateGenerator = get_truncated_normal(base_params["skew"], base_params["initSD"], -1, 1)

    model = models_checks.DPAHModel(base_params["nwsize"], base_params["degree"],
                                    skew=base_params["skew"],
                                    friendshipWeightGenerator=friendshipWeightGenerator,
                                    initialStateGenerator=initialStateGenerator,
                                    args=base_params)

    print(f"   Model algorithm: {model.algo}")
    print(f"   Model local_args['rewiringAlgorithm']: {model.local_args['rewiringAlgorithm']}")
    print(f"   Model local_args['rewiringMode']: {model.local_args['rewiringMode']}")
    print(f"   Model has call_algo method: {hasattr(model, 'call_algo')}")
    print(f"   call_algo method name: {model.call_algo.__name__ if hasattr(model, 'call_algo') else 'N/A'}")

    # 3. Verify initial opinion distribution
    print("\n3. CHECKING INITIAL OPINION DISTRIBUTION")
    opinions = [model.graph.nodes[i]['agent'].state for i in range(base_params["nwsize"])]
    print(f"   Mean opinion: {np.mean(opinions):.3f} (expected ~{base_params['skew']})")
    print(f"   Std opinion: {np.std(opinions):.3f} (expected ~{base_params['initSD']})")
    print(f"   Min opinion: {np.min(opinions):.3f}")
    print(f"   Max opinion: {np.max(opinions):.3f}")
    print(f"   % Cooperators (>0): {100*np.mean(np.array(opinions) > 0):.1f}%")

    # 4. Test rewiring behavior
    print("\n4. TESTING REWIRING BEHAVIOR")

    # Get a node and check its neighbors before rewiring
    test_node = 0
    neighbors_before = set(model.graph.adj[test_node].keys())
    print(f"   Test node {test_node} neighbors before: {len(neighbors_before)}")

    # Run a few interactions to trigger rewiring
    for i in range(100):
        model.interact_main()

    neighbors_after = set(model.graph.adj[test_node].keys())
    print(f"   Test node {test_node} neighbors after 100 interactions: {len(neighbors_after)}")
    print(f"   Neighbor change: {len(neighbors_after - neighbors_before)} new, {len(neighbors_before - neighbors_after)} removed")

    # 5. Run short simulation and check convergence direction
    print("\n5. RUNNING SHORT SIMULATION")
    initial_avg = np.mean(opinions)

    model.runSim(base_params["timesteps"], clusters=True, drawModel=False)

    final_opinions = [model.graph.nodes[i]['agent'].state for i in range(base_params["nwsize"])]
    final_avg = np.mean(final_opinions)

    print(f"   Initial avg opinion: {initial_avg:.3f}")
    print(f"   Final avg opinion: {final_avg:.3f}")
    print(f"   Change: {final_avg - initial_avg:.3f}")
    print(f"   Trajectory: {model.states[0]:.3f} -> {model.states[-1]:.3f}")

    # Show trajectory over time
    print("\n6. CONVERGENCE TRAJECTORY")
    checkpoints = [0, len(model.states)//4, len(model.states)//2, 3*len(model.states)//4, len(model.states)-1]
    for t in checkpoints:
        print(f"   t={t:4d}: opinion = {model.states[t]:.3f}")

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

    return model

if __name__ == "__main__":
    model = verify_lsim_setup()
