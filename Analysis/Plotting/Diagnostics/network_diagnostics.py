"""
Network Structure Diagnostics
Analyze degree distributions and layout centrality for directed networks
"""
import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import date
import random

sys.path.append('../../')
sys.path.append('..')
import models_checks

cm = 1/2.54
FONT_SIZE = 8

def run_single_simulation(algo, mode, timesteps=15000):
    """Run a single simulation and return the model"""
    params = {
        "type": "DPAH",
        "nwsize": 70,
        "degree": 8,
        "polarisingNode_f": 0.10,
        "timesteps": timesteps,
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
        "save_snapshots": True,
        "rewiringAlgorithm": algo,
        "rewiringMode": mode,
    }

    run_id = 0
    np.random.seed(42)
    random.seed(42)
    result = models_checks.simulate(run_id, params)

    if isinstance(result, tuple):
        model, snapshots = result
        model.snapshots = snapshots
    else:
        model = result
        model.snapshots = {}

    return model

def analyze_network_structure(G, name="Network"):
    """Analyze degree structure and centrality of a directed network"""
    print(f"\n{'='*60}")
    print(f"Analysis: {name}")
    print(f"{'='*60}")

    if not nx.is_directed(G):
        print("Warning: Graph is not directed!")
        return

    # Basic stats
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Degree distributions
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    total_degrees = {n: in_degrees[n] + out_degrees[n] for n in G.nodes()}

    print(f"\nDegree Statistics:")
    print(f"  In-degree:  mean={np.mean(list(in_degrees.values())):.2f}, "
          f"std={np.std(list(in_degrees.values())):.2f}, "
          f"max={max(in_degrees.values())}")
    print(f"  Out-degree: mean={np.mean(list(out_degrees.values())):.2f}, "
          f"std={np.std(list(out_degrees.values())):.2f}, "
          f"max={max(out_degrees.values())}")
    print(f"  Total:      mean={np.mean(list(total_degrees.values())):.2f}, "
          f"std={np.std(list(total_degrees.values())):.2f}, "
          f"max={max(total_degrees.values())}")

    # Find hub nodes (top 5 by different metrics)
    top_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    top_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    top_total = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\nTop 5 nodes by IN-degree (followers/influence):")
    for node, deg in top_in:
        print(f"  Node {node}: in={in_degrees[node]}, out={out_degrees[node]}, total={total_degrees[node]}")

    print(f"\nTop 5 nodes by OUT-degree (following):")
    for node, deg in top_out:
        print(f"  Node {node}: in={in_degrees[node]}, out={out_degrees[node]}, total={total_degrees[node]}")

    print(f"\nTop 5 nodes by TOTAL degree:")
    for node, deg in top_total:
        print(f"  Node {node}: in={in_degrees[node]}, out={out_degrees[node]}, total={total_degrees[node]}")

    # Compute layout and analyze positioning
    print(f"\nLayout Analysis (spring_layout):")
    layout = nx.spring_layout(G, k=0.25, iterations=50, seed=42)

    # Find distance from center for each node
    center = np.array([0, 0])
    distances = {n: np.linalg.norm(np.array(pos) - center) for n, pos in layout.items()}

    # Find most central nodes (closest to center)
    central_nodes = sorted(distances.items(), key=lambda x: x[1])[:5]

    print(f"  Top 5 CENTRAL nodes in layout (closest to center):")
    for node, dist in central_nodes:
        print(f"    Node {node}: distance={dist:.3f}, in={in_degrees[node]}, "
              f"out={out_degrees[node]}, total={total_degrees[node]}")

    # Compute correlation between degree and centrality
    from scipy.stats import spearmanr
    nodes_list = list(G.nodes())
    dist_list = [distances[n] for n in nodes_list]
    in_deg_list = [in_degrees[n] for n in nodes_list]
    out_deg_list = [out_degrees[n] for n in nodes_list]
    total_deg_list = [total_degrees[n] for n in nodes_list]

    # Negative correlation means high degree = close to center
    corr_in = spearmanr(-np.array(dist_list), in_deg_list)[0]
    corr_out = spearmanr(-np.array(dist_list), out_deg_list)[0]
    corr_total = spearmanr(-np.array(dist_list), total_deg_list)[0]

    print(f"\n  Correlation (centrality vs degree):")
    print(f"    In-degree correlation:    {corr_in:.3f}")
    print(f"    Out-degree correlation:   {corr_out:.3f}")
    print(f"    Total degree correlation: {corr_total:.3f}")
    print(f"  (1.0 = high degree → central, -1.0 = high degree → peripheral)")

    # Analyze edge patterns relative to center
    print(f"\nEdge Direction Analysis:")
    edges_out_from_center = 0  # Edges pointing away from center
    edges_in_to_center = 0      # Edges pointing toward center

    # Define "central" as nodes in top quartile of proximity to center
    dist_threshold = np.percentile(list(distances.values()), 25)
    central_node_set = {n for n, d in distances.items() if d <= dist_threshold}

    for u, v in G.edges():
        u_central = u in central_node_set
        v_central = v in central_node_set

        if u_central and not v_central:
            edges_out_from_center += 1
        elif v_central and not u_central:
            edges_in_to_center += 1

    total_central_edges = edges_out_from_center + edges_in_to_center
    print(f"  Central nodes: {len(central_node_set)} (within {dist_threshold:.3f} of center)")
    print(f"  Edges FROM central → peripheral: {edges_out_from_center} "
          f"({100*edges_out_from_center/total_central_edges if total_central_edges > 0 else 0:.1f}%)")
    print(f"  Edges FROM peripheral → central: {edges_in_to_center} "
          f"({100*edges_in_to_center/total_central_edges if total_central_edges > 0 else 0:.1f}%)")

    return {
        'in_degrees': in_degrees,
        'out_degrees': out_degrees,
        'total_degrees': total_degrees,
        'layout': layout,
        'distances': distances,
        'correlations': {'in': corr_in, 'out': corr_out, 'total': corr_total}
    }

def plot_degree_comparison(results_dict):
    """Create visualization comparing degree patterns across algorithms"""
    fig, axes = plt.subplots(2, 2, figsize=(12*cm, 12*cm))
    fig.suptitle('Degree Structure Comparison', fontsize=FONT_SIZE+1)

    algorithms = list(results_dict.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))

    # Plot 1: In-degree distributions
    ax = axes[0, 0]
    for i, (alg, data) in enumerate(results_dict.items()):
        in_degs = list(data['in_degrees'].values())
        ax.hist(in_degs, bins=20, alpha=0.5, label=alg, color=colors[i])
    ax.set_xlabel('In-degree (followers)', fontsize=FONT_SIZE)
    ax.set_ylabel('Count', fontsize=FONT_SIZE)
    ax.set_title('In-degree Distribution', fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE-1)
    ax.grid(alpha=0.3)

    # Plot 2: Out-degree distributions
    ax = axes[0, 1]
    for i, (alg, data) in enumerate(results_dict.items()):
        out_degs = list(data['out_degrees'].values())
        ax.hist(out_degs, bins=20, alpha=0.5, label=alg, color=colors[i])
    ax.set_xlabel('Out-degree (following)', fontsize=FONT_SIZE)
    ax.set_ylabel('Count', fontsize=FONT_SIZE)
    ax.set_title('Out-degree Distribution', fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE-1)
    ax.grid(alpha=0.3)

    # Plot 3: In vs Out degree scatter
    ax = axes[1, 0]
    for i, (alg, data) in enumerate(results_dict.items()):
        in_degs = list(data['in_degrees'].values())
        out_degs = list(data['out_degrees'].values())
        ax.scatter(in_degs, out_degs, alpha=0.5, s=10, label=alg, color=colors[i])
    ax.set_xlabel('In-degree', fontsize=FONT_SIZE)
    ax.set_ylabel('Out-degree', fontsize=FONT_SIZE)
    ax.set_title('In vs Out Degree', fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE-1)
    ax.grid(alpha=0.3)
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', alpha=0.3, linewidth=0.5)

    # Plot 4: Correlation bar chart
    ax = axes[1, 1]
    x = np.arange(len(algorithms))
    width = 0.25

    in_corrs = [data['correlations']['in'] for data in results_dict.values()]
    out_corrs = [data['correlations']['out'] for data in results_dict.values()]
    total_corrs = [data['correlations']['total'] for data in results_dict.values()]

    ax.bar(x - width, in_corrs, width, label='In-degree', alpha=0.8)
    ax.bar(x, out_corrs, width, label='Out-degree', alpha=0.8)
    ax.bar(x + width, total_corrs, width, label='Total degree', alpha=0.8)

    ax.set_xlabel('Algorithm', fontsize=FONT_SIZE)
    ax.set_ylabel('Correlation with centrality', fontsize=FONT_SIZE)
    ax.set_title('Degree-Centrality Correlation', fontsize=FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=FONT_SIZE-1, rotation=45, ha='right')
    ax.legend(fontsize=FONT_SIZE-1)
    ax.grid(alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()

    # Save
    today = date.today().strftime("%Y%m%d")
    os.makedirs("../../Figs/Diagnostics", exist_ok=True)
    filename = f"../../Figs/Diagnostics/degree_analysis_{today}"
    plt.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}.pdf")

    plt.show()

def main():
    """Run diagnostics on multiple algorithms"""
    print("="*60)
    print("Network Structure Diagnostics")
    print("="*60)

    algorithms = [
        {'name': 'Static', 'algo': 'None', 'mode': 'None'},
        {'name': 'WTF', 'algo': 'wtf', 'mode': 'None'},
        {'name': 'Bridge(opp)', 'algo': 'bridge', 'mode': 'diff'},
        {'name': 'Local(sim)', 'algo': 'biased', 'mode': 'same'},
    ]

    results = {}

    for alg_config in algorithms:
        print(f"\nRunning {alg_config['name']}...")
        model = run_single_simulation(alg_config['algo'], alg_config['mode'], timesteps=15000)

        # Use final graph
        G_final = model.graph

        # Analyze
        analysis = analyze_network_structure(G_final, name=alg_config['name'])
        results[alg_config['name']] = analysis

    # Create comparison plots
    print("\n" + "="*60)
    print("Creating comparison visualizations...")
    plot_degree_comparison(results)

    print("\n" + "="*60)
    print("Diagnostic analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()
