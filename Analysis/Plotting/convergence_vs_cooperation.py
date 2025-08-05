#!/usr/bin/env python3
"""
Focused Pareto analysis: Convergence speed vs final cooperativity
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from matplotlib.lines import Line2D
from datetime import date

cm = 1/2.54
FONT_SIZE = 8

FRIENDLY_COLORS = {
    'static': '#EE7733', 'random': '#0077BB', 'L-sim': '#33BBEE',
    'L-opp': '#009988', 'B-sim': '#CC3311', 'B-opp': '#EE3377',
    'wtf': '#BBBBBB', 'node2vec': '#44BB99'
}

FRIENDLY_NAMES = {
    'none_none': 'static', 'random_none': 'random', 'biased_same': 'L-sim',
    'biased_diff': 'L-opp', 'bridge_same': 'B-sim', 'bridge_diff': 'B-opp',
    'wtf_none': 'wtf', 'node2vec_none': 'node2vec'
}

def setup_style():
    plt.rcParams.update({
        'font.size': FONT_SIZE, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.figsize': (8.7*cm, 8*cm), 'axes.linewidth': 0.8,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.labelsize': FONT_SIZE-1, 'ytick.labelsize': FONT_SIZE-1,
        'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE
    })

def calculate_t95_convergence_speed(trajectory):
    """Calculate time to reach 95% of final value - robust convergence metric"""
    final_val = np.mean(trajectory[-int(len(trajectory)*0.1):])  # Last 10% average
    target = 0.95 * final_val
    
    # Find first time we reach 95% of final value
    t95_idx = np.argmax(trajectory >= target) if np.any(trajectory >= target) else len(trajectory)
    
    # Convert to speed (1 - normalized time), so higher = faster
    speed = 1 - (t95_idx / len(trajectory))
    return max(0, speed)  # Ensure non-negative

def calculate_metrics(data):
    """Calculate speed and cooperativity metrics"""
    data['rewiring'] = data['rewiring'].fillna('none')
    data['scenario'] = data['scenario'].fillna('none')
    data['scenario_grouped'] = data['scenario'].str.cat(data['rewiring'], sep='_')
    
    results = []
    for name, group in data.groupby(['scenario_grouped', 'type']):
        scenario, topology = name
        trajectory = group['avg_state'].values
        
        # Convergence speed (higher = faster)
        speed = calculate_t95_convergence_speed(trajectory)
        
        # Final cooperativity (robust measure)
        final_window = int(len(trajectory) * 0.1)  # Last 10%
        final_coop = np.mean(trajectory[-final_window:]) if final_window > 0 else trajectory[-1]
        
        friendly_name = FRIENDLY_NAMES.get(scenario, scenario)
        results.append({
            'scenario': friendly_name, 'topology': topology,
            'speed': speed, 'cooperativity': final_coop
        })
    
    return pd.DataFrame(results)

def find_pareto_front(metrics_df):
    """Find Pareto optimal algorithms"""
    pareto_mask = np.zeros(len(metrics_df), dtype=bool)
    
    for i, row in metrics_df.iterrows():
        is_dominated = False
        for j, other in metrics_df.iterrows():
            if (i != j and 
                other['speed'] >= row['speed'] and 
                other['cooperativity'] >= row['cooperativity'] and
                (other['speed'] > row['speed'] or other['cooperativity'] > row['cooperativity'])):
                is_dominated = True
                break
        pareto_mask[i] = not is_dominated
    
    return pareto_mask

def plot_pareto_analysis(metrics_df, output_path):
    """Create focused speed vs cooperativity plot with Pareto analysis"""
    setup_style()
    fig, ax = plt.subplots(figsize=(8.7*cm, 8*cm))
    
    # Filter valid data
    valid_mask = (np.isfinite(metrics_df['speed']) & 
                  np.isfinite(metrics_df['cooperativity']) &
                  (metrics_df['speed'] >= 0))
    valid_data = metrics_df[valid_mask].copy()
    
    if len(valid_data) == 0:
        print("No valid data for plotting")
        return
    
    # Convert to rank percentiles (0-1 scale)
    speed_ranks = valid_data['speed'].rank(pct=True)
    coop_ranks = valid_data['cooperativity'].rank(pct=True)
    
    # Find Pareto front using raw values
    pareto_mask = find_pareto_front(valid_data)
    pareto_data = valid_data[pareto_mask]
    dominated_data = valid_data[~pareto_mask]
    
    # Topology markers
    topology_markers = {'DPAH': 'x', 'cl': '+', 'Twitter': '*', 'FB': '.'}
    
    # Plot dominated points
    for i, (idx, row) in enumerate(dominated_data.iterrows()):
        color = FRIENDLY_COLORS.get(row['scenario'], 'black')
        marker = topology_markers.get(row['topology'], 'o')
        size = 40 if marker == '.' else 30
        
        ax.scatter(speed_ranks.loc[idx], coop_ranks.loc[idx], c=color, marker=marker,
                  s=size, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Plot Pareto optimal points (larger, bolder)
    pareto_points_ranked = []
    for i, (idx, row) in enumerate(pareto_data.iterrows()):
        color = FRIENDLY_COLORS.get(row['scenario'], 'black')
        marker = topology_markers.get(row['topology'], 'o')
        size = 60 if marker == '.' else 50
        
        x_rank, y_rank = speed_ranks.loc[idx], coop_ranks.loc[idx]
        ax.scatter(x_rank, y_rank, c=color, marker=marker,
                  s=size, alpha=0.9, edgecolors='black', linewidth=1.0)
        pareto_points_ranked.append([x_rank, y_rank])
    
    # Draw Pareto front line
    if len(pareto_points_ranked) > 1:
        pareto_array = np.array(pareto_points_ranked)
        sorted_indices = np.argsort(pareto_array[:, 0])
        sorted_pareto = pareto_array[sorted_indices]
        
        ax.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 
               'r--', alpha=0.7, linewidth=1.5)
    
    # Add y=x reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=0.8, zorder=1)
    
    # Styling
    ax.set_xlabel('Convergence Rate')
    ax.set_ylabel(r'Final Cooperation, $\langle a \rangle_{t_{end}}$')
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    
    # Adjust subplot to make room for legends
    plt.subplots_adjust(top=0.81, bottom=0.24)
    
    
    # Algorithm legend at bottom (horizontal) first
    algo_elements = [Line2D([0], [0], marker='s', color=color, linestyle='None',
                           markersize=4, label=algo)
                    for algo, color in FRIENDLY_COLORS.items() 
                    if algo in valid_data['scenario'].values]
    algo_legend = ax.legend(handles=algo_elements, bbox_to_anchor=(0.5, -0.25), 
                           loc='center', columnspacing=0.8, frameon=True, fontsize=FONT_SIZE-2, 
                           ncol=4)
    
    # Topology legend at top using figlegend (more reliable)
    topo_elements = [Line2D([0], [0], marker=marker, color='black', linestyle='None', 
                           markersize=5, label=topo) 
                    for topo, marker in topology_markers.items()]
    fig.legend(handles=topo_elements, columnspacing=0.8, bbox_to_anchor=(0.53, 0.995), 
              loc='center', frameon=True, fontsize=FONT_SIZE-2, 
              ncol=4)
    
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return pareto_data, dominated_data

def analyze_pareto_results(pareto_data, dominated_data):
    """Detailed analysis of Pareto results"""
    print("\n" + "="*50)
    print("PARETO EFFICIENCY ANALYSIS")
    print("="*50)
    
    total_algorithms = len(pareto_data) + len(dominated_data)
    print(f"Total algorithms analyzed: {total_algorithms}")
    print(f"Pareto optimal: {len(pareto_data)} ({len(pareto_data)/total_algorithms*100:.1f}%)")
    print(f"Dominated: {len(dominated_data)} ({len(dominated_data)/total_algorithms*100:.1f}%)")
    
    print(f"\n🏆 PARETO OPTIMAL ALGORITHMS:")
    print("-" * 40)
    for _, row in pareto_data.iterrows():
        print(f"  {row['scenario']} ({row['topology']}): "
              f"Speed={row['speed']:.3f}, Coop={row['cooperativity']:.3f}")
    
    if len(dominated_data) > 0:
        print(f"\n📉 DOMINATED ALGORITHMS (examples):")
        print("-" * 40)
        worst_dominated = dominated_data.nsmallest(3, ['speed', 'cooperativity'])
        for _, row in worst_dominated.iterrows():
            print(f"  {row['scenario']} ({row['topology']}): "
                  f"Speed={row['speed']:.3f}, Coop={row['cooperativity']:.3f}")
    
    # Trade-off analysis
    if len(pareto_data) > 1:
        speed_range = pareto_data['speed'].max() - pareto_data['speed'].min()
        coop_range = pareto_data['cooperativity'].max() - pareto_data['cooperativity'].min()
        
        print(f"\n📊 TRADE-OFF ANALYSIS:")
        print("-" * 40)
        print(f"Speed range on Pareto front: {speed_range:.3f}")
        print(f"Cooperativity range on Pareto front: {coop_range:.3f}")
        
        # Find extreme points
        fastest = pareto_data.loc[pareto_data['speed'].idxmax()]
        best_coop = pareto_data.loc[pareto_data['cooperativity'].idxmax()]
        
        print(f"Fastest algorithm: {fastest['scenario']} ({fastest['topology']})")
        print(f"Best cooperativity: {best_coop['scenario']} ({best_coop['topology']})")
        
        if fastest.name != best_coop.name:
            print("⚖️  Clear speed-accuracy trade-off detected!")
        else:
            print("🎯 One algorithm dominates both objectives!")

def main():
    files = [f for f in os.listdir("../../Output") if "default_run_avg" in f and f.endswith(".csv")]
    if not files:
        print("No default_run_avg files found")
        return
    
    for i, f in enumerate(files):
        print(f"{i}: {f}")
    
    idx = int(input("Select file: "))
    data = pd.read_csv(os.path.join("../../Output", files[idx]))
    
    metrics = calculate_metrics(data)
    if metrics.empty:
        print("No valid metrics calculated")
        return
    
    # Create plot and analysis
    output_dir = "../../Figs/Convergence"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/pareto_speed_cooperativity_{date.today()}.pdf"
    
    pareto_data, dominated_data = plot_pareto_analysis(metrics, output_path)
    analyze_pareto_results(pareto_data, dominated_data)
    
    print(f"\nPlot saved: {output_path}")
    return metrics, pareto_data

if __name__ == "__main__":
    main()