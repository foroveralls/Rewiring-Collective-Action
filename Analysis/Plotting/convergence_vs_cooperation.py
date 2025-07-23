#!/usr/bin/env python3
"""
Convergence Rate vs Final Cooperativity comparison plot for network dynamics algorithms.
Compares how fast algorithms converge vs their final cooperative state.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
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
        'figure.figsize': (9*cm, 9*cm), 'axes.linewidth': 0.8,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.labelsize': FONT_SIZE-1, 'ytick.labelsize': FONT_SIZE-1,
        'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE
    })

def find_inflection(seq, sigma=300, min_idx=5000):
    if len(seq) < 1200: return False
    try:
        smooth = gaussian_filter1d(seq, sigma)
        d2 = np.gradient(np.gradient(smooth))
        infls = np.where(np.diff(np.sign(d2)))[0]
        inf_ind = next((i for i in infls if i >= min_idx), None)
        return inf_ind if inf_ind and inf_ind < len(seq) * 0.9 else False
    except: return False

def estimate_convergence_rate(trajec, loc, regwin=15):
    if not isinstance(loc, (int, np.integer)): return 0
    start_idx, end_idx = max(0, loc-regwin), min(len(trajec)-1, loc+regwin+1)
    if end_idx - start_idx < 3: return 0
    
    x, y = np.arange(start_idx, end_idx), trajec[start_idx:end_idx]
    n, mx, my = len(x), np.mean(x), np.mean(y)
    ssxy, ssxx = np.sum(y*x) - n*my*mx, np.sum(x*x) - n*mx*mx
    if ssxx == 0: return 0
    
    b1 = ssxy / ssxx
    denom = abs(trajec[loc] - 1)
    return b1 / (0.001 if denom < 0.001 else denom)

def calculate_metrics(data):
    data['rewiring'] = data['rewiring'].fillna('none')
    data['scenario'] = data['scenario'].fillna('none')
    data['scenario_grouped'] = data['scenario'].str.cat(data['rewiring'], sep='_')
    
    results = []
    for name, group in data.groupby(['scenario_grouped', 'type']):
        scenario, topology = name
        trajectory = group['avg_state'].values
        
        # Final cooperativity (average of last 20% of trajectory)
        final_window = int(len(trajectory) * 0.2)
        final_coop = np.mean(trajectory[-final_window:]) if final_window > 0 else trajectory[-1]
        
        # Convergence rate
        inflection_x = find_inflection(trajectory)
        conv_rate = estimate_convergence_rate(trajectory, inflection_x) * 1000 if inflection_x else 0
        
        friendly_name = FRIENDLY_NAMES.get(scenario, scenario)
        results.append({
            'scenario': friendly_name, 'topology': topology,
            'conv_rate': conv_rate, 'final_coop': final_coop
        })
    
    return pd.DataFrame(results)

def plot_rate_vs_coop(metrics_df, output_path):
    setup_style()
    fig, ax = plt.subplots(figsize=(9*cm, 9*cm))
    
    # Topology markers matching convergence plots
    topology_markers = {'DPAH': 'x', 'cl': '+', 'Twitter': '*', 'FB': '.'}
    
    # Use data-driven normalization with padding for better distribution
    rate_min, rate_max = metrics_df['conv_rate'].min(), metrics_df['conv_rate'].max()
    rate_range = rate_max - rate_min
    # Add 20% padding to spread points better
    rate_norm_max = rate_max + 0.2 * rate_range
    conv_rate_norm = (metrics_df['conv_rate'] - rate_min) / (rate_norm_max - rate_min)
    conv_rate_norm = np.clip(conv_rate_norm, 0, 1)
    coop_norm = (metrics_df['final_coop'] + 1) / 2  # Map [-1,1] to [0,1]
    
    # Calculate Euclidean scores using normalized values
    distances = np.sqrt((1 - conv_rate_norm)**2 + (1 - coop_norm)**2)
    scores = 1 - distances/np.sqrt(2)
    
    # Plot points with normalized coordinates
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        color = FRIENDLY_COLORS.get(row['scenario'], 'black')
        marker = topology_markers.get(row['topology'], 'o')
        size = 100 if marker == '.' else 80
        
        x_norm, y_norm = conv_rate_norm.iloc[i], coop_norm.iloc[i]
        ax.scatter(x_norm, y_norm, c=color, marker=marker, 
                  s=size, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add score text
        ax.text(x_norm + 0.02, y_norm + 0.02, f'{scores.iloc[i]:.2f}', 
               fontsize=5, ha='center', va='bottom', alpha=0.8)
    
    # Parity line on [0,1] scale
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=0.8, zorder=1)
    ax.text(0.7, 0.8, 'y = x', fontsize=FONT_SIZE-1, alpha=0.7)
    
    ax.set_xlabel('Normalized Convergence Rate', fontweight='bold')
    ax.set_ylabel('Normalized Cooperativity', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    
    # Add topology legend (symbols)
    from matplotlib.lines import Line2D
    topo_elements = [
        Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=6, label='DPAH'),
        Line2D([0], [0], marker='+', color='gray', linestyle='None', markersize=6, label='cl'),
        Line2D([0], [0], marker='*', color='gray', linestyle='None', markersize=6, label='Twitter'),
        Line2D([0], [0], marker='.', color='gray', linestyle='None', markersize=8, label='FB')
    ]
    topo_legend = ax.legend(handles=topo_elements, loc='upper left', frameon=True, 
                           fontsize=FONT_SIZE-1, handletextpad=0.3, title='Topology')
    
    # Add algorithm legend (colors)
    algo_elements = [
        Line2D([0], [0], marker='s', color=color, linestyle='None', 
               markersize=4, label=algo) 
        for algo, color in FRIENDLY_COLORS.items()
    ]
    ax.add_artist(topo_legend)  # Keep topology legend
    ax.legend(handles=algo_elements, loc='lower right', frameon=True, 
             fontsize=FONT_SIZE-1, handletextpad=0.3, title='Algorithm')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

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
    
    output_dir = "../../Figs/Convergence"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/rate_vs_coop_{date.today()}.pdf"
    
    plot_rate_vs_coop(metrics, output_path)
    print(f"Plot saved: {output_path}")
    print(f"Metrics calculated for {len(metrics)} algorithm-topology combinations")

if __name__ == "__main__":
    main()