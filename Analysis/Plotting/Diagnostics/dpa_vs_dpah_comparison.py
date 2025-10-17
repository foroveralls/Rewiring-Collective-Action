"""
DPA vs DPAH Comparison
Compare degree distributions and visualizations between DPA and DPAH networks
Using netin's built-in visualization alongside NetworkX
"""
import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import date
from netin import DPA, DPAH, viz

cm = 1/2.54
FONT_SIZE = 8

def create_networks(n=70, seed=42):
    """Create both DPA and DPAH networks with same parameters"""

    # DPA network (no homophily)
    # To match DPAH degree ~8 with n=70: edge density d = degree / (n-1) ≈ 8/69 ≈ 0.116
    g_dpa = DPA(
        n=n,
        d=8/(n-1),  # Edge density to get ~8 average degree
        f_m=0.5,    # 50% minorities (matches typical DPAH setup)
        plo_M=2.5,  # Power law exponent for majority
        plo_m=2.5,  # Power law exponent for minority
        seed=seed
    )
    g_dpa.generate()  # IMPORTANT: Must call generate() to populate the network

    # DPAH network (with homophily)
    g_dpah = DPAH(
        n=n,
        d=8/(n-1),  # Edge density to get ~8 average degree
        f_m=0.5,    # 50% minorities
        plo_M=2.5,  # Power law exponent for majority
        plo_m=2.5,  # Power law exponent for minority
        h_MM=0.5,   # Homophily within majority
        h_mm=0.5,   # Homophily within minority
        seed=seed
    )
    g_dpah.generate()  # IMPORTANT: Must call generate() to populate the network

    return g_dpa, g_dpah

def analyze_degree_distribution(G, name="Network"):
    """Analyze degree structure"""
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Count zero-degree nodes
    zero_in = [n for n in G.nodes() if in_degrees[n] == 0]
    zero_out = [n for n in G.nodes() if out_degrees[n] == 0]

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"\nDegree Statistics:")
    print(f"  In-degree:  mean={np.mean(list(in_degrees.values())):.2f}, "
          f"std={np.std(list(in_degrees.values())):.2f}, "
          f"max={max(in_degrees.values())}, min={min(in_degrees.values())}")
    print(f"  Out-degree: mean={np.mean(list(out_degrees.values())):.2f}, "
          f"std={np.std(list(out_degrees.values())):.2f}, "
          f"max={max(out_degrees.values())}, min={min(out_degrees.values())}")

    print(f"\nZero Degree Nodes:")
    print(f"  Zero in-degree:  {len(zero_in)} nodes ({100*len(zero_in)/G.number_of_nodes():.1f}%)")
    print(f"  Zero out-degree: {len(zero_out)} nodes ({100*len(zero_out)/G.number_of_nodes():.1f}%)")

    return {
        'in_degrees': in_degrees,
        'out_degrees': out_degrees,
        'zero_in_count': len(zero_in),
        'zero_out_count': len(zero_out),
    }

def plot_netin_comparison(g_dpa, g_dpah):
    """Use netin's built-in visualization to compare networks"""
    print("\nCreating netin visualization...")

    today = date.today().strftime("%Y%m%d")
    os.makedirs("../../Figs/Diagnostics", exist_ok=True)
    filename = f"../../Figs/Diagnostics/dpa_dpah_netin_viz_{today}.png"

    # Plot both networks using netin's viz
    graphs = [g_dpa, g_dpah]
    viz.plot_graph(
        graphs,
        cell_size=2,
        ignore_singletons=True,
        share_pos=False,
        fn=filename
    )

    print(f"✓ Saved: {filename}")

def plot_networkx_comparison(dpa_data, dpah_data, dpa_graph, dpah_graph):
    """Create NetworkX comparison visualization"""
    print("\nCreating NetworkX visualization...")

    fig = plt.figure(figsize=(18*cm, 12*cm))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Plot 1: In-degree distributions
    ax1 = fig.add_subplot(gs[0, 0])
    in_dpa = list(dpa_data['in_degrees'].values())
    in_dpah = list(dpah_data['in_degrees'].values())

    bins = np.arange(0, max(max(in_dpa), max(in_dpah)) + 2) - 0.5
    ax1.hist(in_dpa, bins=bins, alpha=0.6, label='DPA', color='C0', edgecolor='black')
    ax1.hist(in_dpah, bins=bins, alpha=0.6, label='DPAH', color='C1', edgecolor='black')
    ax1.set_xlabel('In-degree (followers)', fontsize=FONT_SIZE)
    ax1.set_ylabel('Count', fontsize=FONT_SIZE)
    ax1.set_title('In-degree Distribution', fontsize=FONT_SIZE+1, fontweight='bold')
    ax1.legend(fontsize=FONT_SIZE)
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=FONT_SIZE-1)

    # Plot 2: Out-degree distributions
    ax2 = fig.add_subplot(gs[0, 1])
    out_dpa = list(dpa_data['out_degrees'].values())
    out_dpah = list(dpah_data['out_degrees'].values())

    bins = np.arange(0, max(max(out_dpa), max(out_dpah)) + 2) - 0.5
    ax2.hist(out_dpa, bins=bins, alpha=0.6, label='DPA', color='C0', edgecolor='black')
    ax2.hist(out_dpah, bins=bins, alpha=0.6, label='DPAH', color='C1', edgecolor='black')
    ax2.set_xlabel('Out-degree (following)', fontsize=FONT_SIZE)
    ax2.set_ylabel('Count', fontsize=FONT_SIZE)
    ax2.set_title('Out-degree Distribution', fontsize=FONT_SIZE+1, fontweight='bold')
    ax2.legend(fontsize=FONT_SIZE)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=FONT_SIZE-1)

    # Plot 3: Zero-degree comparison
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(2)
    width = 0.35

    zero_in = [dpa_data['zero_in_count'], dpah_data['zero_in_count']]
    zero_out = [dpa_data['zero_out_count'], dpah_data['zero_out_count']]

    ax3.bar(x - width/2, zero_in, width, label='Zero in-degree', alpha=0.8, color='C2')
    ax3.bar(x + width/2, zero_out, width, label='Zero out-degree', alpha=0.8, color='C3')
    ax3.set_ylabel('Node count', fontsize=FONT_SIZE)
    ax3.set_title('Zero-Degree Nodes', fontsize=FONT_SIZE+1, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['DPA', 'DPAH'], fontsize=FONT_SIZE)
    ax3.legend(fontsize=FONT_SIZE-1)
    ax3.grid(alpha=0.3, axis='y')
    ax3.tick_params(labelsize=FONT_SIZE-1)

    # Plot 4: DPA network visualization (NetworkX)
    ax4 = fig.add_subplot(gs[1, :2])
    layout_dpa = nx.spring_layout(dpa_graph, k=0.25, iterations=50, seed=42)

    # Color by in-degree
    node_colors_dpa = [dpa_data['in_degrees'][n] for n in dpa_graph.nodes()]

    nx.draw_networkx_nodes(dpa_graph, layout_dpa, ax=ax4,
                           node_color=node_colors_dpa, cmap='viridis',
                           node_size=50, alpha=0.8, vmin=0)
    nx.draw_networkx_edges(dpa_graph, layout_dpa, ax=ax4,
                           edge_color='gray', alpha=0.2, arrows=True,
                           arrowsize=5, width=0.5, connectionstyle='arc3,rad=0.1')
    ax4.set_title('DPA Network (NetworkX spring_layout, colored by in-degree)',
                  fontsize=FONT_SIZE+1, fontweight='bold')
    ax4.axis('off')

    # Plot 5: DPAH network visualization (NetworkX)
    ax5 = fig.add_subplot(gs[1, 2])
    layout_dpah = nx.spring_layout(dpah_graph, k=0.25, iterations=50, seed=42)

    # Color by in-degree
    node_colors_dpah = [dpah_data['in_degrees'][n] for n in dpah_graph.nodes()]

    nx.draw_networkx_nodes(dpah_graph, layout_dpah, ax=ax5,
                           node_color=node_colors_dpah, cmap='viridis',
                           node_size=50, alpha=0.8, vmin=0)
    nx.draw_networkx_edges(dpah_graph, layout_dpah, ax=ax5,
                           edge_color='gray', alpha=0.2, arrows=True,
                           arrowsize=5, width=0.5, connectionstyle='arc3,rad=0.1')
    ax5.set_title('DPAH Network\n(NetworkX spring_layout, colored by in-degree)',
                  fontsize=FONT_SIZE+1, fontweight='bold')
    ax5.axis('off')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=0,
                                                 vmax=max(max(node_colors_dpa), max(node_colors_dpah))))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[ax4, ax5], orientation='horizontal',
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('In-degree (followers)', fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE-1)

    plt.suptitle('DPA vs DPAH Network Comparison (NetworkX)',
                 fontsize=FONT_SIZE+2, fontweight='bold', y=0.98)

    # Save
    today = date.today().strftime("%Y%m%d")
    os.makedirs("../../Figs/Diagnostics", exist_ok=True)
    filename = f"../../Figs/Diagnostics/dpa_dpah_networkx_viz_{today}"
    plt.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}.pdf")

    plt.show()

def main():
    """Run comparison"""
    print("="*60)
    print("DPA vs DPAH Comparison")
    print("="*60)

    # Create networks
    print("\nCreating networks...")
    g_dpa, g_dpah = create_networks(n=70, seed=42)

    # Analyze
    dpa_data = analyze_degree_distribution(g_dpa, "DPA (No Homophily)")
    dpah_data = analyze_degree_distribution(g_dpah, "DPAH (With Homophily)")

    # Plot with netin's viz
    plot_netin_comparison(g_dpa, g_dpah)

    # Plot with NetworkX
    plot_networkx_comparison(dpa_data, dpah_data, g_dpa, g_dpah)

    # Summary
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    print(f"Zero in-degree nodes:")
    print(f"  DPA:  {dpa_data['zero_in_count']} ({100*dpa_data['zero_in_count']/70:.1f}%)")
    print(f"  DPAH: {dpah_data['zero_in_count']} ({100*dpah_data['zero_in_count']/70:.1f}%)")
    print(f"\nConclusion:")
    if abs(dpa_data['zero_in_count'] - dpah_data['zero_in_count']) < 5:
        print("  Both DPA and DPAH produce similar asymmetric structures.")
        print("  The issue is with preferential attachment initialization,")
        print("  not the homophily component.")
    else:
        print("  DPA and DPAH differ significantly in zero in-degree nodes.")
        print("  Homophily may be affecting the initialization structure.")
    print("="*60)

if __name__ == "__main__":
    main()
