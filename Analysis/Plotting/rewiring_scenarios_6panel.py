"""
Generate 6-panel figure showing different rewiring scenarios for publication.

Panels:
A - local(similar): Connect to nearby similar node
B - local(opposite): Connect to nearby opposite node
C - bridge(opposite): Bridge to opposite node in different cluster
D - bridge(similar): Bridge to similar node in different cluster
E - WTF: Who-To-Follow - 2-hop connection through "circle of trust"
F - Node2Vec: Structural equivalence - connect structurally similar nodes
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Define colors matching the provided image
COLOR_BLUE = '#5B7FB4'
COLOR_ORANGE = '#D9965F'
COLOR_HIGHLIGHT = '#7FFF7F'  # Green highlight for focal nodes
COLOR_EDGE = '#333333'
COLOR_PROPOSED = '#00CC00'  # Green for proposed edges

def create_base_network():
    """Create a base network structure used across scenarios."""
    G = nx.Graph()

    # Define node positions manually for consistent layout
    pos = {
        # Left cluster (mostly blue)
        0: (0.1, 0.8),
        1: (0.2, 0.95),
        2: (0.35, 0.9),
        3: (0.15, 0.65),
        4: (0.05, 0.5),
        # Right cluster (mostly orange)
        5: (0.65, 0.95),
        6: (0.8, 0.9),
        7: (0.9, 0.75),
        8: (0.85, 0.6),
        9: (0.95, 0.5),
        # Bridge nodes
        10: (0.45, 0.65),
        11: (0.55, 0.4),
    }

    # Add nodes
    for node in pos:
        G.add_node(node)

    # Add edges to create two loosely connected clusters
    edges = [
        # Left cluster
        (0, 1), (1, 2), (0, 3), (3, 4),
        # Right cluster
        (5, 6), (6, 7), (7, 8), (8, 9),
        # Bridge connections
        (2, 10), (10, 11), (11, 8),
        (4, 3), (0, 2),
        (5, 7), (6, 8),
    ]
    G.add_edges_from(edges)

    return G, pos

def assign_colors_similar(G):
    """Assign node colors for similar scenarios (left=blue, right=orange)."""
    colors = {}
    # Left cluster = blue
    for node in [0, 1, 2, 3, 4]:
        colors[node] = 'blue'
    # Right cluster = orange
    for node in [5, 6, 7, 8, 9]:
        colors[node] = 'orange'
    # Bridge nodes
    colors[10] = 'orange'
    colors[11] = 'blue'
    return colors

def assign_colors_mixed(G):
    """Assign node colors for mixed scenarios."""
    colors = {}
    # Mix colors more
    colors.update({0: 'blue', 1: 'blue', 2: 'orange', 3: 'blue', 4: 'blue',
                   5: 'orange', 6: 'blue', 7: 'orange', 8: 'orange', 9: 'orange',
                   10: 'blue', 11: 'orange'})
    return colors

def draw_scenario(ax, G, pos, node_colors, focal_node, target_node,
                  title, panel_label, trust_circle=None, show_path=False):
    """
    Draw a single rewiring scenario.

    Parameters:
    -----------
    ax : matplotlib axis
    G : networkx graph
    pos : dict of node positions
    node_colors : dict of node colors ('blue' or 'orange')
    focal_node : int, node initiating rewiring
    target_node : int, proposed connection target
    title : str, panel title
    panel_label : str, panel letter (A, B, C, etc.)
    trust_circle : list, nodes to highlight as "circle of trust" (for WTF)
    show_path : bool, show path between focal and target (for WTF)
    """

    # Convert colors to hex
    node_color_map = {'blue': COLOR_BLUE, 'orange': COLOR_ORANGE}
    colors = [node_color_map[node_colors[n]] for n in G.nodes()]

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=COLOR_EDGE,
                          width=1.5, alpha=0.6)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                          node_size=400, linewidths=2,
                          edgecolors='white')

    # Highlight focal node with green circle
    focal_pos = pos[focal_node]
    circle = plt.Circle(focal_pos, 0.04, color=COLOR_HIGHLIGHT,
                       fill=False, linewidth=3, zorder=10)
    ax.add_patch(circle)

    # Highlight trust circle if specified (for WTF)
    if trust_circle:
        for node in trust_circle:
            trust_pos = pos[node]
            circle = plt.Circle(trust_pos, 0.03, color=COLOR_HIGHLIGHT,
                              fill=False, linewidth=2, zorder=10, linestyle='--')
            ax.add_patch(circle)

    # Draw proposed edge as dotted green line
    if target_node is not None and target_node != focal_node:
        x = [pos[focal_node][0], pos[target_node][0]]
        y = [pos[focal_node][1], pos[target_node][1]]
        ax.plot(x, y, COLOR_PROPOSED, linewidth=2.5,
               linestyle='--', zorder=5, alpha=0.8)

        # Highlight target node
        target_pos = pos[target_node]
        circle = plt.Circle(target_pos, 0.04, color=COLOR_HIGHLIGHT,
                           fill=False, linewidth=3, zorder=10)
        ax.add_patch(circle)

    # Add title and panel label
    ax.text(0.02, 0.98, f'{panel_label}', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='top')
    ax.text(0.5, 0.98, title, transform=ax.transAxes,
           fontsize=13, va='top', ha='center', fontweight='bold')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')

def create_local_similar():
    """Panel A: local(similar) - connect to nearby node of same color."""
    G, pos = create_base_network()
    colors = assign_colors_similar(G)

    # Focal: blue node on left, target: nearby blue node
    focal = 3
    target = 1

    return G, pos, colors, focal, target, None

def create_local_opposite():
    """Panel B: local(opposite) - connect to nearby node of opposite color."""
    G, pos = create_base_network()
    colors = assign_colors_similar(G)
    colors[2] = 'orange'  # Make node 2 orange for contrast
    colors[10] = 'orange'

    # Focal: blue node, target: nearby orange node
    focal = 3
    target = 10

    return G, pos, colors, focal, target, None

def create_bridge_opposite():
    """Panel C: bridge(opposite) - bridge to opposite color in different cluster."""
    G, pos = create_base_network()
    colors = assign_colors_similar(G)

    # Focal: left cluster, target: right cluster opposite color
    focal = 3
    target = 8

    return G, pos, colors, focal, target, None

def create_bridge_similar():
    """Panel D: bridge(similar) - bridge to similar color in different cluster."""
    G, pos = create_base_network()
    colors = assign_colors_similar(G)
    colors[8] = 'blue'  # Make target node same color

    # Focal: left blue node, target: right blue node
    focal = 1
    target = 8

    return G, pos, colors, focal, target, None

def create_wtf():
    """Panel E: WTF - connect to 2-hop neighbor through circle of trust."""
    G, pos = create_base_network()
    colors = assign_colors_mixed(G)

    # Focal node with clear neighbors
    focal = 4
    # Circle of trust: immediate neighbors
    trust_circle = [3]
    # Target: 2-hop away (neighbor of neighbor)
    target = 0

    return G, pos, colors, focal, target, trust_circle

def create_node2vec():
    """Panel F: Node2Vec - connect structurally similar nodes (both hubs)."""
    G, pos = create_base_network()
    colors = assign_colors_mixed(G)

    # Connect two hub nodes (high degree) that are distant
    # Add extra edges to make clear hubs
    G.add_edges_from([(1, 3), (7, 9)])

    focal = 1  # Hub in left cluster
    target = 7  # Hub in right cluster (similar degree)

    return G, pos, colors, focal, target, None

def create_figure():
    """Create the complete 6-panel figure."""

    fig = plt.figure(figsize=(15, 10))

    # Create 2x3 grid
    scenarios = [
        ('A', 'local(similar)', create_local_similar),
        ('B', 'local(opposite)', create_local_opposite),
        ('C', 'bridge(opposite)', create_bridge_opposite),
        ('D', 'bridge(similar)', create_bridge_similar),
        ('E', 'WTF', create_wtf),
        ('F', 'Node2Vec', create_node2vec),
    ]

    for idx, (panel, title, create_func) in enumerate(scenarios):
        ax = plt.subplot(2, 3, idx + 1)
        G, pos, colors, focal, target, trust_circle = create_func()
        draw_scenario(ax, G, pos, colors, focal, target,
                     title, panel, trust_circle)

    plt.tight_layout()

    return fig

if __name__ == '__main__':
    # Create and save figure
    fig = create_figure()

    # Save as SVG for publication
    output_path = '/home/jpoveralls/Documents/Projects_code/Rewiring-Collective-Action/Figs/rewiring_scenarios_6panel.svg'
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    # Also save as PNG for quick viewing
    output_png = output_path.replace('.svg', '.png')
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
    print(f"PNG version saved to: {output_png}")

    plt.show()
