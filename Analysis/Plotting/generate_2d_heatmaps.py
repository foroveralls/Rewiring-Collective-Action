#!/usr/bin/env python3
"""
Generate 2D heatmaps showing cooperation and polarization as functions of both
stubbornness and backfirer fraction (not averaged)
"""

import os
from datetime import date
import matplotlib.pyplot as plt
from continuous_regime_analysis import (
    load_2d_parameter_data,
    create_2d_heatmap_grid,
    setup_style
)

def main():
    """Generate 2D heatmap visualizations"""
    print("Creating 2D parameter space heatmaps...")

    # Load 2D parameter data (not averaged)
    df_2d = load_2d_parameter_data()
    if df_2d is None:
        print("Error: Could not load 2D parameter data")
        return

    print(f"Loaded {len(df_2d)} data points")
    print(f"Parameter combinations: {len(df_2d.groupby(['stubbornness', 'backfirer_fraction']))}")

    # Create output directory
    output_dir = "../../Figs/Regime_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")

    # Generate 2D heatmap grid
    print("Creating 2D heatmap grid...")
    fig = create_2d_heatmap_grid(df_2d)
    output_path = f"{output_dir}/2d_heatmap_stubbornness_backfirer_{today}.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D heatmap saved: {output_path}")
    plt.show()
    plt.close()

    print("2D heatmap generation complete!")

if __name__ == "__main__":
    main()