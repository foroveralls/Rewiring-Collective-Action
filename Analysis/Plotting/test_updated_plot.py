#!/usr/bin/env python3
"""
Test script to generate updated single-panel plot with backfirer filtering
"""

import os
from datetime import date
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from regime_performance_panels import (
    load_continuous_stubbornness_data,
    create_continuous_single_panel_plot
)

def main():
    """Generate updated single-panel plot with backfirer fraction filtering"""
    print("Creating updated single-panel plot with backfirer filtering...")

    # Load continuous stubbornness data with backfirer_fraction <= 0.4
    df = load_continuous_stubbornness_data(max_backfirer_fraction=0.4)

    if df is None:
        print("No data found. Please check heatmap sweep files.")
        return

    print(f"Loaded data for algorithms: {sorted(df['algorithm'].unique())}")
    print(f"Stubbornness parameter range: {df['stubbornness'].min():.3f} - {df['stubbornness'].max():.3f}")

    # Create output directory
    output_dir = "../../Figs/Regime_Analysis"
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y%m%d")

    # Generate continuous single-panel plot with shaded bands for WTF
    print("Creating continuous single-panel plot with WTF bands...")
    fig = create_continuous_single_panel_plot(df)
    output_path = f"{output_dir}/continuous_single_panel_{today}.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {output_path}")
    plt.close()

    print("Plot generation complete!")

    # Print some statistics
    print("\n=== STATISTICS ===")
    for alg in ['WTF', 'Opposite', 'Static']:
        alg_data = df[df['algorithm'] == alg]
        if len(alg_data) > 0:
            high_stub_data = alg_data[alg_data['stubbornness'] >= 0.7]
            if len(high_stub_data) > 0:
                print(f"{alg} at high stubbornness (>= 0.7):")
                print(f"  Mean cooperation: {high_stub_data['mean_cooperation'].mean():.3f}")
                print(f"  Range: [{high_stub_data['min_cooperation'].mean():.3f}, {high_stub_data['max_cooperation'].mean():.3f}]")

if __name__ == "__main__":
    main()