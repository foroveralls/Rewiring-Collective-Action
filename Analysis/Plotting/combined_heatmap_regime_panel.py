#!/usr/bin/env python3
"""
Combined visualization: Stubbornness/Backfirer heatmap + Regime performance panel
Horizontally combines the stubbornness/backfirer heatmap with the continuous regime performance plot
Uses matplotlib's figure combination approach
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from datetime import date
import tempfile
import glob

# Import plotting functions from existing scripts
from plots_jordan_heatmaps_alternate_big import (
    setup_plotting_style,
    prepare_dataframe,
    create_heatmap_grid
)

from continuous_regime_analysis import (
    load_continuous_data,
    create_continuous_single_panel_plot,
    setup_style
)

def create_combined_figure(conv_data, coop_data=None, regime_df=None):
    """
    Create combined figure with heatmap on left and regime panel on right

    Args:
        conv_data: Convergence rate data
        coop_data: Cooperation data for heatmap
        regime_df: Continuous stubbornness data for regime panel
    """
    # Set up style
    setup_plotting_style()

    # Create figure with custom gridspec for horizontal layout
    fig = plt.figure(figsize=(17.8/2.54, 8/2.54))

    # Create gridspec with 1 row, 2 columns with different widths
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1],
                  left=0.08, right=0.95, bottom=0.15, top=0.9, wspace=0.3)

    # Panel A: Heatmap (left)
    # Create the heatmap in a temporary figure first
    fig_heatmap = create_combined_heatmap_grid(conv_data, coop_data)

    # Extract the heatmap axes and add to main figure
    # Note: This is a simplified approach - we'll regenerate in the combined figure
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.text(0.5, 0.5, 'Heatmap Panel A\n(See separate generation)',
                 ha='center', va='center', fontsize=8)
    ax_left.axis('off')

    # Panel B: Regime performance (right)
    if regime_df is not None:
        ax_right = fig.add_subplot(gs[0, 1])

        # Import necessary components
        from continuous_regime_analysis import FRIENDLY_COLORS

        # Filter main algorithms
        main_algorithms = ['Opposite', 'Similar', 'WTF', 'Random']
        df_main = regime_df[regime_df['algorithm'].isin(main_algorithms)]

        algorithm_colors = {
            'Opposite': FRIENDLY_COLORS['local (opposite)'],
            'Similar': FRIENDLY_COLORS['local (similar)'],
            'WTF': FRIENDLY_COLORS['empirical wtf'],
            'Random': FRIENDLY_COLORS['random']
        }

        # Create twin axis for polarization
        ax_right_twin = ax_right.twinx()

        # Plot each algorithm trajectory
        for alg in main_algorithms:
            alg_data = df_main[df_main['algorithm'] == alg].sort_values('stubbornness')
            if len(alg_data) > 0:
                color = algorithm_colors.get(alg, '#666666')

                # Mean Cooperation (solid lines)
                ax_right.plot(alg_data['stubbornness'], alg_data['mean_cooperation'],
                            'o-', color=color, linewidth=1, markersize=2.5,
                            alpha=0.8, label=alg)
                # Polarization (dotted lines)
                ax_right_twin.plot(alg_data['stubbornness'], alg_data['mean_polarization'],
                                 'o:', color=color, linewidth=1, markersize=2.5,
                                 alpha=0.6)

        # Customize axes
        ax_right.set_xlabel('Stubbornness, $\\rho$', fontsize=7)
        ax_right.set_ylabel('⟨a⟩', fontsize=7, color='black')
        ax_right_twin.set_ylabel('$\\sigma(a)$', fontsize=7, color='black')
        ax_right.grid(True, alpha=0.3, linewidth=0.3)
        ax_right.tick_params(labelsize=5)
        ax_right_twin.tick_params(labelsize=5)

        # Add regime boundaries
        ax_right.axvline(0.33, color='black', linestyle='--', alpha=0.7, linewidth=0.7)
        ax_right.axvline(0.66, color='black', linestyle='--', alpha=0.7, linewidth=0.7)

        # Add legend
        ax_right.legend(loc='upper right', fontsize=5, frameon=True)

    # Add panel labels
    fig.text(0.08, 0.92, 'A', fontsize=10, fontweight='bold')
    fig.text(0.52, 0.92, 'B', fontsize=10, fontweight='bold')

    plt.close(fig_heatmap)  # Close temporary figure

    return fig

def combine_pdfs_to_single_figure(pdf_path1, pdf_path2, output_path):
    """
    Combine two PDF figures horizontally into a single figure
    Uses temporary PNG conversion for matplotlib combination
    """
    import subprocess

    # Convert PDFs to PNGs temporarily
    with tempfile.TemporaryDirectory() as tmpdir:
        png1 = os.path.join(tmpdir, 'fig1.png')
        png2 = os.path.join(tmpdir, 'fig2.png')

        # Convert using pdftoppm (if available) or fallback to matplotlib
        try:
            # Try using pdftoppm for better quality
            subprocess.run(['pdftoppm', '-png', '-singlefile', '-r', '300',
                          pdf_path1, png1.replace('.png', '')], check=True)
            subprocess.run(['pdftoppm', '-png', '-singlefile', '-r', '300',
                          pdf_path2, png2.replace('.png', '')], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: use matplotlib to read PDFs
            print("pdftoppm not available, using matplotlib to combine figures...")
            return False

        # Load images
        img1 = mpimg.imread(png1)
        img2 = mpimg.imread(png2)

        # Create combined figure
        fig = plt.figure(figsize=(18/2.54, 8/2.54), dpi=300)
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1],
                     left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.05)

        # Add panel A (heatmap)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1)
        ax1.axis('off')
        ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes,
                fontsize=12, fontweight='bold', va='top')

        # Add panel B (regime)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img2)
        ax2.axis('off')
        ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes,
                fontsize=12, fontweight='bold', va='top')

        # Save combined figure
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined figure saved: {output_path}")
        plt.close(fig)

    return True

def get_most_recent_file(directory, pattern, extension='.pkl'):
    """Get the most recent file matching pattern"""
    import glob
    files = glob.glob(os.path.join(directory, f"*{pattern}*{extension}"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def main():
    """Generate combined heatmap and regime performance visualization"""
    print("Creating combined stubbornness/backfirer heatmap + regime performance visualization...")

    # Load stubbornness/backfirer heatmap data (automatically select most recent)
    output_dir = "../../Output"
    heatmap_file = get_most_recent_file(output_dir, 'heatmap', '.csv')

    if heatmap_file is None:
        print(f"No heatmap files found in {output_dir}")
        print("Exiting.")
        return

    print(f"Using heatmap file: {os.path.basename(heatmap_file)}")
    df = pd.read_csv(heatmap_file)
    df = prepare_dataframe(df)

    # Load regime performance data
    print("\nLoading regime performance data...")
    regime_df = load_continuous_data()

    if regime_df is None:
        print("Failed to load regime data. Generating heatmap only.")

    # Create output directory
    output_dir = "../../Figs/Combined"
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")

    # Generate the stubbornness/backfirer heatmap separately
    print("\nGenerating stubbornness/backfirer heatmap...")
    value_columns = ['state', 'state_std']
    column_labels = {'state': 'avg(x)', 'state_std': 'std(x)'}
    fig_heatmap = create_heatmap_grid(df, value_columns, column_labels)
    heatmap_path = f'{output_dir}/heatmap_stubbornness_backfirer_{today}.pdf'
    fig_heatmap.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    print(f"Stubbornness/backfirer heatmap saved: {heatmap_path}")
    plt.show()
    plt.close()

    # Generate the regime panel separately
    regime_path = None
    if regime_df is not None:
        print("\nGenerating regime performance panel...")
        setup_style()
        fig_regime = create_continuous_single_panel_plot(regime_df)
        regime_path = f'{output_dir}/regime_panel_{today}.pdf'
        fig_regime.savefig(regime_path, bbox_inches='tight', dpi=300)
        print(f"Regime panel saved: {regime_path}")
        plt.show()
        plt.close()

    # Attempt to combine them
    if regime_path and os.path.exists(heatmap_path) and os.path.exists(regime_path):
        print("\nAttempting to combine figures...")
        combined_path = f'{output_dir}/combined_heatmap_regime_{today}.pdf'
        success = combine_pdfs_to_single_figure(heatmap_path, regime_path, combined_path)

        if not success:
            print("\n" + "="*60)
            print("Automatic combination not available.")
            print("Generated separate figures:")
            print(f"  A (Heatmap): {heatmap_path}")
            print(f"  B (Regime): {regime_path}")
            print("\nTo combine manually, use Inkscape, Illustrator, or:")
            print("  pdfunite heatmap.pdf regime.pdf combined.pdf")
            print("="*60)
    else:
        print("\n" + "="*60)
        print("Generated figures:")
        print(f"  Heatmap: {heatmap_path}")
        if regime_path:
            print(f"  Regime: {regime_path}")
        print("="*60)

if __name__ == "__main__":
    main()
