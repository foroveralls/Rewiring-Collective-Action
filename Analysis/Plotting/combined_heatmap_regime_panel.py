#!/usr/bin/env python3
"""
Combined visualization: Stubbornness/Backfirer heatmap + Regime performance panel
Two-row layout: heatmap on top, regime performance panel below
Uses matplotlib's figure combination approach
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from datetime import date
import tempfile
import glob

# Import plotting functions from existing scripts
from plots_heatmaps_stubbornness_backfirer import (
    setup_plotting_style,
    prepare_dataframe,
    create_heatmap_grid
)

from continuous_regime_analysis import (
    load_continuous_data,
    create_continuous_single_panel_plot,
    create_split_panel_plot,
    setup_style,
    FRIENDLY_COLORS as REGIME_COLORS,
    FONT_SIZE as REGIME_FONT_SIZE
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
    # Using equal top/bottom alignment
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1],
                  left=0.08, right=0.95, bottom=0.15, top=0.88, wspace=0.3)

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
        ax_right.set_xlabel('Stubbornness, $w_i$', fontsize=7)
        ax_right.set_ylabel('Opinion, $\\langle a^* \\rangle$', fontsize=7, color='black')
        ax_right_twin.set_ylabel('Polarization, $\\langle P^* \\rangle$', fontsize=7, color='black')
        ax_right.grid(True, alpha=0.3, linewidth=0.3)
        ax_right.tick_params(axis='both', which='major', direction='out', length=3, width=0.5, labelsize=6)
        ax_right_twin.tick_params(axis='both', which='major', direction='out', length=3, width=0.5, labelsize=6)

        # Add regime boundaries
        ax_right.axvline(0.33, color='black', linestyle='--', alpha=0.7, linewidth=0.7)
        ax_right.axvline(0.66, color='black', linestyle='--', alpha=0.7, linewidth=0.7)

        # Add legend
        ax_right.legend(loc='upper right', fontsize=6, frameon=True)

    # Add panel labels (lowercase)
    fig.text(0.08, 0.92, 'a', fontsize=8, fontweight='bold')
    fig.text(0.52, 0.92, 'b', fontsize=8, fontweight='bold')

    plt.close(fig_heatmap)  # Close temporary figure

    return fig

def crop_whitespace(img, threshold=0.95):
    """
    Crop whitespace from image edges

    Args:
        img: Image array (H, W, C) with values in [0, 1]
        threshold: Pixel values above this are considered white

    Returns:
        Cropped image array
    """
    # Convert to grayscale for whitespace detection
    if img.shape[2] == 4:  # RGBA
        gray = img[:, :, :3].mean(axis=2)  # Average RGB channels
    else:  # RGB
        gray = img.mean(axis=2)

    # Find non-white pixels
    non_white_pixels = gray < threshold

    # Find bounding box
    rows = np.any(non_white_pixels, axis=1)
    cols = np.any(non_white_pixels, axis=0)

    if not rows.any() or not cols.any():
        return img  # Return original if all white

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Add small padding (2% of dimension) to avoid cutting too close
    h, w = img.shape[:2]
    pad_h = int(h * 0.02)
    pad_w = int(w * 0.02)

    row_min = max(0, row_min - pad_h)
    row_max = min(h, row_max + pad_h)
    col_min = max(0, col_min - pad_w)
    col_max = min(w, col_max + pad_w)

    return img[row_min:row_max+1, col_min:col_max+1]

def combine_figures_vertically(fig_path1, fig_path2, output_path):
    """
    Combine two figures vertically (heatmap on top, regime below) — two-row layout

    Args:
        fig_path1: Path to top figure (heatmap) - PNG format
        fig_path2: Path to bottom figure (regime) - PNG format
        output_path: Path for combined output (saves as both PDF and PNG)
    """
    try:
        # Load PNG images directly
        img1 = mpimg.imread(fig_path1)
        img2 = mpimg.imread(fig_path2)

        # Crop whitespace from both images
        img1_cropped = crop_whitespace(img1)
        img2_cropped = crop_whitespace(img2)

        # Two-row layout: full width, stacked vertically
        # Use higher DPI (600) for publication quality
        cm = 1/2.54
        fig = plt.figure(figsize=(17.8*cm, 13*cm), dpi=600)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[0.85, 1],
                     left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.03)

        # Add panel a (heatmap) — top row
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1_cropped, interpolation='none')
        ax1.axis('off')
        ax1.text(0.01, 0.99, 'a', transform=ax1.transAxes,
                fontsize=7, fontweight='bold', va='top', ha='left')

        # Add panel b (regime) — bottom row
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(img2_cropped, interpolation='none')
        ax2.axis('off')
        ax2.text(0.01, 0.99, 'b', transform=ax2.transAxes,
                fontsize=7, fontweight='bold', va='top', ha='left')

        # Save combined figure as both PDF and PNG at high quality
        pdf_output = output_path if output_path.endswith('.pdf') else output_path.replace('.png', '.pdf')
        png_output = pdf_output.replace('.pdf', '.png')

        fig.savefig(pdf_output, dpi=600, bbox_inches='tight', format='pdf',
                   metadata={'Creator': 'Collective Rewiring Analysis'})
        fig.savefig(png_output, dpi=600, bbox_inches='tight')

        print(f"Combined figure saved:")
        print(f"  PDF: {pdf_output}")
        print(f"  PNG: {png_output}")

        plt.close(fig)
        return True

    except Exception as e:
        print(f"Error combining figures: {e}")
        return False

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
    heatmap_file = os.environ.get("HEATMAP_CSV") or get_most_recent_file(output_dir, 'heatmap', '.csv')

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

    cm = 1/2.54

    # Generate the stubbornness/backfirer heatmap separately
    print("\nGenerating stubbornness/backfirer heatmap...")
    value_columns = ['state', 'state_std']
    column_labels = {'state': 'avg(x)', 'state_std': 'std(x)'}
    fig_heatmap = create_heatmap_grid(df, value_columns, column_labels, for_combined=False)
    # Shrink heatmap height for better proportions in combined layout
    fig_heatmap.set_size_inches(17.8*cm, 6.5*cm)
    heatmap_pdf_path = f'{output_dir}/heatmap_stubbornness_backfirer_{today}.pdf'
    heatmap_png_path = f'{output_dir}/heatmap_stubbornness_backfirer_{today}.png'
    fig_heatmap.savefig(heatmap_pdf_path, bbox_inches='tight', dpi=300)
    fig_heatmap.savefig(heatmap_png_path, bbox_inches='tight', dpi=300)
    print(f"Stubbornness/backfirer heatmap saved: {heatmap_pdf_path}")
    plt.show()
    plt.close()

    # Generate the regime panel separately (side-by-side: cooperation + polarization)
    regime_png_path = None
    if regime_df is not None:
        print("\nGenerating regime performance panel (side-by-side)...")
        setup_style()

        fig_regime, (ax_coop, ax_polar) = plt.subplots(1, 2, figsize=(17.8*cm, 5*cm))

        main_algorithms = ['Opposite', 'Similar', 'WTF', 'Node2Vec', 'Static', 'Random']
        algorithm_colors = {
            'Opposite': REGIME_COLORS['local (opposite)'],
            'Similar': REGIME_COLORS['local (similar)'],
            'WTF': REGIME_COLORS['empirical wtf'],
            'Node2Vec': REGIME_COLORS['empirical node2vec'],
            'Static': REGIME_COLORS['static'],
            'Random': REGIME_COLORS['random']
        }

        df_main = regime_df[regime_df['algorithm'].isin(main_algorithms)]

        for alg in main_algorithms:
            alg_data = df_main[df_main['algorithm'] == alg].sort_values('stubbornness_numeric')
            if len(alg_data) > 0:
                color = algorithm_colors.get(alg, '#666666')
                ax_coop.plot(alg_data['stubbornness_numeric'], alg_data['mean_cooperation'],
                             'o-', color=color, linewidth=1, markersize=2.5, alpha=0.8, label=alg)
                ax_polar.plot(alg_data['stubbornness_numeric'], alg_data['mean_polarization'],
                              'o-', color=color, linewidth=1, markersize=2.5, alpha=0.8)

        for ax in [ax_coop, ax_polar]:
            ax.axvline(0.4, color='black', linestyle='--', alpha=0.7, linewidth=0.8)
            ax.axvline(0.7, color='black', linestyle='--', alpha=0.7, linewidth=0.8)
            ax.grid(True, alpha=0.3, linewidth=0.3)
            ax.tick_params(labelsize=REGIME_FONT_SIZE-2, length=2, width=0.5)
            ax.set_xlabel('Stubbornness, $w_i$', fontsize=REGIME_FONT_SIZE-1)

        ax_coop.set_ylabel(r'Opinion, $\langle a^* \rangle$', fontsize=REGIME_FONT_SIZE-1)
        ax_polar.set_ylabel(r'Polarization, $\langle P^* \rangle$', fontsize=REGIME_FONT_SIZE-1)

        # Regime labels on cooperation panel
        ylim = ax_coop.get_ylim()
        label_y = ylim[1] + (ylim[1] - ylim[0]) * 0.08
        ax_coop.text(0.2, label_y, 'Low', ha='center', fontsize=REGIME_FONT_SIZE-1, clip_on=False)
        ax_coop.text(0.55, label_y, 'Medium', ha='center', fontsize=REGIME_FONT_SIZE-1, clip_on=False)
        ax_coop.text(0.85, label_y, 'High', ha='center', fontsize=REGIME_FONT_SIZE-1, clip_on=False)

        fig_regime.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=6,
                          fontsize=REGIME_FONT_SIZE-2, frameon=True, fancybox=False,
                          edgecolor='black', facecolor='white')
        plt.tight_layout()

        regime_pdf_path = f'{output_dir}/regime_panel_{today}.pdf'
        regime_png_path = f'{output_dir}/regime_panel_{today}.png'
        fig_regime.savefig(regime_pdf_path, bbox_inches='tight', dpi=300)
        fig_regime.savefig(regime_png_path, bbox_inches='tight', dpi=300)
        print(f"Regime panel saved: {regime_pdf_path}")
        plt.show()
        plt.close()

    # Attempt to combine them using PNG versions
    if regime_png_path and os.path.exists(heatmap_png_path) and os.path.exists(regime_png_path):
        print("\nCombining figures...")
        combined_path = f'{output_dir}/combined_heatmap_regime_{today}.pdf'
        success = combine_figures_vertically(heatmap_png_path, regime_png_path, combined_path)

        if not success:
            print("\n" + "="*60)
            print("Automatic combination failed.")
            print("Generated separate figures:")
            print(f"  a (Heatmap): {heatmap_pdf_path}")
            print(f"  b (Regime): {regime_pdf_path}")
            print("\nTo combine manually, use Inkscape, Illustrator, or ImageMagick")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("Success: combined figure created.")
            print(f"  Combined: {combined_path}")
            print(f"  Heatmap: {heatmap_pdf_path}")
            print(f"  Regime: {regime_pdf_path}")
            print("="*60)
    else:
        print("\n" + "="*60)
        print("Generated figures:")
        print(f"  Heatmap: {heatmap_pdf_path}")
        if regime_png_path:
            print(f"  Regime: {regime_pdf_path}")
        print("="*60)

if __name__ == "__main__":
    main()
