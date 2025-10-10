#!/usr/bin/env python3
"""
Combined visualization: Trajectory dynamics + Network transformation grid
Horizontally combines the truncated trajectory plot with the transformation grid
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from datetime import date

cm = 1/2.54
FONT_SIZE = 7

def combine_figures_horizontally(fig_path1, fig_path2, output_path, width_ratios=[1, 0.7]):
    """
    Combine two PNG figures horizontally

    Args:
        fig_path1: Path to left figure (trajectory) - PNG format
        fig_path2: Path to right figure (transformation grid) - PNG format
        output_path: Path for combined output (will be saved as PNG)
        width_ratios: Relative widths of the two panels
    """
    try:
        # Load PNG images directly
        img1 = mpimg.imread(fig_path1)
        img2 = mpimg.imread(fig_path2)

        # Create combined figure with custom width ratios
        fig = plt.figure(figsize=(17.8*cm, 8*cm), dpi=300)
        gs = GridSpec(1, 2, figure=fig, width_ratios=width_ratios,
                     left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.03)

        # Add panel a (trajectory)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1)
        ax1.axis('off')
        ax1.text(0.01, 0.99, 'a', transform=ax1.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

        # Add panel b (transformation grid)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img2)
        ax2.axis('off')
        ax2.text(0.01, 0.99, 'b', transform=ax2.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

        # Save combined figure as PNG only
        png_output = output_path.replace('.pdf', '.png')
        fig.savefig(png_output, dpi=300, bbox_inches='tight')
        print(f"✓ Combined figure saved: {png_output}")

        plt.close(fig)
        return True

    except Exception as e:
        print(f"Error combining figures: {e}")
        return False

def main():
    """Generate combined trajectory + transformation grid visualization"""
    print("=" * 60)
    print("Combined Trajectory + Transformation Grid Visualization")
    print("=" * 60)

    output_dir = "../../Figs/Combined"
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")

    # Check if required figures exist
    print("\nStep 1: Check for required figures...")

    # Find most recent trajectory figure (PNG format)
    traj_dir = "../../Figs/Trajectories"
    traj_files = [f for f in os.listdir(traj_dir)
                  if f.startswith("network_dynamics") and f.endswith(".png")]

    if not traj_files:
        print("\nERROR: No trajectory PNG figure found!")
        print("Please run 'modified_trajectory_plot.py' first to generate the trajectory figure.")
        return

    # Use the most recent trajectory file
    traj_files.sort(key=lambda x: os.path.getmtime(os.path.join(traj_dir, x)), reverse=True)
    traj_path = os.path.join(traj_dir, traj_files[0])
    print(f"  Found trajectory: {traj_files[0]}")

    # Find transformation grid figure (PNG format)
    grid_dir = "../../Figs/Networks"
    grid_files = [f for f in os.listdir(grid_dir)
                  if f.startswith("transformation_grid") and f.endswith(".png")]

    if not grid_files:
        print("\nERROR: No transformation grid PNG figure found!")
        print("Please run 'transformation_grid_plot.py' first to generate the grid as PNG.")
        return

    # Use the most recent grid file
    grid_files.sort(key=lambda x: os.path.getmtime(os.path.join(grid_dir, x)), reverse=True)
    grid_path = os.path.join(grid_dir, grid_files[0])
    print(f"  Found grid: {grid_files[0]}")

    # Combine the figures
    print("\nStep 2: Combining figures...")
    output_path = f"{output_dir}/combined_trajectory_grid_{today}.png"

    success = combine_figures_horizontally(traj_path, grid_path, output_path,
                                          width_ratios=[1.0, 0.8])

    if success:
        print("\n" + "=" * 60)
        print("Success! Combined figure created.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Automatic combination failed.")
        print("Input figures:")
        print(f"  A (Trajectory): {traj_path}")
        print(f"  B (Grid): {grid_path}")
        print("\nTo combine manually:")
        print("  1. Use Inkscape or Illustrator")
        print("  2. Or use pdfunite: pdfunite fig1.pdf fig2.pdf output.pdf")
        print("  3. Or use ImageMagick: convert +append fig1.pdf fig2.pdf output.pdf")
        print("=" * 60)

if __name__ == "__main__":
    main()
