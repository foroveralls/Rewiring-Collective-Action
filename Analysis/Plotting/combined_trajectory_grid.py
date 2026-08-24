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
FONT_SIZE = 11  # Increased from 7 for better readability

def combine_figures_horizontally(fig_path1, fig_path2, output_path, width_ratios=[1, 0.7]):
    """
    Combine two figures horizontally with maximum quality preservation

    Args:
        fig_path1: Path to left figure (trajectory) - PNG or PDF format
        fig_path2: Path to right figure (transformation grid) - PNG or PDF format
        output_path: Path for combined output (will be saved as both PDF and PNG)
        width_ratios: Relative widths of the two panels
    """
    try:
        # Load images - prefer PNG for matplotlib compatibility
        # Use high DPI for maximum quality
        img1 = mpimg.imread(fig_path1)
        img2 = mpimg.imread(fig_path2)

        # Create combined figure with custom width ratios
        # Use higher DPI (600) for publication quality
        # Increased height from 8cm to 10cm for better visibility in manuscript
        fig = plt.figure(figsize=(17.8*cm, 10*cm), dpi=600)
        gs = GridSpec(1, 2, figure=fig, width_ratios=width_ratios,
                     left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.02)  # Reduced wspace from 0.03 to 0.02

        # Add panel a (trajectory)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1, interpolation='none')  # No interpolation to preserve sharpness
        ax1.axis('off')
        ax1.text(0.01, 0.99, 'a', transform=ax1.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')

        # Add panel b (transformation grid)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img2, interpolation='none')  # No interpolation to preserve sharpness
        ax2.axis('off')
        ax2.text(-0.02, 0.99, 'b', transform=ax2.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='right')  # Moved left (x=-0.02) to avoid occluding network grid

        # Save combined figure as both PDF (for publication) and PNG (for preview)
        pdf_output = output_path if output_path.endswith('.pdf') else output_path.replace('.png', '.pdf')
        png_output = pdf_output.replace('.pdf', '.png')

        # Save PDF with embedded high-quality raster
        fig.savefig(pdf_output, dpi=600, bbox_inches='tight', format='pdf',
                   metadata={'Creator': 'Collective Rewiring Analysis'})
        # Save PNG at high DPI for preview/web use
        fig.savefig(png_output, dpi=600, bbox_inches='tight')

        print(f"Combined figure saved:")
        print(f"  PDF: {pdf_output}")
        print(f"  PNG: {png_output}")

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
    # Prefer "for_combined" version which has matching height with transformation grid
    traj_dir = "../../Figs/Trajectories"

    # First look for the "for_combined" version
    traj_files_combined = [f for f in os.listdir(traj_dir)
                          if "for_combined" in f and f.endswith(".png")]

    if traj_files_combined:
        # Use the most recent "for_combined" file
        traj_files_combined.sort(key=lambda x: os.path.getmtime(os.path.join(traj_dir, x)), reverse=True)
        traj_path = os.path.join(traj_dir, traj_files_combined[0])
        print(f"  Found trajectory (for_combined): {traj_files_combined[0]}")
    else:
        # Fall back to regular trajectory files
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
        print(f"  WARNING: Using regular trajectory plot. For better height matching, regenerate with for_combined=True")

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
                                          width_ratios=[0.85, 0.8])

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
