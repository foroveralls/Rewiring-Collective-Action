"""
Quick test of transformation grid to verify rewiring bug fixes
"""
import sys
sys.path.append('../../')
sys.path.append('..')

from transformation_grid_plot import plot_transformation_grid

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Rewiring Bug Fixes")
    print("=" * 60)
    print("\nRunning with 10 simulations, timesteps [0, 30000]...")
    print("This will verify that hub-and-spoke patterns are eliminated.\n")

    # Run with test parameters (short run, few iterations)
    plot_transformation_grid(n_runs=10, timesteps=[0, 30000], max_timesteps=30000)

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("Check the generated PNG in Figs/Networks/")
    print("Hub-and-spoke patterns should be eliminated if fix works.")
    print("=" * 60)
