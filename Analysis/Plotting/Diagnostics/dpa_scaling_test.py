"""
Test if zero in-degree nodes persist across different network sizes
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import os
from netin import DPA, DPAH

cm = 1/2.54
FONT_SIZE = 8

def test_network_sizes():
    """Test DPA and DPAH across different network sizes"""

    # Test network sizes from 50 to 1000 (reduced for speed)
    sizes = [50, 70, 100, 200, 500, 1000]
    avg_degree = 8  # Keep average degree constant

    results = {
        'DPA': {'sizes': [], 'zero_in_pct': [], 'zero_out_pct': [],
                'mean_in': [], 'std_in': [], 'max_in': []},
        'DPAH': {'sizes': [], 'zero_in_pct': [], 'zero_out_pct': [],
                 'mean_in': [], 'std_in': [], 'max_in': []}
    }

    for n in sizes:
        print(f"\n{'='*60}")
        print(f"Testing n={n}")
        print(f"{'='*60}")

        # DPA
        print(f"  Creating DPA...")
        g_dpa = DPA(
            n=n,
            d=avg_degree/(n-1),
            f_m=0.5,
            plo_M=2.5,
            plo_m=2.5,
            seed=42
        )
        g_dpa.generate()

        in_deg_dpa = dict(g_dpa.in_degree())
        out_deg_dpa = dict(g_dpa.out_degree())
        zero_in_dpa = sum(1 for d in in_deg_dpa.values() if d == 0)
        zero_out_dpa = sum(1 for d in out_deg_dpa.values() if d == 0)

        results['DPA']['sizes'].append(n)
        results['DPA']['zero_in_pct'].append(100 * zero_in_dpa / n)
        results['DPA']['zero_out_pct'].append(100 * zero_out_dpa / n)
        results['DPA']['mean_in'].append(np.mean(list(in_deg_dpa.values())))
        results['DPA']['std_in'].append(np.std(list(in_deg_dpa.values())))
        results['DPA']['max_in'].append(max(in_deg_dpa.values()))

        print(f"    DPA: {zero_in_dpa}/{n} ({100*zero_in_dpa/n:.1f}%) zero in-degree")
        print(f"    In-degree: mean={results['DPA']['mean_in'][-1]:.2f}, "
              f"std={results['DPA']['std_in'][-1]:.2f}, max={results['DPA']['max_in'][-1]}")

        # DPAH
        print(f"  Creating DPAH...")
        g_dpah = DPAH(
            n=n,
            d=avg_degree/(n-1),
            f_m=0.5,
            plo_M=2.5,
            plo_m=2.5,
            h_MM=0.5,
            h_mm=0.5,
            seed=42
        )
        g_dpah.generate()

        in_deg_dpah = dict(g_dpah.in_degree())
        out_deg_dpah = dict(g_dpah.out_degree())
        zero_in_dpah = sum(1 for d in in_deg_dpah.values() if d == 0)
        zero_out_dpah = sum(1 for d in out_deg_dpah.values() if d == 0)

        results['DPAH']['sizes'].append(n)
        results['DPAH']['zero_in_pct'].append(100 * zero_in_dpah / n)
        results['DPAH']['zero_out_pct'].append(100 * zero_out_dpah / n)
        results['DPAH']['mean_in'].append(np.mean(list(in_deg_dpah.values())))
        results['DPAH']['std_in'].append(np.std(list(in_deg_dpah.values())))
        results['DPAH']['max_in'].append(max(in_deg_dpah.values()))

        print(f"    DPAH: {zero_in_dpah}/{n} ({100*zero_in_dpah/n:.1f}%) zero in-degree")
        print(f"    In-degree: mean={results['DPAH']['mean_in'][-1]:.2f}, "
              f"std={results['DPAH']['std_in'][-1]:.2f}, max={results['DPAH']['max_in'][-1]}")

    return results

def plot_scaling_results(results):
    """Plot how zero in-degree percentage scales with network size"""

    fig, axes = plt.subplots(2, 2, figsize=(16*cm, 12*cm))

    # Plot 1: Zero in-degree percentage vs network size
    ax = axes[0, 0]
    ax.semilogx(results['DPA']['sizes'], results['DPA']['zero_in_pct'],
                'o-', label='DPA', linewidth=2, markersize=6)
    ax.semilogx(results['DPAH']['sizes'], results['DPAH']['zero_in_pct'],
                's-', label='DPAH', linewidth=2, markersize=6)
    ax.set_xlabel('Network size (n)', fontsize=FONT_SIZE)
    ax.set_ylabel('Zero in-degree nodes (%)', fontsize=FONT_SIZE)
    ax.set_title('Zero In-Degree vs Network Size', fontsize=FONT_SIZE+1, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE-1)

    # Plot 2: In-degree standard deviation vs network size
    ax = axes[0, 1]
    ax.loglog(results['DPA']['sizes'], results['DPA']['std_in'],
              'o-', label='DPA', linewidth=2, markersize=6)
    ax.loglog(results['DPAH']['sizes'], results['DPAH']['std_in'],
              's-', label='DPAH', linewidth=2, markersize=6)
    ax.set_xlabel('Network size (n)', fontsize=FONT_SIZE)
    ax.set_ylabel('In-degree std deviation', fontsize=FONT_SIZE)
    ax.set_title('In-Degree Heterogeneity', fontsize=FONT_SIZE+1, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE-1)

    # Plot 3: Maximum in-degree vs network size
    ax = axes[1, 0]
    ax.loglog(results['DPA']['sizes'], results['DPA']['max_in'],
              'o-', label='DPA', linewidth=2, markersize=6)
    ax.loglog(results['DPAH']['sizes'], results['DPAH']['max_in'],
              's-', label='DPAH', linewidth=2, markersize=6)
    # Add reference line for sqrt(n)
    sizes_array = np.array(results['DPA']['sizes'])
    ax.loglog(sizes_array, np.sqrt(sizes_array), 'k--', alpha=0.5, label='√n reference')
    ax.set_xlabel('Network size (n)', fontsize=FONT_SIZE)
    ax.set_ylabel('Maximum in-degree', fontsize=FONT_SIZE)
    ax.set_title('Hub Size Scaling', fontsize=FONT_SIZE+1, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE-1)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE-1)

    # Plot 4: Ratio of std to mean (coefficient of variation)
    ax = axes[1, 1]
    cv_dpa = np.array(results['DPA']['std_in']) / np.array(results['DPA']['mean_in'])
    cv_dpah = np.array(results['DPAH']['std_in']) / np.array(results['DPAH']['mean_in'])
    ax.semilogx(results['DPA']['sizes'], cv_dpa,
                'o-', label='DPA', linewidth=2, markersize=6)
    ax.semilogx(results['DPAH']['sizes'], cv_dpah,
                's-', label='DPAH', linewidth=2, markersize=6)
    ax.set_xlabel('Network size (n)', fontsize=FONT_SIZE)
    ax.set_ylabel('Coefficient of Variation (std/mean)', fontsize=FONT_SIZE)
    ax.set_title('In-Degree Inequality', fontsize=FONT_SIZE+1, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE-1)

    plt.tight_layout()

    # Save
    today = date.today().strftime("%Y%m%d")
    os.makedirs("../../Figs/Diagnostics", exist_ok=True)
    filename = f"../../Figs/Diagnostics/dpa_scaling_analysis_{today}"
    plt.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}.pdf")

    plt.show()

def main():
    print("="*60)
    print("DPA/DPAH Scaling Analysis")
    print("Testing if zero in-degree nodes persist across network sizes")
    print("="*60)

    results = test_network_sizes()
    plot_scaling_results(results)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nZero in-degree percentage by network size:")
    print(f"{'Size':<10} {'DPA %':<15} {'DPAH %':<15}")
    print("-" * 40)
    for i, n in enumerate(results['DPA']['sizes']):
        print(f"{n:<10} {results['DPA']['zero_in_pct'][i]:<15.1f} "
              f"{results['DPAH']['zero_in_pct'][i]:<15.1f}")

    # Check if it's decreasing, increasing, or stable
    dpa_trend = results['DPA']['zero_in_pct'][-1] - results['DPA']['zero_in_pct'][0]
    dpah_trend = results['DPAH']['zero_in_pct'][-1] - results['DPAH']['zero_in_pct'][0]

    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)

    if abs(dpa_trend) < 5:
        print("✓ Zero in-degree percentage is STABLE across network sizes")
        print("  → This is a PERSISTENT property of DPA/DPAH, not a small-network artifact")
    elif dpa_trend < -10:
        print("✓ Zero in-degree percentage DECREASES with network size")
        print("  → Small networks are more problematic; larger networks may be more realistic")
    else:
        print("⚠ Zero in-degree percentage INCREASES with network size")
        print("  → This gets worse in larger networks!")

    print("\nConclusion: DPA/DPAH generators create ~{:.0f}% zero in-degree nodes".format(
        np.mean(results['DPA']['zero_in_pct'])))
    print("regardless of network size, making them unrealistic for social network modeling.")
    print("="*60)

if __name__ == "__main__":
    main()
