"""
Test if power law exponents (plo_M, plo_m) affect zero in-degree nodes
These parameters control the out-degree (activity) distribution
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import os
from netin import DPA, DPAH

cm = 1/2.54
FONT_SIZE = 8

def test_plo_values(n=200, seed=42):
    """Test different power law exponent values"""

    # Test different plo values from 1.1 (very skewed) to 5.0 (more uniform)
    # Note: minimum is 1.0 + epsilon per netin validation
    plo_values = [1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    avg_degree = 8

    results = {
        'DPA': {'plo': [], 'zero_in_pct': [], 'zero_out_pct': [],
                'mean_in': [], 'std_in': [], 'mean_out': [], 'std_out': [],
                'max_in': [], 'max_out': []},
        'DPAH': {'plo': [], 'zero_in_pct': [], 'zero_out_pct': [],
                 'mean_in': [], 'std_in': [], 'mean_out': [], 'std_out': [],
                 'max_in': [], 'max_out': []}
    }

    for plo in plo_values:
        print(f"\n{'='*60}")
        print(f"Testing plo={plo}")
        print(f"{'='*60}")

        # DPA
        print(f"  Creating DPA...")
        g_dpa = DPA(
            n=n,
            d=avg_degree/(n-1),
            f_m=0.5,
            plo_M=plo,  # Test parameter
            plo_m=plo,  # Test parameter
            seed=seed
        )
        g_dpa.generate()

        in_deg_dpa = dict(g_dpa.in_degree())
        out_deg_dpa = dict(g_dpa.out_degree())
        zero_in_dpa = sum(1 for d in in_deg_dpa.values() if d == 0)
        zero_out_dpa = sum(1 for d in out_deg_dpa.values() if d == 0)

        results['DPA']['plo'].append(plo)
        results['DPA']['zero_in_pct'].append(100 * zero_in_dpa / n)
        results['DPA']['zero_out_pct'].append(100 * zero_out_dpa / n)
        results['DPA']['mean_in'].append(np.mean(list(in_deg_dpa.values())))
        results['DPA']['std_in'].append(np.std(list(in_deg_dpa.values())))
        results['DPA']['mean_out'].append(np.mean(list(out_deg_dpa.values())))
        results['DPA']['std_out'].append(np.std(list(out_deg_dpa.values())))
        results['DPA']['max_in'].append(max(in_deg_dpa.values()))
        results['DPA']['max_out'].append(max(out_deg_dpa.values()))

        print(f"    Zero in-degree:  {zero_in_dpa}/{n} ({100*zero_in_dpa/n:.1f}%)")
        print(f"    Zero out-degree: {zero_out_dpa}/{n} ({100*zero_out_dpa/n:.1f}%)")
        print(f"    In-degree:  mean={results['DPA']['mean_in'][-1]:.2f}, "
              f"std={results['DPA']['std_in'][-1]:.2f}, max={results['DPA']['max_in'][-1]}")
        print(f"    Out-degree: mean={results['DPA']['mean_out'][-1]:.2f}, "
              f"std={results['DPA']['std_out'][-1]:.2f}, max={results['DPA']['max_out'][-1]}")

        # DPAH
        print(f"  Creating DPAH...")
        g_dpah = DPAH(
            n=n,
            d=avg_degree/(n-1),
            f_m=0.5,
            plo_M=plo,  # Test parameter
            plo_m=plo,  # Test parameter
            h_MM=0.5,
            h_mm=0.5,
            seed=seed
        )
        g_dpah.generate()

        in_deg_dpah = dict(g_dpah.in_degree())
        out_deg_dpah = dict(g_dpah.out_degree())
        zero_in_dpah = sum(1 for d in in_deg_dpah.values() if d == 0)
        zero_out_dpah = sum(1 for d in out_deg_dpah.values() if d == 0)

        results['DPAH']['plo'].append(plo)
        results['DPAH']['zero_in_pct'].append(100 * zero_in_dpah / n)
        results['DPAH']['zero_out_pct'].append(100 * zero_out_dpah / n)
        results['DPAH']['mean_in'].append(np.mean(list(in_deg_dpah.values())))
        results['DPAH']['std_in'].append(np.std(list(in_deg_dpah.values())))
        results['DPAH']['mean_out'].append(np.mean(list(out_deg_dpah.values())))
        results['DPAH']['std_out'].append(np.std(list(out_deg_dpah.values())))
        results['DPAH']['max_in'].append(max(in_deg_dpah.values()))
        results['DPAH']['max_out'].append(max(out_deg_dpah.values()))

        print(f"    Zero in-degree:  {zero_in_dpah}/{n} ({100*zero_in_dpah/n:.1f}%)")
        print(f"    Zero out-degree: {zero_out_dpah}/{n} ({100*zero_out_dpah/n:.1f}%)")

    return results

def plot_plo_results(results):
    """Plot how network properties change with plo parameter"""

    fig, axes = plt.subplots(2, 3, figsize=(18*cm, 12*cm))

    # Plot 1: Zero in-degree vs plo
    ax = axes[0, 0]
    ax.plot(results['DPA']['plo'], results['DPA']['zero_in_pct'],
            'o-', label='DPA', linewidth=2, markersize=6)
    ax.plot(results['DPAH']['plo'], results['DPAH']['zero_in_pct'],
            's-', label='DPAH', linewidth=2, markersize=6)
    ax.set_xlabel('Power law exponent (plo)', fontsize=FONT_SIZE)
    ax.set_ylabel('Zero in-degree nodes (%)', fontsize=FONT_SIZE)
    ax.set_title('Zero In-Degree vs plo', fontsize=FONT_SIZE+1, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE-1)
    ax.axhline(10, color='red', linestyle='--', alpha=0.5, linewidth=1, label='10% threshold')

    # Plot 2: Zero out-degree vs plo
    ax = axes[0, 1]
    ax.plot(results['DPA']['plo'], results['DPA']['zero_out_pct'],
            'o-', label='DPA', linewidth=2, markersize=6)
    ax.plot(results['DPAH']['plo'], results['DPAH']['zero_out_pct'],
            's-', label='DPAH', linewidth=2, markersize=6)
    ax.set_xlabel('Power law exponent (plo)', fontsize=FONT_SIZE)
    ax.set_ylabel('Zero out-degree nodes (%)', fontsize=FONT_SIZE)
    ax.set_title('Zero Out-Degree vs plo', fontsize=FONT_SIZE+1, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE-1)

    # Plot 3: In-degree std deviation vs plo
    ax = axes[0, 2]
    ax.plot(results['DPA']['plo'], results['DPA']['std_in'],
            'o-', label='DPA', linewidth=2, markersize=6)
    ax.plot(results['DPAH']['plo'], results['DPAH']['std_in'],
            's-', label='DPAH', linewidth=2, markersize=6)
    ax.set_xlabel('Power law exponent (plo)', fontsize=FONT_SIZE)
    ax.set_ylabel('In-degree std deviation', fontsize=FONT_SIZE)
    ax.set_title('In-Degree Heterogeneity vs plo', fontsize=FONT_SIZE+1, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE-1)

    # Plot 4: Out-degree std deviation vs plo
    ax = axes[1, 0]
    ax.plot(results['DPA']['plo'], results['DPA']['std_out'],
            'o-', label='DPA', linewidth=2, markersize=6)
    ax.plot(results['DPAH']['plo'], results['DPAH']['std_out'],
            's-', label='DPAH', linewidth=2, markersize=6)
    ax.set_xlabel('Power law exponent (plo)', fontsize=FONT_SIZE)
    ax.set_ylabel('Out-degree std deviation', fontsize=FONT_SIZE)
    ax.set_title('Out-Degree Heterogeneity vs plo', fontsize=FONT_SIZE+1, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE-1)

    # Plot 5: Max in-degree vs plo
    ax = axes[1, 1]
    ax.plot(results['DPA']['plo'], results['DPA']['max_in'],
            'o-', label='DPA', linewidth=2, markersize=6)
    ax.plot(results['DPAH']['plo'], results['DPAH']['max_in'],
            's-', label='DPAH', linewidth=2, markersize=6)
    ax.set_xlabel('Power law exponent (plo)', fontsize=FONT_SIZE)
    ax.set_ylabel('Maximum in-degree', fontsize=FONT_SIZE)
    ax.set_title('Hub Size vs plo', fontsize=FONT_SIZE+1, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE-1)

    # Plot 6: Coefficient of variation for in-degree
    ax = axes[1, 2]
    cv_in_dpa = np.array(results['DPA']['std_in']) / np.array(results['DPA']['mean_in'])
    cv_in_dpah = np.array(results['DPAH']['std_in']) / np.array(results['DPAH']['mean_in'])
    ax.plot(results['DPA']['plo'], cv_in_dpa,
            'o-', label='DPA', linewidth=2, markersize=6)
    ax.plot(results['DPAH']['plo'], cv_in_dpah,
            's-', label='DPAH', linewidth=2, markersize=6)
    ax.set_xlabel('Power law exponent (plo)', fontsize=FONT_SIZE)
    ax.set_ylabel('In-degree CV (std/mean)', fontsize=FONT_SIZE)
    ax.set_title('In-Degree Inequality vs plo', fontsize=FONT_SIZE+1, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE-1)

    plt.tight_layout()

    # Save
    today = date.today().strftime("%Y%m%d")
    os.makedirs("../../Figs/Diagnostics", exist_ok=True)
    filename = f"../../Figs/Diagnostics/dpa_plo_parameter_test_{today}"
    plt.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}.pdf")

    plt.show()

def main():
    print("="*60)
    print("DPA/DPAH Power Law Exponent Parameter Test")
    print("Testing if plo_M/plo_m affects zero in-degree nodes")
    print("="*60)
    print("\nNote: plo controls OUT-degree (activity) distribution")
    print("  - Lower plo (1.1) = more skewed, few very active nodes")
    print("  - Higher plo (5.0) = more uniform, many moderately active nodes")
    print("="*60)

    results = test_plo_values(n=200, seed=42)
    plot_plo_results(results)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Zero In-Degree by plo Value (DPA)")
    print("="*60)
    print(f"{'plo':<10} {'Zero in %':<15} {'Zero out %':<15} {'Max in':<10}")
    print("-" * 50)
    for i, plo in enumerate(results['DPA']['plo']):
        print(f"{plo:<10.1f} {results['DPA']['zero_in_pct'][i]:<15.1f} "
              f"{results['DPA']['zero_out_pct'][i]:<15.1f} {results['DPA']['max_in'][i]:<10}")

    # Find best plo value
    best_idx = np.argmin(results['DPA']['zero_in_pct'])
    best_plo = results['DPA']['plo'][best_idx]
    best_zero = results['DPA']['zero_in_pct'][best_idx]

    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    print(f"Best plo value: {best_plo}")
    print(f"Zero in-degree at best plo: {best_zero:.1f}%")

    if best_zero < 10:
        print(f"\n✓ Using plo={best_plo} significantly reduces zero in-degree nodes!")
        print(f"  Recommend using plo_M={best_plo}, plo_m={best_plo} in your simulations.")
    elif best_zero < 30:
        print(f"\n⚠ Using plo={best_plo} improves but doesn't eliminate the problem.")
        print(f"  Still have {best_zero:.0f}% zero in-degree nodes.")
    else:
        print(f"\n✗ Changing plo does NOT solve the zero in-degree problem.")
        print(f"  Even at best plo={best_plo}, still {best_zero:.0f}% zero in-degree nodes.")
        print(f"  This appears to be a fundamental issue with DPA/DPAH generators.")

    print("="*60)

if __name__ == "__main__":
    main()
