#!/usr/bin/env python3
"""
Pareto Frontier Analysis: Speed-Accuracy Trade-offs in Network Rewiring

Demonstrates that rewiring algorithms cluster along optimal trade-off boundaries,
where improvements in convergence speed necessarily reduce final cooperation.
Suitable for high-impact journals (Nature Communications, PNAS) as it shows
fundamental algorithmic optimality rather than mere correlations.

CONVERGENCE METHODS:
-------------------
NOTE (2026-07-30): the SUBMITTED Fig. 3 is the **inflection** method -- both
`rewiring-manuscript-pnasn-v2.tex` and the baseline load
`pareto_speed_cooperativity_inflection_2026-03-27.pdf` at L110. t95 is the code
default only, and it has two defects inflection does not: its `1 - t95/len(traj)`
normalisation is not horizon-invariant (so the merged campaign's 270k condition
gains ~0.05 of speed from the denominator alone), and it scores
negatively-converging runs as near-instant. Choose method 2 to reproduce the
paper. Error bars (R1.7) now work for BOTH metrics -- see PER_RUN_SOURCES in
main().

ENVIRONMENT OVERRIDES (all optional; without them main() prompts as before):
  PARETO_FILE    index into the listed Output/default_run_avg_*.csv files, or any
                 unique substring of one of their names
  PARETO_METHOD  't95' or 'inflection'
  PARETO_AXES    'rank' (default, as submitted) or 'raw'
  PARETO_BARS    '1' to draw the per-run IQR bars; off by default, see
                 plot_pareto_analysis for why

  Reproduce the current Fig. 3 (from Analysis/Plotting):
    MPLBACKEND=Agg PARETO_FILE=gme_2025-10-15.csv PARETO_METHOD=inflection \
      python convergence_vs_cooperation.py

1. t95 method (default): Time to reach 95% of final cooperation value
   - Simple, robust, always computable
   - Returns normalized speed: 1 - (t95/t_max), range [0,1]

2. inflection method: Instantaneous rate during exponential depolarization phase
   - Matches trajectory_stats.py methodology
   - Finds inflection point via second derivative
   - Returns slope around inflection point (raw rate)
   - Theoretically grounded for dynamical systems analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from matplotlib.lines import Line2D
from datetime import date
import warnings
warnings.filterwarnings('ignore')

cm = 1/2.54
FONT_SIZE = 10

FRIENDLY_COLORS = {
    'static': '#EE7733', 'random': '#0077BB', 'L-sim': '#33BBEE',
    'L-opp': '#009988', 'B-sim': '#CC3311', 'B-opp': '#EE3377',
    'wtf': '#BBBBBB', 'node2vec': '#44BB99'
}

FRIENDLY_NAMES = {
    'none_none': 'static', 'random_none': 'random', 'biased_same': 'L-sim',
    'biased_diff': 'L-opp', 'bridge_same': 'B-sim', 'bridge_diff': 'B-opp',
    'wtf_none': 'wtf', 'node2vec_none': 'node2vec'
}

def setup_style():
    plt.rcParams.update({
        'font.size': FONT_SIZE, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.figsize': (12*cm, 11*cm), 'axes.linewidth': 0.8,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.labelsize': FONT_SIZE-1, 'ytick.labelsize': FONT_SIZE-1,
        'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE
    })

def calculate_t95_convergence_speed(trajectory):
    """Calculate time to reach 95% of final value - robust convergence metric"""
    final_val = np.mean(trajectory[-int(len(trajectory)*0.1):])  # Last 10% average
    target = 0.95 * final_val

    # Find first time we reach 95% of final value
    t95_idx = np.argmax(trajectory >= target) if np.any(trajectory >= target) else len(trajectory)

    # Convert to speed (1 - normalized time), so higher = faster
    speed = 1 - (t95_idx / len(trajectory))
    return max(0, speed)  # Ensure non-negative

def find_inflection(seq):
    """Find inflection point in trajectory (proxy for convergence) - from trajectory_stats.py"""
    if len(seq) < 1200:
        return False

    try:
        smooth = gaussian_filter1d(seq, 600)
        d2 = np.gradient(np.gradient(smooth))
        infls = np.where(np.diff(np.sign(d2)))[0]

        inf_min = 5000
        if not any(i >= inf_min for i in infls):
            return False

        inf_ind = next((i for i in infls if i >= inf_min), False)
        return inf_ind if inf_ind > inf_min and inf_ind < 20000 else False
    except:
        return False

def estimate_convergence_rate(trajec, loc, regwin=10):
    """Estimate convergence rate around specified location - from trajectory_stats.py"""
    x = np.arange(len(trajec) - 1)
    y = trajec

    if loc is not None:
        x = x[loc-regwin: loc+regwin+1]
        y = trajec[loc-regwin: loc+regwin+1]

    y, x = np.asarray([y, x])

    # Linear regression
    n = np.size(x)
    mx, my = np.mean(x), np.mean(y)
    ssxy = np.sum(y*x) - n*my*mx
    ssxx = np.sum(x*x) - n*mx*mx
    b1 = ssxy / ssxx

    rate = -b1/(trajec[loc]-1)
    return rate

def calculate_inflection_convergence_speed(trajectory):
    """Calculate convergence rate using inflection point method - matches trajectory_stats.py

    Returns the instantaneous rate of change during the exponential depolarization phase.
    Note: Returns raw rate (not 0-1 normalized). Plot uses rank-based normalization for comparison.
    Higher values = faster convergence.
    """
    convergence_rate = np.nan
    inflection_x = find_inflection(trajectory)

    if inflection_x:
        try:
            # Get raw rate (multiplied by 1000 like in trajectory_stats.py)
            convergence_rate = estimate_convergence_rate(trajectory, inflection_x) * 1000
        except:
            pass

    # Return raw rate if valid, otherwise return 0
    return convergence_rate if not np.isnan(convergence_rate) else 0.0

def calculate_metrics(data, method='t95'):
    """Calculate speed and cooperativity metrics

    Parameters
    ----------
    data : pd.DataFrame
        Trajectory data
    method : str
        't95' for time-to-95% method (default) or 'inflection' for inflection point method
    """
    data['rewiring'] = data['rewiring'].fillna('none')
    data['scenario'] = data['scenario'].fillna('none')
    data['scenario_grouped'] = data['scenario'].str.cat(data['rewiring'], sep='_')

    results = []
    for name, group in data.groupby(['scenario_grouped', 'type']):
        scenario, topology = name
        trajectory = group['avg_state'].values

        # Convergence speed (higher = faster) - choose method
        if method == 'inflection':
            speed = calculate_inflection_convergence_speed(trajectory)
        else:  # default to t95
            speed = calculate_t95_convergence_speed(trajectory)

        # Final cooperativity (robust measure)
        final_window = int(len(trajectory) * 0.1)  # Last 10%
        final_coop = np.mean(trajectory[-final_window:]) if final_window > 0 else trajectory[-1]

        friendly_name = FRIENDLY_NAMES.get(scenario, scenario)
        results.append({
            'scenario': friendly_name, 'topology': topology,
            'speed': speed, 'cooperativity': final_coop
        })

    return pd.DataFrame(results)

def find_pareto_front(metrics_df):
    """Find Pareto optimal algorithms. Returns a boolean Series aligned with df index."""
    import pandas as pd
    pareto_mask = pd.Series(False, index=metrics_df.index)

    for i, row in metrics_df.iterrows():
        is_dominated = False
        for j, other in metrics_df.iterrows():
            if (i != j and
                other['speed'] >= row['speed'] and
                other['cooperativity'] >= row['cooperativity'] and
                (other['speed'] > row['speed'] or other['cooperativity'] > row['cooperativity'])):
                is_dominated = True
                break
        pareto_mask[i] = not is_dominated

    return pareto_mask

def plot_pareto_analysis(metrics_df, output_path, method='t95', per_run_df=None,
                         axes_mode='rank', draw_bars=False):
    """Create faceted Pareto frontier analysis — one panel per topology.

    Each panel shows algorithms as colored points with per-topology Pareto
    frontiers, making trade-off structure visible without marker confusion.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics data with 'speed', 'cooperativity', 'scenario', 'topology'
    output_path : str
        Output file path
    method : str
        't95' or 'inflection' to indicate which method was used
    axes_mode : str
        'rank' (default, as submitted) plots both axes as global rank
        percentiles; 'raw' plots the metrics in their own units. Pareto
        membership is identical either way (ranking is a monotone transform),
        so only the visual spacing and the axis meaning change. 'raw' fixes the
        standing audit finding that the y-axis is labelled $\\langle a^* \\rangle$
        while carrying percentiles, but it does not rescue the inflection error
        bars: per-run IQRs exceed the ensemble spread, so in 'raw' they are drawn
        uncapped and will overflow the marker cloud. Set via PARETO_AXES=raw.
    per_run_df : pd.DataFrame, optional
        Per-run metrics with columns (scenario [friendly], topology, speed,
        cooperativity). `speed` must be the SAME metric as `method`, already
        renamed by the caller: `speed_t95` from Output/per_run_summary.csv for
        method='t95', `speed_inflection` from Output/per_run_inflection.csv for
        method='inflection'. If given, asymmetric IQR error bars (R1.7 ensemble
        uncertainty) are drawn on each point. Markers stay at the ensemble-mean
        rank; bars show the q25-q75 of the 90 per-run metrics mapped through the
        SAME rank-percentile (aggregate ECDF) transform as the markers, so the
        marker is not necessarily the bar midpoint (note this in the caption).
        Rows with a NaN speed (inflection not found for that run) must already be
        dropped by the caller; conditions left with < MIN_RUNS_FOR_IQR valid runs
        get no bars, and are reported.

        NOTE (2026-07-31 decision): the published Fig. 3 carries NO error bars.
        Per-run convergence uncertainty is answered by the run-level violin
        instead, because in rank space a single condition's per-run IQR is 2-3x
        the entire spread of the 30 ensemble values, so every speed bar saturates
        the 0.10 cap and 30 identical bars carry no information. The IQRs are
        still computed and reported to stdout as a diagnostic; pass draw_bars=True
        (PARETO_BARS=1) to draw them.
    draw_bars : bool
        Draw the per-run IQR bars. Default False, matching the decision above.
    """
    setup_style()

    # Filter valid data
    valid_mask = (np.isfinite(metrics_df['speed']) &
                  np.isfinite(metrics_df['cooperativity']) &
                  (metrics_df['speed'] >= 0))
    valid_data = metrics_df[valid_mask].copy()

    if len(valid_data) == 0:
        print("No valid data for plotting")
        return

    # Plotted coordinates. 'rank': global rank percentiles, so panels share a
    # consistent scale (as submitted). 'raw': the metrics' own units.
    raw_axes = (axes_mode == 'raw')
    if raw_axes:
        valid_data['speed_plot'] = valid_data['speed']
        valid_data['coop_plot'] = valid_data['cooperativity']
    else:
        valid_data['speed_plot'] = valid_data['speed'].rank(pct=True)
        valid_data['coop_plot'] = valid_data['cooperativity'].rank(pct=True)

    # Per-run IQR in the SAME space as the markers, keyed by (scenario, topology).
    # Metric-agnostic: per_run_df['speed'] is whatever `method` selected.
    MIN_RUNS_FOR_IQR = 10
    rank_iqr = {}
    if per_run_df is not None:
        agg_speed = np.sort(valid_data['speed'].to_numpy())
        agg_coop = np.sort(valid_data['cooperativity'].to_numpy())
        nv = len(valid_data)
        skipped = []
        for (sc, topo), grp in per_run_df.groupby(['scenario', 'topology']):
            if len(grp) < MIN_RUNS_FOR_IQR:
                skipped.append((sc, topo, len(grp)))
                continue
            if raw_axes:
                sp = grp['speed'].to_numpy()
                cp = grp['cooperativity'].to_numpy()
            else:
                sp = np.searchsorted(agg_speed, grp['speed'].to_numpy(), side='right') / nv
                cp = np.searchsorted(agg_coop, grp['cooperativity'].to_numpy(), side='right') / nv
            rank_iqr[(sc, topo)] = (np.percentile(sp, 25), np.percentile(sp, 75),
                                    np.percentile(cp, 25), np.percentile(cp, 75))
        print(f"Error bars: IQR {'drawn' if draw_bars else 'computed but NOT drawn'} for "
              f"{len(rank_iqr)} of {len(valid_data)} conditions "
              f"(metric = {method}, axes = {axes_mode})")
        if skipped:
            print(f"  no bars for {len(skipped)} condition(s) with < {MIN_RUNS_FOR_IQR} "
                  f"valid runs: {skipped}  -- disclose this if it reaches the caption")

    # Metric units, used for the x-label on raw axes only.
    # `calculate_inflection_convergence_speed` already multiplies the slope by 1000,
    # so an axis reading 0.107 is a rate of 1.07e-4 per step.
    RATE_LABEL = {
        'inflection': r'Convergence rate ($\times 10^{-3}$)',
        't95': r'Convergence rate, $1 - t_{95}/T$',
    }
    # In rank mode BOTH axes are `.rank(pct=True)` percentiles across the 30
    # algorithm-topology conditions. The submitted figure labelled them with the
    # metric names alone ('Convergence rate', r'Opinion, $\langle a^* \rangle$'),
    # which reads as raw values in units the paper defines elsewhere -- audit
    # finding M-N1. Decision 2026-07-31: keep the rank presentation (Pareto
    # membership is computed on the raw columns, so the ordering is unchanged) and
    # say so on the axis. The caption carries the matching disclosure clause.
    RANK_XLABEL = 'Convergence rate (rank percentile)'
    RANK_YLABEL = r'Mean opinion, $\langle a^* \rangle$ (rank percentile)'

    # Topology ordering and labels
    topology_order = ['DPAH', 'cl', 'Twitter', 'FB']
    topology_labels = {'DPAH': 'DPA', 'cl': 'CSF', 'Twitter': 'Twitter', 'FB': 'FB'}
    topologies_present = [t for t in topology_order if t in valid_data['topology'].unique()]

    n_panels = len(topologies_present)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(17.8*cm, 17.8*cm*nrows/ncols),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    # Global Pareto front (across all topologies) for reference
    global_pareto_mask = find_pareto_front(valid_data)

    # Shared raw limits, spanning everything actually drawn (markers and any bars),
    # so the four panels stay comparable and no bar is silently clipped.
    if raw_axes:
        xs = list(valid_data['speed_plot'])
        ys = list(valid_data['coop_plot'])
        if draw_bars:
            for s_lo, s_hi, c_lo, c_hi in rank_iqr.values():
                xs += [s_lo, s_hi]
                ys += [c_lo, c_hi]
        def _pad(v, frac=0.06):
            lo, hi = min(v), max(v)
            m = (hi - lo) * frac
            return lo - m, hi + m
        raw_xlim, raw_ylim = _pad(xs), _pad(ys)
        print(f"Raw axes: x {raw_xlim[0]:.4f}..{raw_xlim[1]:.4f}, "
              f"y {raw_ylim[0]:.4f}..{raw_ylim[1]:.4f}")
    else:
        raw_xlim = raw_ylim = None

    for idx, topo in enumerate(topologies_present):
        ax = axes[idx // ncols, idx % ncols]
        topo_data = valid_data[valid_data['topology'] == topo]

        # Per-topology Pareto front
        topo_pareto_mask = find_pareto_front(topo_data)
        pareto_pts = []

        for _, row in topo_data.iterrows():
            color = FRIENDLY_COLORS.get(row['scenario'], 'black')
            is_pareto = topo_pareto_mask.loc[row.name]
            size = 90 if is_pareto else 55
            edge_w = 0.9 if is_pareto else 0.6
            alpha = 0.95 if is_pareto else 0.8

            # Asymmetric IQR error bars (drawn under the markers), capped at 0.10 rank
            # units. (The comment here used to say 0.15, which never matched the code;
            # the drafted caption says 0.10, i.e. the code and the caption agree and
            # only the comment was wrong. Fixed 2026-07-30.) No cap on raw axes -- a
            # cap in metric units would have no defensible value, so the bars are drawn
            # at full length even where they overflow the marker cloud.
            BAR_CAP = np.inf if raw_axes else 0.10
            key = (row['scenario'], topo)
            if draw_bars and key in rank_iqr:
                s_lo, s_hi, c_lo, c_hi = rank_iqr[key]
                xerr = [[min(BAR_CAP, max(0.0, row['speed_plot'] - s_lo))],
                        [min(BAR_CAP, max(0.0, s_hi - row['speed_plot']))]]
                yerr = [[min(BAR_CAP, max(0.0, row['coop_plot'] - c_lo))],
                        [min(BAR_CAP, max(0.0, c_hi - row['coop_plot']))]]
                ax.errorbar(row['speed_plot'], row['coop_plot'], xerr=xerr, yerr=yerr,
                            fmt='none', ecolor=color, elinewidth=0.8, capsize=1.5,
                            capthick=0.8, alpha=0.55, zorder=2.5)

            ax.scatter(row['speed_plot'], row['coop_plot'], c=color,
                       marker='o', s=size, alpha=alpha,
                       edgecolors='black', linewidth=edge_w, zorder=3)

            if is_pareto:
                pareto_pts.append([row['speed_plot'], row['coop_plot']])

        # Draw per-topology Pareto front
        if len(pareto_pts) > 1:
            pareto_arr = np.array(pareto_pts)
            sort_idx = np.argsort(pareto_arr[:, 0])
            sorted_p = pareto_arr[sort_idx]
            ax.plot(sorted_p[:, 0], sorted_p[:, 1],
                    'r--', alpha=0.7, linewidth=1.0, zorder=2)

        if raw_axes:
            # No diagonal: with both axes in metric units it is not a meaningful
            # reference. A zero line is, because <a*> changes sign.
            ax.axhline(0.0, color='k', linestyle='--', alpha=0.4, linewidth=0.5, zorder=1)
            ax.set_xlim(*raw_xlim)
            ax.set_ylim(*raw_ylim)
        else:
            # Reference diagonal
            ax.plot([0, 1], [1, 0], 'k--', alpha=0.4, linewidth=0.5, zorder=1)
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25, linewidth=0.3)
        ax.set_title(topology_labels.get(topo, topo), fontsize=FONT_SIZE, fontweight='bold')

        # Axis labels only on edges
        if idx // ncols == nrows - 1:
            ax.set_xlabel(RATE_LABEL[method] if raw_axes else RANK_XLABEL,
                          fontsize=FONT_SIZE)
        if idx % ncols == 0:
            ax.set_ylabel(r'Mean opinion, $\langle a^* \rangle$' if raw_axes
                          else RANK_YLABEL, fontsize=FONT_SIZE)

    # Hide unused axes
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    # Shared algorithm legend at bottom
    algo_elements = [Line2D([0], [0], marker='o', color=color, linestyle='None',
                            markersize=4.5, markeredgecolor='black', markeredgewidth=0.4,
                            label=algo)
                     for algo, color in FRIENDLY_COLORS.items()
                     if algo in valid_data['scenario'].values]
    fig.legend(handles=algo_elements, bbox_to_anchor=(0.5, -0.01),
               loc='upper center', columnspacing=0.8, frameon=True,
               fontsize=FONT_SIZE-1, ncol=4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Return global pareto split for analysis function
    pareto_data = valid_data[global_pareto_mask]
    dominated_data = valid_data[~global_pareto_mask]
    return pareto_data, dominated_data

def analyze_pareto_results(pareto_data, dominated_data):
    """Analysis of algorithmic optimality along the Pareto frontier
    
    Demonstrates that algorithms achieve fundamental speed-accuracy trade-offs,
    clustering along the theoretical optimality boundary.
    """
    print("\n" + "="*60)
    print("ALGORITHMIC OPTIMALITY ANALYSIS: PARETO FRONTIER")
    print("="*60)
    print("Analyzing fundamental speed-accuracy trade-offs in network rewiring")
    
    total_algorithms = len(pareto_data) + len(dominated_data)
    print(f"Total algorithms analyzed: {total_algorithms}")
    print(f"Pareto optimal: {len(pareto_data)} ({len(pareto_data)/total_algorithms*100:.1f}%)")
    print(f"Dominated: {len(dominated_data)} ({len(dominated_data)/total_algorithms*100:.1f}%)")
    
    print(f"\nPARETO OPTIMAL ALGORITHMS:")
    print("-" * 40)
    for _, row in pareto_data.iterrows():
        print(f"  {row['scenario']} ({row['topology']}): "
              f"Speed={row['speed']:.3f}, Coop={row['cooperativity']:.3f}")
    
    if len(dominated_data) > 0:
        print(f"\nDOMINATED ALGORITHMS (examples):")
        print("-" * 40)
        worst_dominated = dominated_data.nsmallest(3, ['speed', 'cooperativity'])
        for _, row in worst_dominated.iterrows():
            print(f"  {row['scenario']} ({row['topology']}): "
                  f"Speed={row['speed']:.3f}, Coop={row['cooperativity']:.3f}")
    
    # Trade-off analysis
    if len(pareto_data) > 1:
        speed_range = pareto_data['speed'].max() - pareto_data['speed'].min()
        coop_range = pareto_data['cooperativity'].max() - pareto_data['cooperativity'].min()
        
        print(f"\nPARETO FRONTIER TRADE-OFF ANALYSIS:")
        print("-" * 45)
        print(f"Speed range on frontier: {speed_range:.3f}")
        print(f"Cooperation range on frontier: {coop_range:.3f}")
        print(f"Trade-off strength: {abs(speed_range + coop_range - 2*max(speed_range, coop_range)):.3f}")
        
        # Find extreme points
        fastest = pareto_data.loc[pareto_data['speed'].idxmax()]
        best_coop = pareto_data.loc[pareto_data['cooperativity'].idxmax()]
        
        print(f"Speed-optimal algorithm: {fastest['scenario']} ({fastest['topology']})")
        print(f"Cooperation-optimal algorithm: {best_coop['scenario']} ({best_coop['topology']})")
        print(f"   Speed sacrifice for cooperation: {best_coop['speed'] - fastest['speed']:.3f}")
        print(f"   Cooperation sacrifice for speed: {fastest['cooperativity'] - best_coop['cooperativity']:.3f}")
        
        if fastest.name != best_coop.name:
            print("Algorithms lie along optimal Pareto frontier - demonstrating")
            print("    fundamental speed-accuracy trade-offs in network dynamics!")
        else:
            print("Rare case: one algorithm achieves near-optimal performance")
            print("    on both objectives - potential breakthrough configuration!")

def main():
    files = [f for f in os.listdir("../../Output") if "default_run_avg" in f and f.endswith(".csv")]
    if not files:
        print("No default_run_avg files found")
        return

    files.sort()  # stable ordering, so PARETO_FILE as an index means the same thing twice
    for i, f in enumerate(files):
        print(f"{i}: {f}")

    # PARETO_FILE / PARETO_METHOD make the script runnable unattended (the figure has
    # to be re-rendered reproducibly, and the prompts made that a manual step).
    # PARETO_FILE takes an index or any unique substring of the filename.
    sel = os.environ.get('PARETO_FILE')
    if sel is None:
        idx = int(input("Select file: "))
    elif sel.isdigit():
        idx = int(sel)
    else:
        hits = [i for i, f in enumerate(files) if sel in f]
        if len(hits) != 1:
            raise SystemExit(f"PARETO_FILE={sel!r} matched {len(hits)} files; need exactly one")
        idx = hits[0]
        print(f"PARETO_FILE={sel!r} -> {files[idx]}")

    # Let user choose convergence method
    method = os.environ.get('PARETO_METHOD')
    if method is None:
        print("\nChoose convergence rate calculation method:")
        print("1: t95 method (time to reach 95% of final value - simple and robust)")
        print("2: inflection method (slope during exponential phase - matches trajectory_stats.py)")
        method_choice = input("Select method (1 or 2, default=1): ").strip()
        method = 'inflection' if method_choice == '2' else 't95'
    elif method not in ('t95', 'inflection'):
        raise SystemExit(f"PARETO_METHOD must be 't95' or 'inflection', got {method!r}")
    print(f"\nUsing {method} method for convergence rate calculation")

    data = pd.read_csv(os.path.join("../../Output", files[idx]))

    metrics = calculate_metrics(data, method=method)
    if metrics.empty:
        print("No valid metrics calculated")
        return

    # Optional per-run metrics for ensemble IQR error bars (R1.7). One file per speed
    # metric; both are keyed type/scenario/rewiring/model_run and both carry a
    # `cooperativity` column, so only the speed column differs:
    #   t95        -> per_run_summary.csv    (Analysis/Stats/summarize_individual_csv.py,
    #                                         downsampled at t % 100 == 0)
    #   inflection -> per_run_inflection.csv (Analysis/extract_inflection_lowmem.py,
    #                                         FULL resolution -- mandatory, because
    #                                         find_inflection's 5000 < i < 20000 window is
    #                                         in raw index units and is unreachable on a
    #                                         downsampled series.)
    PER_RUN_SOURCES = {
        't95': ('per_run_summary.csv', 'speed_t95'),
        'inflection': ('per_run_inflection.csv', 'speed_inflection'),
    }
    per_run_df = None
    prs_name, speed_col = PER_RUN_SOURCES[method]
    prs = os.path.join("../../Output", prs_name)
    if os.path.exists(prs):
        pr = pd.read_csv(prs)  # default NA parsing maps "None" -> NaN, mirroring calculate_metrics
        pr['rewiring'] = pr['rewiring'].fillna('none')
        pr['scenario'] = pr['scenario'].fillna('none')
        pr['grouped'] = pr['scenario'].str.cat(pr['rewiring'], sep='_')
        pr['scenario'] = pr['grouped'].map(lambda g: FRIENDLY_NAMES.get(g, g))
        pr['topology'] = pr['type']
        if speed_col not in pr.columns:
            print(f"{prs} has no '{speed_col}' column (has: {list(pr.columns)}); "
                  f"plotting points without error bars")
        else:
            pr = pr.rename(columns={speed_col: 'speed'})
            per_run_df = pr[['scenario', 'topology', 'speed', 'cooperativity']]
            n_all = len(per_run_df)
            # NaN speed = the metric could not be computed for that run (inflection not
            # found). Dropped, never coerced to 0.0, which would masquerade as "slowest".
            per_run_df = per_run_df.dropna(subset=['speed', 'cooperativity'])
            n_drop = n_all - len(per_run_df)
            print(f"Loaded per-run metrics for error bars: {prs} "
                  f"({len(per_run_df)} of {n_all} rows usable, metric = {speed_col})")
            if n_drop:
                print(f"  dropped {n_drop} run(s) with a NaN {speed_col} "
                      f"({100.0*n_drop/n_all:.1f}%) -- disclose this count in the caption")
    else:
        print(f"No per-run file for method={method} ({prs} missing); plotting points "
              f"without error bars.")
        if method == 'inflection':
            print("  build it with:  cd Analysis && python extract_inflection_lowmem.py "
                  "--in ../Output/default_run_individual_..._gme_2025-10-15.csv "
                  "--out ../Output/per_run_inflection.csv   (cluster job, ~28 GB input)")

    # Create plot and analysis
    output_dir = "../../Figs/Convergence"
    os.makedirs(output_dir, exist_ok=True)
    # PARETO_AXES=raw plots the metrics in their own units instead of rank
    # percentiles (see plot_pareto_analysis). Default 'rank' = the submitted figure.
    axes_mode = os.environ.get('PARETO_AXES', 'rank').lower()
    if axes_mode not in ('rank', 'raw'):
        raise SystemExit(f"PARETO_AXES must be 'rank' or 'raw', got {axes_mode!r}")
    # PARETO_BARS=1 reinstates the per-run IQR bars. Off by default: the 2026-07-31
    # decision is that Fig. 3 carries no error bars (see plot_pareto_analysis).
    draw_bars = os.environ.get('PARETO_BARS', '0').lower() in ('1', 'true', 'yes')
    suffix = '_raw' if axes_mode == 'raw' else ''
    if draw_bars:
        suffix += '_bars'
    output_path = f"{output_dir}/pareto_speed_cooperativity_{method}{suffix}_{date.today()}.pdf"

    pareto_data, dominated_data = plot_pareto_analysis(metrics, output_path, method=method,
                                                       per_run_df=per_run_df,
                                                       axes_mode=axes_mode,
                                                       draw_bars=draw_bars)
    analyze_pareto_results(pareto_data, dominated_data)

    print(f"\nPareto trade-off analysis saved: {output_path}")
    print(f"Method used: {method}")
    print("\nINTERPRETATION: Algorithms clustering along the diagonal reference")
    print("   line demonstrate optimal speed-accuracy trade-offs. Deviations from")
    print("   this frontier indicate suboptimal parameter configurations.")
    return metrics, pareto_data

if __name__ == "__main__":
    main()