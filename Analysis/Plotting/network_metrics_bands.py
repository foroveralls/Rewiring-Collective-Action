#!/usr/bin/env python3
"""
Time evolution of network metrics WITH run-to-run uncertainty (SI Fig. S2), for
reviewer R1.7: "Error bars, confidence intervals, or distributions across runs
would be especially helpful for final mean opinion, final polarization,
convergence rate, and network metrics."

The published Nov-2025 figure showed mean +/- STANDARD ERROR at alpha 0.15. That is
the precision of the mean, which shrinks with n and is exactly the statistic that
hides the bistability R1.7 is asking about. This draws the MEDIAN with an
interquartile band across runs, matching the band definition used for Fig. 2's
trajectory ribbons, and optionally overlays the individual runs as hairlines.

DATA: Output/network_metrics_per_run.csv from
Analysis/extract_network_metrics_lowmem.py, one row per
(scenario, rewiring, type, model_run, timestep) with clustering, modularity,
assortativity, gini_in_degree. Snapshots are kept for even runs only, so n = 45
per combo, not 90. Read that script's header before quoting any number: the metric
conventions changed (directed graphs are now measured directed) and the values
differ from the published figure.

RESIDUALS (--residuals). On the shared (0,1) axis a typical IQR is ~0.005 wide and
renders sub-pixel, which reads as "no run-to-run variation" when in fact only the
static condition has none (across-run SD at the final step is 0.002-0.03 for every
rewiring algorithm). --residuals adds a strip under each metric row holding
each run minus its condition's median trajectory, so the medians keep the comparable
(0,1) axis and the spread is still legible. Two styles:

  --residual-style overlay  (default) half-height strip, all topologies in it. In the
                            combined layout they share one colour, so what you read is
                            their envelope, not which topology is which.
  --residual-style lanes    full-height strip split into one baseline per topology,
                            indexed by the global TYPE_ORDER so a lane means the same
                            thing in every panel and WTF's undirected topologies read
                            as gaps. Ignored when a figure holds a single topology.

TWO LAYOUTS, one script:
  --topology all   (default) all four topologies in each panel, IQR bands only.
                   Drop-in replacement for the published combined figure; four
                   same-coloured bands per panel, so hairlines are not drawn.
  --topology DPAH  one topology, so each panel is a single algorithm-topology
                   combination: IQR band + the 45 individual runs as hairlines.
                   This is the version that actually shows the distribution.

Usage:
    python network_metrics_bands.py
    python network_metrics_bands.py --topology DPAH
    python network_metrics_bands.py --csv <path> --metrics clustering modularity \
        gini_in_degree assortativity --include-static
"""
import os
import argparse
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (ScalarFormatter, AutoMinorLocator, MaxNLocator,
                               NullLocator)
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.transforms import blended_transform_factory

# ---- style (mirrors plot_network_properties.py / plots_lines.py) ---------------
cm = 1 / 2.54
FONT_SIZE = 8
line_params = {"data_line_width": 0.9, "axis_line_width": 0.8,
               "grid_line_width": 0.5, "tick_major_width": 0.8,
               "tick_minor_width": 0.6, "markersize": 3}

PLOT_COLORS = {
    'none_none': '#EE7733', 'random_none': '#0077BB',
    'biased_same': '#33BBEE', 'biased_diff': '#009988',
    'bridge_same': '#CC3311', 'bridge_diff': '#EE3377',
    'wtf_none': '#BBBBBB', 'node2vec_none': '#44BB99',
}
FRIENDLY_NAMES = {
    'none_none': 'Static', 'random_none': 'Random', 'biased_same': 'Local (Similar)',
    'biased_diff': 'Local (Opposite)', 'bridge_same': 'Bridge (Similar)',
    'bridge_diff': 'Bridge (Opposite)', 'wtf_none': 'WTF', 'node2vec_none': 'Node2Vec',
}
ALGO_ORDER = ['none_none', 'random_none', 'biased_same', 'biased_diff',
              'bridge_same', 'bridge_diff', 'wtf_none', 'node2vec_none']
TYPE_ORDER = ['DPAH', 'cl', 'Twitter', 'FB']
NETWORK_DISPLAY_NAMES = {'cl': 'CSF', 'DPAH': 'DPA', 'Twitter': 'Twitter', 'FB': 'FB'}
NETWORK_MARKERS = {'cl': 'o', 'DPAH': 's', 'Twitter': '^', 'FB': 'D'}

# (column, label, ylim). Directedness is topology-dependent, so the Gini label
# carries both readings; see the extractor header.
METRIC_SPECS = {
    'clustering':     ('Clustering', (0, 1)),
    'modularity':     ('Modularity', (0, 1)),
    'gini_in_degree': ('Gini (in-degree)', (0, 1)),
    'assortativity':  ('Assortativity', (-1, 1)),
}
DEFAULT_METRICS = ['clustering', 'modularity', 'gini_in_degree']
RESIDUAL_H = 0.5  # residual strip height, as a fraction of the metric panel above it
RESIDUAL_H_LANES = 1.0  # lanes need four baselines in the same strip, so more of it
LANE_FRAC = 0.42  # deviation of the panel's half-range, in lane-spacing units

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.normpath(os.path.join(HERE, "..", "..", "Output",
                                            "network_metrics_per_run.csv"))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "Figs", "Networks"))


def _norm(x):
    x = str(x).strip()
    return 'none' if x in ('', 'None', 'none', 'nan', 'NaN', '<NA>') else x.lower()


def load_data(path):
    df = pd.read_csv(path, keep_default_na=False,
                     dtype={'scenario': str, 'rewiring': str, 'type': str})
    for c in ['model_run', 'timestep'] + list(METRIC_SPECS):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['grouped'] = df['scenario'].map(_norm) + '_' + df['rewiring'].map(_norm)
    return df


def set_plot_style():
    plt.rcParams.update({
        'font.size': FONT_SIZE, 'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE, 'axes.linewidth': line_params["axis_line_width"],
        'figure.dpi': 300, 'savefig.dpi': 300, 'grid.alpha': 0.4,
        'grid.linestyle': '--', 'mathtext.default': 'regular',
        'axes.formatter.use_mathtext': True, 'axes.axisbelow': True,
    })


def configure_axis_style(ax, show_ylabel=True, show_xlabel=True):
    ax.grid(True, alpha=0.4, linestyle='--',
            linewidth=line_params["grid_line_width"], zorder=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(line_params["axis_line_width"])
        spine.set_zorder(100)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', direction='out', length=3,
                   width=line_params["tick_major_width"], colors='black', zorder=100,
                   bottom=True, top=False, left=True, right=False,
                   labelbottom=show_xlabel, labelleft=show_ylabel)
    ax.tick_params(axis='both', which='minor', direction='out', length=1.5,
                   width=line_params["tick_minor_width"], colors='black', zorder=100,
                   bottom=True, top=False, left=True, right=False,
                   labelbottom=False, labelleft=False)


def _band(sub, metric):
    """Median and interquartile band across runs, per timestep."""
    g = sub.groupby('timestep')[metric]
    out = g.quantile([0.25, 0.5, 0.75]).unstack()
    out.columns = ['q25', 'q50', 'q75']
    out['n'] = g.size()
    return out.reset_index().sort_values('timestep')


def _row_limits(df, metric, algos, pad=0.06):
    """Shared y-limits for one metric row, tight enough that the IQR is visible.

    The natural (0,1) limits render a 0.007-wide band as a sub-pixel line, which
    defeats the purpose of the figure whenever run-to-run spread is small.
    """
    v = df.loc[df['grouped'].isin(algos), metric].dropna()
    if v.empty:
        return METRIC_SPECS[metric][1]
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1e-3)
    return lo - pad * span, hi + pad * span


def _deviations(sub, metric):
    """Per-run deviation from the condition's own median trajectory.

    No monotone y-transform can enlarge a 0.005-wide band on an axis that must
    span 0.02-0.61: stretching the band stretches the between-condition
    differences by the same factor, which is what --autoscale already does.
    Subtracting the median is a change of variable instead, so every condition's
    spread centres on zero and shares one small axis while the medians keep the
    shared (0,1) axis in the row above.
    """
    out = sub.copy()
    out[metric] = sub[metric] - sub.groupby('timestep')[metric].transform('median')
    return out


def _residual_limits(df, metric, algos, types, hairlines, pad=0.08):
    """Symmetric limits for a residual strip, from the deviations actually drawn.

    Scaled per panel, not per row: WTF on Twitter has a final modularity IQR of
    0.163 against a 0.004 median elsewhere, so one shared scale would flatten 26
    of 28 strips back into the invisible band this row exists to fix. Each strip
    therefore carries its own tick labels, and --residual-lim forces a common
    scale when cross-panel comparison of the spread matters more.

    Hairlines draw every run, so the axis has to hold the full deviation range;
    without them only the quartile band is drawn.
    """
    dev = []
    for algo in algos:
        for topo in types:
            sub = df[(df['grouped'] == algo) & (df['type'] == topo)]
            if sub.empty or sub[metric].isna().all():
                continue
            d = _deviations(sub, metric)
            if d[metric].isna().all():
                continue
            if hairlines:
                dev.append(d[metric].abs().max())
            else:
                # Per timestep, matching the band that gets plotted; pooling over
                # timesteps would dilute it with the near-zero early ones.
                b = _band(d, metric)
                dev.append(max(b['q25'].abs().max(), b['q75'].abs().max()))
    dev = [v for v in dev if np.isfinite(v)]
    if not dev:
        return (-1e-3, 1e-3)
    m = max(dev) * (1 + pad) or 1e-3
    return (-m, m)


def make_figure(df, metrics, topology, out_path, hairlines=True, include_static=False,
                autoscale=False, residuals=False, residual_lim=None,
                residual_style='overlay'):
    algos = [a for a in ALGO_ORDER if (df['grouped'] == a).any()]
    if not include_static:
        algos = [a for a in algos if a != 'none_none']
    types = [t for t in TYPE_ORDER if t in df['type'].unique()]
    combined = topology == 'all'
    # Hairlines only make sense when a panel holds a single combination.
    hairlines = hairlines and not combined
    # One topology per panel is already unambiguous, so lanes would just be an
    # overlay strip with the axis relabelled.
    lanes = residuals and residual_style == 'lanes' and len(types) > 1

    res_h = RESIDUAL_H_LANES if lanes else RESIDUAL_H
    panel_h = 4 * (1 + res_h) if residuals else 4
    fig = plt.figure(figsize=(4 * len(algos) * cm, panel_h * len(metrics) * cm))
    gs = GridSpec(len(metrics), len(algos), figure=fig, top=0.94, bottom=0.12,
                  hspace=(0.30 if lanes else 0.45) if residuals else 0.25, wspace=0.15,
                  left=0.10, right=0.98)

    for r, metric in enumerate(metrics):
        label, ylim = METRIC_SPECS[metric]
        if autoscale:
            ylim = _row_limits(df, metric, algos)
        for c, algo in enumerate(algos):
            rlim = ((-residual_lim, residual_lim) if residual_lim else
                    _residual_limits(df, metric, [algo], types, hairlines))
            if residuals:
                # Nested so the strip sits tight under its own metric panel while
                # the pairs stay separated.
                inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[r, c],
                                                height_ratios=[1, res_h],
                                                hspace=0.12)
                ax = fig.add_subplot(inner[0])
                rax = fig.add_subplot(inner[1], sharex=ax)
            else:
                ax, rax = fig.add_subplot(gs[r, c]), None
            color = PLOT_COLORS.get(algo, '#FE6900')
            drawn_lanes = []

            for topo in types:
                sub = df[(df['grouped'] == algo) & (df['type'] == topo)]
                if sub.empty or sub[metric].isna().all():
                    continue
                b = _band(sub, metric)

                if hairlines and rax is None:
                    for _run, run_df in sub.groupby('model_run'):
                        run_df = run_df.sort_values('timestep')
                        ax.plot(run_df['timestep'], run_df[metric], color=color,
                                linewidth=0.35, alpha=0.10, zorder=1.5,
                                solid_capstyle='butt')

                ax.fill_between(b['timestep'], b['q25'], b['q75'], color=color,
                                alpha=0.22 if combined else 0.28, linewidth=0, zorder=2)
                ax.plot(b['timestep'], b['q50'], color=color,
                        linewidth=line_params["data_line_width"],
                        marker=NETWORK_MARKERS.get(topo, 'o') if combined else None,
                        markersize=line_params["markersize"], zorder=3,
                        label=NETWORK_DISPLAY_NAMES.get(topo, topo))

                if rax is not None:
                    d = _deviations(sub, metric)
                    # Lanes are indexed by the global topology order, not by what
                    # this panel happens to hold, so the lane a band sits in means
                    # the same thing in every panel and WTF's missing undirected
                    # topologies read as gaps rather than shifting the others up.
                    lane = types.index(topo) if lanes else 0
                    scale = LANE_FRAC / rlim[1] if lanes else 1.0
                    if hairlines:
                        for _run, run_df in d.groupby('model_run'):
                            run_df = run_df.sort_values('timestep')
                            rax.plot(run_df['timestep'], lane + run_df[metric] * scale,
                                     color=color, linewidth=0.35, alpha=0.10,
                                     zorder=1.5, solid_capstyle='butt')
                    rb = _band(d, metric)
                    rax.fill_between(rb['timestep'], lane + rb['q25'] * scale,
                                     lane + rb['q75'] * scale, color=color,
                                     alpha=0.22 if combined else 0.28, linewidth=0,
                                     zorder=2)
                    drawn_lanes.append(lane)

            is_bottom, is_left = r == len(metrics) - 1, c == 0
            configure_axis_style(ax, show_ylabel=is_left,
                                 show_xlabel=is_bottom and rax is None)
            ax.set_ylim(*ylim)
            if is_left:
                ax.set_ylabel(label, fontsize=FONT_SIZE, fontweight='bold')
            if r == 0:
                ax.set_title(FRIENDLY_NAMES.get(algo, algo), fontsize=FONT_SIZE,
                             fontweight='bold')
            if rax is not None:
                # Per-panel scales must be readable panel by panel, so the strip
                # states its own half-range instead of borrowing column 0's ticks.
                shared = residual_lim is not None
                configure_axis_style(rax, show_ylabel=is_left and (shared or lanes),
                                     show_xlabel=is_bottom)
                if lanes:
                    # Inverted, so lane 0 is the top one and matches TYPE_ORDER;
                    # the extra headroom keeps the scale label off the top band.
                    rax.set_ylim(len(types) - 0.5, -0.85)
                    rax.yaxis.set_minor_locator(NullLocator())
                    rax.set_yticks(range(len(types)))
                    rax.set_yticklabels(
                        [NETWORK_DISPLAY_NAMES.get(t, t) for t in types] if is_left
                        else [], fontsize=FONT_SIZE - 2)
                    at_left = blended_transform_factory(rax.transAxes, rax.transData)
                    for lane in sorted(set(drawn_lanes)):
                        rax.axhline(lane, color='0.35', linewidth=0.5, zorder=2.5)
                        rax.plot([0.035], [lane], transform=at_left, color=color,
                                 marker=NETWORK_MARKERS.get(types[lane], 'o'),
                                 markersize=2.2, zorder=4, clip_on=False)
                else:
                    rax.axhline(0, color='0.35', linewidth=0.5, zorder=2.5)
                    rax.set_ylim(*rlim)
                    rax.yaxis.set_major_locator(MaxNLocator(3, symmetric=True))
                if is_left:
                    # Outboard of the lane names, which occupy the tick labels.
                    rax.set_ylabel(r'$\Delta$', fontsize=FONT_SIZE)
                if not shared:
                    rax.text(0.98, 0.97, rf'$\pm${rlim[1]:.2g}', transform=rax.transAxes,
                             ha='right', va='top', fontsize=FONT_SIZE - 2, color='0.35')
            if is_bottom:
                bottom_ax = rax if rax is not None else ax
                bottom_ax.set_xlabel('Time, t')
                f = ScalarFormatter(useMathText=True)
                f.set_scientific(True)
                f.set_powerlimits((-2, 1))
                bottom_ax.xaxis.set_major_formatter(f)

    if combined:
        handles = [Line2D([0], [0], color='black', marker=NETWORK_MARKERS.get(t, 'o'),
                          markersize=line_params["markersize"],
                          linewidth=line_params["data_line_width"],
                          label=NETWORK_DISPLAY_NAMES.get(t, t)) for t in types]
        fig.legend(handles=handles, loc='upper center', ncol=len(types),
                   frameon=False, fontsize=FONT_SIZE, bbox_to_anchor=(0.5, 0.02))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(os.path.splitext(out_path)[0] + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def report(df, metrics):
    """Numbers for the caption: where run-to-run spread is and is not negligible."""
    print("\nIQR width at the final timestep (q75 - q25 across runs):")
    fin = df.loc[df.groupby(['grouped', 'type'])['timestep'].transform('max')
                 == df['timestep']]
    for metric in metrics:
        w = (fin.groupby(['grouped', 'type'])[metric]
                .agg(lambda s: s.quantile(0.75) - s.quantile(0.25)).dropna())
        if w.empty:
            continue
        top = w.sort_values(ascending=False).head(3)
        print(f"  {metric:15s} median {w.median():.4f}  max {w.max():.4f}")
        for (g_, t_), v in top.items():
            print(f"      {FRIENDLY_NAMES.get(g_, g_):18s} {NETWORK_DISPLAY_NAMES.get(t_, t_):8s} {v:.4f}")
    t0 = df[df['timestep'] == 0]
    if not t0.empty:
        print("\nt=0 spread (initial graphs are fixed by construction, so this is ~0):")
        for metric in metrics:
            s = t0.groupby(['grouped', 'type'])[metric].std().dropna()
            if not s.empty:
                print(f"  {metric:15s} max across-run SD {s.max():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--topology', default='all', choices=['all'] + TYPE_ORDER)
    ap.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS,
                    choices=list(METRIC_SPECS))
    ap.add_argument('--include-static', action='store_true')
    ap.add_argument('--no-hairlines', action='store_true')
    ap.add_argument('--autoscale', action='store_true',
                    help='fit each metric row to its data range instead of (0,1); '
                         'needed to see the band wherever run-to-run spread is small')
    ap.add_argument('--residuals', action='store_true',
                    help='add a half-height strip under each metric row showing each '
                         'run minus the median trajectory; keeps the shared (0,1) axis '
                         'for the medians and still shows the spread')
    ap.add_argument('--residual-style', default='overlay', choices=['overlay', 'lanes'],
                    help="'overlay' (default) stacks the topologies in one strip, so "
                         "the combined layout shows their envelope; 'lanes' gives each "
                         "topology its own baseline inside a taller strip")
    ap.add_argument('--residual-lim', type=float, default=None,
                    help='fix the residual strips at +/- this value instead of '
                         'scaling each metric row to its own deviations')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    set_plot_style()
    df = load_data(a.csv)
    if a.topology != 'all':
        df = df[df['type'] == a.topology]
        if df.empty:
            raise SystemExit(f"no rows for topology {a.topology}")

    suffix = 'combined' if a.topology == 'all' else a.topology
    if a.residuals:
        # A single-topology figure falls back to overlay, so don't label it lanes.
        suffix += '_residuals' + ('_lanes' if a.residual_style == 'lanes'
                                  and a.topology == 'all' else '')
    out = a.out or os.path.join(FIG_DIR,
                                f"network_metrics_bands_{suffix}_{date.today()}.pdf")
    make_figure(df, a.metrics, a.topology, out,
                hairlines=not a.no_hairlines, include_static=a.include_static,
                autoscale=a.autoscale, residuals=a.residuals,
                residual_lim=a.residual_lim, residual_style=a.residual_style)
    report(df, a.metrics)


if __name__ == '__main__':
    main()
