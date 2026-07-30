#!/usr/bin/env python3
"""
Run-level violins for reviewer R1.7, which asks for "distributions across runs ...
for final mean opinion, final polarization, convergence rate, and network metrics",
motivated by whether "relative algorithm rankings and Pareto-like comparisons"
survive run-to-run variation.

Each violin body is therefore a distribution of RUN-LEVEL quantities: one point per
model_run (90 per combo), NOT per-agent states. The per-agent view is a separate SI
figure (violin_main_results.py) answering a different question (R1.1: mean alignment
vs genuine consensus).

DATA: Output/per_run_summary.csv from Analysis/Stats/summarize_individual_csv.py,
one row per (type, scenario, rewiring, model_run) with
    speed_t95            convergence rate, same metric Fig 3 uses
    cooperativity        tail-window mean opinion = final mean opinion <a*>
    final_polarization   tail-window within-run agent dispersion <P*>
Re-run summarize_individual_csv.py if the campaign CSV changes; the 2026-07-27
horizon merge moved bridge_diff/FB from 0.2652 to 0.2228.

SPEED CAVEAT (surfaced deliberately, not hidden): calculate_t95_convergence_speed
compares the trajectory against 0.95*final_value, which is meaningless when the
final value is negative - such runs satisfy the test at t=0 and register speed ~1.
That is 50/90 wtf runs on DPA and 14/90 on Twitter. The metric is kept identical to
Fig 3's on purpose, so the artefact is MARKED on the figure (red rings) rather than
corrected, and named in the caption.

Usage:
    python violin_run_level.py                     # uses Output/per_run_summary.csv
    python violin_run_level.py --csv <path> [--out <path.pdf>] [--no-artefact-marks]

Style constants mirror violin_main_results.py / plots_lines.py /
convergence_vs_cooperation.py so the figure sits consistently beside Figs 2 and 3.
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

# ---- style (copied from violin_main_results.py) --------------------------------
cm = 1 / 2.54
FONT_SIZE = 11
line_params = {"axis_line_width": 1.2, "grid_line_width": 0.6,
               "tick_major_width": 1.2, "tick_minor_width": 0.8}

PLOT_COLORS = {
    'none_none': '#EE7733', 'random_none': '#0077BB',
    'biased_same': '#33BBEE', 'biased_diff': '#009988',
    'bridge_same': '#CC3311', 'bridge_diff': '#EE3377',
    'wtf_none': '#BBBBBB', 'node2vec_none': '#44BB99',
}
NETWORK_DISPLAY_NAMES = {'cl': 'CSF', 'DPAH': 'DPA', 'Twitter': 'Twitter', 'FB': 'FB'}
FRIENDLY_NAMES = {
    'none_none': 'static', 'random_none': 'random', 'biased_same': 'L-sim',
    'biased_diff': 'L-opp', 'bridge_same': 'B-sim', 'bridge_diff': 'B-opp',
    'wtf_none': 'wtf', 'node2vec_none': 'node2vec',
}
ALGO_ORDER = ['none_none', 'random_none', 'biased_same', 'biased_diff',
              'bridge_same', 'bridge_diff', 'wtf_none', 'node2vec_none']
TYPE_ORDER = ['DPAH', 'cl', 'Twitter', 'FB']

# (column, axis label, y-limits, reference line or None)
# The reference lines are the consensus criteria from the Methods definition;
# they let the reader see directly that no condition attains either one.
# Axis labels break over two lines: at true print size a single-line label is taller
# than the axis and rows 1 and 2 collide at the spine.
METRICS = [
    ('cooperativity',      'Final mean opinion' '\n' r'$\langle a^* \rangle$', (-1.05, 1.05), 0.8),
    ('final_polarization', 'Final polarization' '\n' r'$\langle P^* \rangle$', (0.0, 1.05), 0.3),
    ('speed_t95',          'Convergence rate' '\n' r'($t_{95}$)',              (0.0, 1.05), None),
]
# Main text carries mean opinion + polarization only; convergence rate goes to the SI
# with the per-agent violins. If it moves, Fig 3's caption must point at the SI row,
# since Fig 3 uses the same t95 metric and would otherwise carry the artefact silently.
METRIC_SETS = {'main': ['cooperativity', 'final_polarization'],
               'all': [m[0] for m in METRICS]}

# Author at the true printed width so nothing is rescaled on inclusion:
# oup-authoring-template [contemporary,large] has \textwidth 488.5pt, column 235.25pt.
TEXTWIDTH = 488.5 / 72 * 2.54 * cm
ROW_HEIGHT = 5.6 * cm

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.normpath(os.path.join(HERE, "..", "..", "Output",
                                            "per_run_summary.csv"))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "Figs", "Distributions"))


def _norm(x):
    x = str(x).strip()
    return 'none' if x in ('', 'None', 'none', 'nan', 'NaN', '<NA>') else x.lower()


def load_data(path):
    df = pd.read_csv(path, keep_default_na=False,
                     dtype={'scenario': str, 'rewiring': str, 'type': str})
    for c in ('speed_t95', 'cooperativity', 'final_polarization', 'model_run'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['cooperativity'])
    df['grouped'] = df['scenario'].map(_norm) + '_' + df['rewiring'].map(_norm)
    # negative-target t95 artefact: trajectory clears 0.95*final at t=0
    df['speed_artefact'] = (df['cooperativity'] < 0) & (df['speed_t95'] > 0.99)
    return df


def _style_axis(ax, ylim):
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, alpha=0.4, linestyle='--',
                  linewidth=line_params["grid_line_width"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(line_params["axis_line_width"])
    ax.tick_params(axis='both', which='major', width=line_params["tick_major_width"])


def make_figure(df, out_path, mark_artefact=True, metric_set='main'):
    metrics = [m for m in METRICS if m[0] in METRIC_SETS[metric_set]]
    types_present = [t for t in TYPE_ORDER if t in df['type'].unique()]
    nrows, ncols = len(metrics), len(types_present)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(TEXTWIDTH, ROW_HEIGHT * nrows),
                             sharey='row')
    axes = np.atleast_2d(axes)
    rng = np.random.default_rng(0)
    any_artefact = False

    for r, (col, ylab, ylim, refline) in enumerate(metrics):
        for c, type_ in enumerate(types_present):
            ax = axes[r, c]
            sub = df[df['type'] == type_]
            algos = [g for g in ALGO_ORDER if (sub['grouped'] == g).any()]
            data = [sub.loc[sub['grouped'] == g, col].dropna().to_numpy() for g in algos]
            pos = np.arange(len(algos))

            if refline is not None:
                ax.axhline(refline, color='0.35', lw=0.9, ls=':', alpha=0.9, zorder=1)
            if ylim[0] < 0:
                ax.axhline(0.0, color='0.5', lw=0.8, ls='--', alpha=0.8, zorder=1)

            parts = ax.violinplot(data, positions=pos, widths=0.85,
                                  showmeans=False, showmedians=False,
                                  showextrema=False)
            for body, g, arr in zip(parts['bodies'], algos, data):
                body.set_facecolor(PLOT_COLORS[g])
                body.set_edgecolor('black')
                body.set_linewidth(0.6)
                body.set_alpha(0.85)
                # clip the KDE spill past the metric's natural bounds
                v = body.get_paths()[0].vertices
                v[:, 1] = np.clip(v[:, 1], ylim[0], ylim[1])

            # one point per run (n=90): this is the quantity R1.7 asked to see
            for p, g in zip(pos, algos):
                s = sub[sub['grouped'] == g]
                vals = s[col].to_numpy()
                jit = rng.uniform(-0.13, 0.13, size=len(vals))
                if mark_artefact and col == 'speed_t95':
                    bad = s['speed_artefact'].to_numpy()
                    ax.scatter(p + jit[~bad], vals[~bad], s=5, color='black',
                               alpha=0.40, linewidth=0, zorder=3.5)
                    if bad.any():
                        any_artefact = True
                        ax.scatter(p + jit[bad], vals[bad], s=16,
                                   facecolor='none', edgecolor='#CC3311',
                                   linewidth=0.8, zorder=3.6)
                else:
                    ax.scatter(p + jit, vals, s=5, color='black', alpha=0.40,
                               linewidth=0, zorder=3.5)

            # median dot + IQR bar
            for p, arr in zip(pos, data):
                if not len(arr):
                    continue
                q1, med, q3 = np.percentile(arr, [25, 50, 75])
                ax.vlines(p, q1, q3, color='black', lw=2.4, zorder=4)
                ax.scatter(p, med, s=14, color='white', edgecolor='black',
                           linewidth=0.7, zorder=5)

            ax.set_xticks(pos)
            ax.set_xlim(-0.6, len(algos) - 0.4)
            if r == nrows - 1:
                # 8 labels across a 4.3 cm panel: at 45 deg they overlap at print
                # size (a label's horizontal footprint exceeds the tick spacing),
                # at 90 deg the footprint is one font height.
                ax.set_xticklabels([FRIENDLY_NAMES[g] for g in algos],
                                   rotation=90, ha='center',
                                   fontsize=FONT_SIZE - 2)
            else:
                ax.set_xticklabels([])
            if r == 0:
                ax.set_title(NETWORK_DISPLAY_NAMES.get(type_, type_),
                             fontsize=FONT_SIZE)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=FONT_SIZE - 1)
            _style_axis(ax, ylim)

    handles = [
        Line2D([], [], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='none', alpha=0.5, markersize=3.5,
               label='individual run ($n=90$)'),
        Line2D([], [], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=5, label='median'),
        Line2D([], [], color='black', lw=2.4, label='IQR'),
        Line2D([], [], color='0.35', lw=0.9, ls=':', label='consensus criterion'),
    ]
    if any_artefact and mark_artefact:
        handles.append(Line2D([], [], marker='o', color='none', markerfacecolor='none',
                              markeredgecolor='#CC3311', markersize=5,
                              label=r'$t_{95}$ undefined (negative target)'))
    # legend sits INSIDE the canvas: anchored below the axes it widened the tight
    # bbox to 21.1 cm, so inclusion at \textwidth rescaled the whole figure by 0.82
    # and the 11 pt labels printed at 9 pt.
    legend_band = 0.9 * cm / (ROW_HEIGHT * nrows)
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=FONT_SIZE - 2,
               bbox_to_anchor=(0.5, legend_band * 0.12))

    fig.tight_layout(rect=(0, legend_band, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    png = os.path.splitext(out_path)[0] + ".png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"saved:\n  {out_path}\n  {png}")


def print_ranking_report(df):
    """Rank stability across runs: the headline the figure has to carry."""
    print("\nRun-level spread by topology (SD across runs of the per-run mean opinion):")
    g = df.groupby(['type', 'grouped'])['cooperativity'].agg(['mean', 'std', 'min', 'max'])
    for t in [x for x in TYPE_ORDER if x in df['type'].unique()]:
        s = g.loc[t].sort_values('mean', ascending=False)
        worst = s['std'].idxmax()
        print(f"  {NETWORK_DISPLAY_NAMES.get(t, t):8s} max across-run SD = {s['std'].max():.3f} "
              f"({FRIENDLY_NAMES[worst]}); median across-run SD = {s['std'].median():.3f}")
    n_bad = int(df['speed_artefact'].sum())
    print(f"\n  t95 artefact runs (negative target): {n_bad} of {len(df)} "
          f"({100 * n_bad / len(df):.1f}%)")
    ab = df[df['speed_artefact']].groupby(['type', 'grouped']).size()
    for (t, g_), n in ab.items():
        print(f"    {NETWORK_DISPLAY_NAMES.get(t, t):8s} {FRIENDLY_NAMES[g_]:9s} {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-artefact-marks", action="store_true")
    ap.add_argument("--metrics", default="main", choices=list(METRIC_SETS),
                    help="'main': mean opinion + polarization (main text); "
                         "'all': adds the convergence rate row (SI)")
    args = ap.parse_args()

    df = load_data(args.csv)
    out = args.out or os.path.join(
        FIG_DIR, f"violin_run_level_{args.metrics}_{date.today():%Y%m%d}.pdf")
    print(f"rows={len(df):,}  types={sorted(df['type'].unique())}  "
          f"runs/combo={df.groupby(['type', 'grouped'])['model_run'].nunique().median():.0f}")
    make_figure(df, out, mark_artefact=not args.no_artefact_marks,
                metric_set=args.metrics)
    print_ranking_report(df)


if __name__ == "__main__":
    main()
