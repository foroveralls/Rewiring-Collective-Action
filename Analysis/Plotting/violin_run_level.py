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
    speed_t95            t95 convergence rate (NOT the metric Fig 3 uses -- see below)
    cooperativity        tail-window mean opinion = final mean opinion <a*>
    final_polarization   tail-window within-run agent dispersion <P*>
Re-run summarize_individual_csv.py if the campaign CSV changes; the 2026-07-27
horizon merge moved bridge_diff/FB from 0.2652 to 0.2228.

WHICH CONVERGENCE METRIC (decided 2026-07-31, read before promoting this figure):
Fig. 3 uses **inflection**, not t95. If the convergence row goes to the main text it
must be inflection too, or the paper carries two different convergence metrics in
two adjacent main figures. Concretely they disagree: on t95 wtf/DPA sits at the TOP
of the convergence axis (median ~0.99, red-ringed below), on inflection it ranks
24th of 30, and manuscript L185 says WTF converges slower than most scenarios. So:

    --inflection-csv Output/per_run_inflection.csv --metrics main+inflection

joins the per-run inflection rates from Analysis/extract_inflection_lowmem.py and
plots those instead. That file must be built at FULL resolution (per_run_summary.csv
is downsampled at t%100, where find_inflection's 5000<i<20000 window is unreachable).

REGWIN, and why --regwin exists: the published rate regresses over regwin=10, i.e.
21 steps out of 160,000. That is fine on the ensemble mean (already averaged over 90
runs) but on a SINGLE run it is largely noise -- the smoke test found 23% of per-run
rates negative and a per-run IQR 2x the entire 30-condition ensemble spread. A violin
draws the whole distribution, so plotting that would publish the estimator's noise
floor captioned as run-to-run variation. Check the REGWIN SPREAD DIAGNOSTIC block in
per_run_inflection_META.txt first; if the spread collapses with width, pass e.g.
--regwin 1000 for the per-run estimate and SAY SO in the caption. Fig 3's ENSEMBLE
rate stays at regwin=10 regardless.

SPEED CAVEAT for the t95 row (surfaced deliberately, not hidden):
calculate_t95_convergence_speed compares the trajectory against 0.95*final_value,
which is meaningless when the final value is negative - such runs satisfy the test at
t=0 and register speed ~1. That is 50/90 wtf runs on DPA and 14/90 on Twitter. The
artefact is MARKED on the figure (red rings) rather than corrected. Moving to
inflection removes the artefact and the rings entirely.

SUPERSEDED FOR S6 (round-2 R1b.3, 2026-10-19): Fig. 3 now plots t_{a>0}, the first step
at which the ensemble mean opinion is positive, and the inflection rate is no longer
reported. SI Fig. S6 is therefore `--metrics t_zero --tzero-csv Output/per_run_t_zero.csv`
(from `Analysis/Stats/bootstrap_convergence_A1.py --perrun`): each run's own first step
with a > 0, on a log axis (2.6k to 155k steps), with the ensemble value and its 95%
bootstrap interval from Output/convergence_bootstrap_A1.csv beside each violin. The two
are different quantities: the run spread is not uncertainty in the Fig. 3 value, and the
ensemble mean waits for slow runs, so on DPA per-run medians sit 1-5k steps earlier.
Runs that never turn positive (15 of 2700) are omitted and must be counted in the caption.

Usage:
    python violin_run_level.py                     # uses Output/per_run_summary.csv
    python violin_run_level.py --metrics t_zero --tzero-csv ../../Output/per_run_t_zero.csv
    python violin_run_level.py --csv <path> [--out <path.pdf>] [--no-artefact-marks]
    python violin_run_level.py --metrics main+inflection \
        --inflection-csv ../../Output/per_run_inflection.csv [--regwin 1000]

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
# y-limits None = resolve from the data (used by the inflection row, whose units are
# not [0,1] and whose per-run values can go negative).
INFLECTION_COL = 'speed_inflection'
T_ZERO_COL = 't_zero'
METRICS = [
    # Per-run labels are $a^*$ / $P^*$, not $\langle\cdot\rangle$: manuscript L87 reserves
    # angle brackets for the ensemble average over runs, and these rows plot one value per run.
    ('cooperativity',      'Final mean opinion' '\n' r'$a^*$', (-1.05, 1.05), 0.8),
    ('final_polarization', 'Final polarization' '\n' r'$P^*$', (0.0, 1.05), 0.3),
    ('speed_t95',          'Convergence rate' '\n' r'($t_{95}$)',              (0.0, 1.05), None),
    # No scaling is applied to the inflection values anywhere in this script, so the
    # axis carries the raw rate. The label claimed $\times10^{-3}$ until 2026-07-31,
    # which understated every plotted rate by 1000x; do not reinstate it without
    # actually scaling the data.
    (INFLECTION_COL,       'Convergence rate' '\n' r'(inflection)',                 None, None),
    # time, not rate: Fig. 3's x-axis since round 2. Per-run, so no angle brackets.
    (T_ZERO_COL,           'Steps until' '\n' r'$a > 0$',                          None, None),
]
# rows drawn on log10 values (KDE included); tick labels are put back in steps
LOG_ROWS = {T_ZERO_COL}
# 'main'           mean opinion + polarization (what was rendered 2026-07-29)
# 'main+inflection' adds the convergence row on Fig 3's OWN metric -- the main-text
#                  candidate agreed 2026-07-31; needs --inflection-csv
# 'all'            legacy t95 version; SI only, and only if the t95 row is wanted at
#                  all. Do not put this in the main text beside an inflection Fig 3.
METRIC_SETS = {'main': ['cooperativity', 'final_polarization'],
               'main+inflection': ['cooperativity', 'final_polarization', INFLECTION_COL],
               # single convergence-rate row for SI Fig. S6 (decision 2026-08-06: the
               # main-text figure stays 2-row, the rate row goes to the SI on inflection)
               'inflection': [INFLECTION_COL],
               # SI Fig. S6 from round 2 (R1b.3): replaces the inflection row
               't_zero': [T_ZERO_COL],
               'all': ['cooperativity', 'final_polarization', 'speed_t95']}

# Author at the true printed width so nothing is rescaled on inclusion:
# oup-authoring-template [contemporary,large] has \textwidth 488.5pt, column 235.25pt.
TEXTWIDTH = 488.5 / 72 * 2.54 * cm
ROW_HEIGHT = 5.6 * cm

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.normpath(os.path.join(HERE, "..", "..", "Output",
                                            "per_run_summary.csv"))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "Figs", "Distributions"))
DEFAULT_ENSEMBLE_CSV = os.path.normpath(os.path.join(HERE, "..", "..", "Output",
                                                     "convergence_bootstrap_A1.csv"))


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


def join_inflection(df, path, regwin=10):
    """Attach per-run inflection rates from Analysis/extract_inflection_lowmem.py.

    Keyed on (type, scenario, rewiring, model_run) after the same _norm mapping the
    main frame uses, so the two files' 'None'/'none' conventions cannot silently
    mismatch. `regwin` selects speed_inflection_rw<N>; 10 is the published window and
    also lives in the plain `speed_inflection` column.

    Returns (df, report) with report describing the join and the NaN count, both of
    which have to be disclosed in the caption.
    """
    inf = pd.read_csv(path, keep_default_na=False,
                      dtype={'scenario': str, 'rewiring': str, 'type': str})
    col = INFLECTION_COL if regwin == 10 else f'speed_inflection_rw{regwin}'
    if col not in inf.columns:
        avail = sorted(c for c in inf.columns if c.startswith('speed_inflection'))
        raise SystemExit(f"{path} has no '{col}' column (has: {avail}).\n"
                         f"Re-run extract_inflection_lowmem.py with --regwins including {regwin}.")
    inf['model_run'] = pd.to_numeric(inf['model_run'], errors='coerce')
    inf[col] = pd.to_numeric(inf[col], errors='coerce')
    for f in (inf, df):
        f['_k'] = (f['type'].map(_norm) + '|' + f['scenario'].map(_norm) + '|'
                   + f['rewiring'].map(_norm) + '|' + f['model_run'].astype('Int64').astype(str))

    n_nan = int(inf[col].isna().sum())
    keep = inf[['_k', col]].rename(columns={col: INFLECTION_COL})
    before = len(df)
    df = df.merge(keep, on='_k', how='left')
    matched = int(df[INFLECTION_COL].notna().sum())
    df = df.drop(columns='_k')

    report = (f"inflection join: {path}\n"
              f"  column used        : {col}"
              + ("  (published window)" if regwin == 10 else
                 f"  (WIDENED per-run window; state this in the caption)") + "\n"
              f"  per-run rows in file: {len(inf)}\n"
              f"  NaN in file (no inflection found): {n_nan} "
              f"({100.0*n_nan/max(1,len(inf)):.2f}%) -- disclose this count\n"
              f"  matched onto {matched} of {before} run rows "
              f"({100.0*matched/max(1,before):.1f}%)")
    if matched == 0:
        raise SystemExit(report + "\n\nNo rows matched -- check that both files come "
                                  "from the same campaign.")
    return df, report


def _key(f):
    return (f['type'].map(_norm) + '|' + f['scenario'].map(_norm) + '|'
            + f['rewiring'].map(_norm) + '|' + f['model_run'].astype('Int64').astype(str))


def join_t_zero(df, path):
    """Attach per-run t_zero (NaN = never positive) from bootstrap_convergence_A1.py --perrun."""
    tz = pd.read_csv(path, keep_default_na=False, na_values=[''],
                     dtype={'scenario': str, 'rewiring': str, 'type': str})
    tz['_k'] = _key(tz)
    df = df.assign(_k=_key(df)).merge(tz[['_k', T_ZERO_COL]], on='_k', how='left')
    matched = int(df['_k'].isin(tz['_k']).sum())
    report = (f"t_zero join: {path}\n"
              f"  matched {matched} of {len(df)} run rows; "
              f"{int(df[T_ZERO_COL].isna().sum())} never positive -- disclose these")
    if matched != len(df):
        raise SystemExit(report + "\n\nUnmatched runs -- check both files come from the same campaign.")
    return df.drop(columns='_k'), report


def load_ensemble(path):
    """Ensemble t_zero and 95% bootstrap interval per (type, grouped), as in Fig. 3."""
    e = pd.read_csv(path, keep_default_na=False, na_values=[''],
                    dtype={'scenario': str, 'rewiring': str, 'type': str})
    e['grouped'] = e['scenario'].map(_norm) + '_' + e['rewiring'].map(_norm)
    return e.set_index(['type', 'grouped'])[['t_zero', 't_zero_lo', 't_zero_hi', 't_zero_never']]


def _resolve_ylim(ylim, series):
    """Data-driven limits for rows declared with ylim=None (the inflection row)."""
    if ylim is not None:
        return ylim
    v = pd.to_numeric(series, errors='coerce').dropna().to_numpy()
    if not v.size:
        return (0.0, 1.0)
    lo, hi = float(v.min()), float(v.max())
    lo = min(lo, 0.0)   # keep zero in frame: a negative rate is a real signal here
    pad = (hi - lo) * 0.06 or 0.1
    return (lo - pad, hi + pad)


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


def make_figure(df, out_path, mark_artefact=True, metric_set='main', ensemble=None):
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
        # resolved once per row, not per panel: sharey='row' means all four panels
        # must agree, and a per-panel range would make topologies look comparable
        # when they are not
        log = col in LOG_ROWS
        tf = np.log10 if log else (lambda x: x)
        if log:
            v = tf(df[col].dropna().to_numpy())
            if ensemble is not None:
                v = np.concatenate([v, tf(ensemble['t_zero_hi'].dropna().to_numpy())])
            ylim = (v.min() - 0.05, v.max() + 0.08)
        else:
            ylim = _resolve_ylim(ylim, df[col] if col in df.columns else pd.Series(dtype=float))
        for c, type_ in enumerate(types_present):
            ax = axes[r, c]
            sub = df[df['type'] == type_]
            algos = [g for g in ALGO_ORDER if (sub['grouped'] == g).any()]
            data = [tf(sub.loc[sub['grouped'] == g, col].dropna().to_numpy()) for g in algos]
            pos = np.arange(len(algos))

            if refline is not None:
                ax.axhline(refline, color='0.35', lw=0.9, ls=':', alpha=0.9, zorder=1)
            if ylim[0] < 0:
                ax.axhline(0.0, color='0.5', lw=0.8, ls='--', alpha=0.8, zorder=1)

            # A condition can be empty or degenerate for a given metric -- e.g. a
            # partial per-run inflection file, or every run failing find_inflection.
            # gaussian_kde needs >=2 points and non-zero variance, so those are
            # skipped here rather than crashing; their run dots, median and IQR are
            # still drawn below, so the panel shows what data exists.
            drawable = [i for i, a in enumerate(data) if a.size >= 2 and np.ptp(a) > 0]
            if drawable:
                parts = ax.violinplot([data[i] for i in drawable],
                                      positions=pos[drawable], widths=0.85,
                                      showmeans=False, showmedians=False,
                                      showextrema=False)
                for body, i in zip(parts['bodies'], drawable):
                    body.set_facecolor(PLOT_COLORS[algos[i]])
                    body.set_edgecolor('black')
                    body.set_linewidth(0.6)
                    body.set_alpha(0.85)
                    # clip the KDE spill past the metric's natural bounds
                    v = body.get_paths()[0].vertices
                    v[:, 1] = np.clip(v[:, 1], ylim[0], ylim[1])
            missing = [algos[i] for i in range(len(algos)) if i not in drawable]
            if missing:
                print(f"  note: no violin body for {type_} "
                      f"{[FRIENDLY_NAMES[g] for g in missing]} on '{col}' "
                      f"(fewer than 2 distinct values)")

            # one point per run (n=90): this is the quantity R1.7 asked to see
            for p, g in zip(pos, algos):
                s = sub[sub['grouped'] == g]
                vals = tf(s[col].to_numpy())
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

            # Fig. 3's ensemble value beside each violin, offset so it never hides the
            # run median: the gap between the two is part of what the figure shows
            if ensemble is not None and log:
                for p, g in zip(pos, algos):
                    if (type_, g) not in ensemble.index:
                        continue
                    e = ensemble.loc[(type_, g)]
                    x = p + 0.36
                    # open ended where >2.5% of resamples never turn positive (DPA WTF)
                    open_end = e['t_zero_never'] > 0.025
                    hi = ylim[1] - 0.02 if open_end else tf(e['t_zero_hi'])
                    ax.vlines(x, tf(e['t_zero_lo']), hi, color='black', lw=0.9, zorder=4)
                    if open_end:
                        ax.scatter(x, hi, marker='^', s=12, color='black', linewidth=0, zorder=4)
                    ax.scatter(x, tf(e['t_zero']), marker='D', s=11, color='black',
                               linewidth=0, zorder=5)

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
            if log:
                ticks = [t for t in (3e3, 1e4, 3e4, 1e5) if ylim[0] <= np.log10(t) <= ylim[1]]
                ax.set_yticks(np.log10(ticks))
                ax.set_yticklabels([f'{t / 1e3:g}k' for t in ticks])

    handles = [
        Line2D([], [], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='none', alpha=0.5, markersize=3.5,
               label='individual run ($n=90$)'),
        Line2D([], [], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='black', markersize=5, label='median'),
        Line2D([], [], color='black', lw=2.4, label='IQR'),
    ]
    # only rows with a criterion refline earn the legend entry (the inflection-only
    # SI figure has none, and a dangling legend key would puzzle the reader)
    if any(m[3] is not None for m in metrics):
        handles.append(Line2D([], [], color='0.35', lw=0.9, ls=':',
                              label='consensus criterion'))
    if ensemble is not None and any(m[0] in LOG_ROWS for m in metrics):
        handles.append(Line2D([], [], marker='D', color='black', lw=0.9, markersize=3.5,
                              label=r'ensemble $t_{a>0}$, 95% CI (Fig. 3)'))
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


def print_inflection_report(df, regwin):
    """Per-run convergence dispersion, for the caption and for the fallback in which
    the number is reported rather than drawn."""
    if INFLECTION_COL not in df.columns:
        return
    v = df[INFLECTION_COL]
    n_ok = int(v.notna().sum())
    neg = int((v < 0).sum())
    print(f"\nPer-run inflection rate (regwin={regwin}), n={n_ok} of {len(df)}:")
    print(f"  negative rates: {neg} ({100.0*neg/max(1,n_ok):.1f}%)"
          + ("   <-- large fractions indicate estimator noise, not slow convergence"
             if neg > 0.05 * max(1, n_ok) else ""))
    iqrs, meds = [], []
    for (t, g), s in df.groupby(['type', 'grouped']):
        a = s[INFLECTION_COL].dropna().to_numpy()
        if a.size >= 10:
            iqrs.append(np.percentile(a, 75) - np.percentile(a, 25))
            meds.append(np.median(a))
    if iqrs:
        print(f"  median per-condition IQR: {np.median(iqrs):.5f} "
              f"(range {min(iqrs):.5f}-{max(iqrs):.5f})")
        if len(meds) >= 2:
            spread = max(meds) - min(meds)
            print(f"  spread of per-condition medians: {spread:.5f} "
                  f"over {len(meds)} conditions")
            if spread > 0:
                print(f"  ratio IQR / between-condition spread: "
                      f"{np.median(iqrs)/spread:.2f}x"
                      "   <-- >1 means one condition's runs vary more "
                      "than the algorithms differ from each other")
        else:
            print(f"  (only {len(meds)} condition in this file -- the "
                  f"between-condition ratio needs the full campaign)")


def print_t_zero_report(df, ensemble):
    """Caption numbers: never-positive runs, run spread vs ensemble CI, median vs ensemble."""
    from scipy.stats import spearmanr
    g = df.groupby(['type', 'grouped'])[T_ZERO_COL]
    never = g.apply(lambda v: int(v.isna().sum()))
    print(f"\nPer-run t_zero: {int(df[T_ZERO_COL].isna().sum())} of {len(df)} runs never positive:")
    for (t, g_), n in never[never > 0].items():
        print(f"    {NETWORK_DISPLAY_NAMES.get(t, t):8s} {FRIENDLY_NAMES[g_]:9s} {n}")
    q = g.quantile([0.25, 0.5, 0.75]).unstack().join(ensemble)
    iqr, ciw = q[0.75] - q[0.25], q['t_zero_hi'] - q['t_zero_lo']
    print(f"  per-run range {df[T_ZERO_COL].min():.0f}-{df[T_ZERO_COL].max():.0f} steps")
    print(f"  median IQR / median ensemble 95% CI width: {iqr.median() / ciw.median():.2f}")
    print(f"  Spearman(per-run median, ensemble t_zero), 30 conditions: "
          f"{spearmanr(q[0.5], q['t_zero']).correlation:.2f}")
    d = q['t_zero'] - q[0.5]
    print("  ensemble minus per-run median (steps), by topology: "
          + ", ".join(f"{NETWORK_DISPLAY_NAMES.get(t, t)} {d.loc[t].min():.0f} to {d.loc[t].max():.0f}"
                      for t in TYPE_ORDER))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-artefact-marks", action="store_true")
    ap.add_argument("--metrics", default="main", choices=list(METRIC_SETS),
                    help="'main': mean opinion + polarization (main text); "
                         "'main+inflection': adds the convergence row on Fig 3's own "
                         "metric (needs --inflection-csv); "
                         "'all': legacy t95 convergence row (SI only)")
    ap.add_argument("--inflection-csv", default=None,
                    help="Output/per_run_inflection.csv from "
                         "Analysis/extract_inflection_lowmem.py (FULL resolution)")
    ap.add_argument("--regwin", type=int, default=10,
                    help="which regression half-width to plot from the inflection file. "
                         "10 = published. A wider window is defensible for the PER-RUN "
                         "estimate (a single run is not pre-averaged like the ensemble "
                         "mean) but MUST be stated in the caption. Read the REGWIN SPREAD "
                         "DIAGNOSTIC in per_run_inflection_META.txt before choosing")
    ap.add_argument("--tzero-csv", default=None,
                    help="Output/per_run_t_zero.csv from Analysis/Stats/bootstrap_convergence_A1.py "
                         "--perrun; needed by --metrics t_zero (SI Fig. S6)")
    ap.add_argument("--ensemble-csv", default=DEFAULT_ENSEMBLE_CSV,
                    help="ensemble t_zero with 95%% CIs drawn beside the t_zero violins")
    args = ap.parse_args()

    needs_inf = INFLECTION_COL in METRIC_SETS[args.metrics]
    if needs_inf and not args.inflection_csv:
        ap.error(f"--metrics {args.metrics} needs --inflection-csv "
                 f"(build it with Analysis/extract_inflection_lowmem.py)")
    if args.inflection_csv and not needs_inf:
        print("note: --inflection-csv given but the selected --metrics has no "
              "inflection row; joining anyway so the report prints")

    needs_tz = T_ZERO_COL in METRIC_SETS[args.metrics]
    if needs_tz and not args.tzero_csv:
        ap.error(f"--metrics {args.metrics} needs --tzero-csv")

    df = load_data(args.csv)
    ensemble = None
    if needs_tz:
        df, report = join_t_zero(df, args.tzero_csv)
        print(report)
        ensemble = load_ensemble(args.ensemble_csv)
    if args.inflection_csv:
        df, report = join_inflection(df, args.inflection_csv, regwin=args.regwin)
        print(report)

    suffix = args.metrics + (f"_rw{args.regwin}" if needs_inf and args.regwin != 10 else "")
    out = args.out or os.path.join(
        FIG_DIR, f"violin_run_level_{suffix}_{date.today():%Y%m%d}.pdf")
    print(f"rows={len(df):,}  types={sorted(df['type'].unique())}  "
          f"runs/combo={df.groupby(['type', 'grouped'])['model_run'].nunique().median():.0f}")
    make_figure(df, out, mark_artefact=not args.no_artefact_marks,
                metric_set=args.metrics, ensemble=ensemble)
    print_ranking_report(df)
    print_inflection_report(df, args.regwin)
    if needs_tz:
        print_t_zero_report(df, ensemble)


if __name__ == "__main__":
    main()
