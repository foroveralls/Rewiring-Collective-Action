#!/usr/bin/env python3
"""
Compact main-text figure for the external field phi (reviewer R1.2): the reviewer
asks that the dependence on the imposed pro-consensus field be made central, since
"some rewiring schemes may simply allow the imposed field to propagate more
efficiently". The figure makes that case in the smallest honest space.

    Panel A  steady state mean opinion <a*> against phi, one line per algorithm.
             Every algorithm is opposed at phi = 0; the rewiring rule sets the field
             strength at which the population flips and the level it then reaches.
    Panel B  threshold-reach map: phi* (field needed to flip the population, with a
             bootstrap CI) against <a*> at the default phi, with crosshairs at the
             static baseline so "worse than no rewiring" is readable directly.

DATA: the corrected political-climate sweep (steady states only), columns
    state,state_std,politicalClimate,rewiring,mode,topology
Default is the `tay` sweep, which was verified on 2026-07-27 to run at rho = 0.10
(see claude_stuff/Review/si_heatmap_regeneration_2026-07-27.md section 2). The older `lou`
and `dyr` sweeps ran at rho = 0 and must NOT be used here. As with the other phi
scripts, PHI_SWEEP_CSV overrides the default path.

Usage:
    python external_field_response.py                    # two-panel, \\textwidth
    python external_field_response.py --single           # panel A alone, 0.55\\textwidth
    python external_field_response.py --topologies directed   # DPA + Twitter only
    PHI_SWEEP_CSV=<path> python external_field_response.py

Figures are drawn at their final printed size and must be included WITHOUT scaling
(width=\\textwidth for the default layout), so the 8pt labels stay 8pt on the page.
A companion `_stats.csv` is written next to the figure holding every number the
manuscript text quotes from it.

Style constants mirror convergence_vs_cooperation.py / violin_main_results.py so the
figure is visually consistent with Figs 2 and 3.
"""
import os
import argparse
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- style (copied from convergence_vs_cooperation.py / violin_main_results.py) ----
cm = 1 / 2.54
FONT_SIZE = 8
TEXTWIDTH = 17.2 * cm          # oup-authoring-template [contemporary,large]: 488.5pt

FRIENDLY_COLORS = {
    'static': '#EE7733', 'random': '#0077BB', 'L-sim': '#33BBEE',
    'L-opp': '#009988', 'B-sim': '#CC3311', 'B-opp': '#EE3377',
    'wtf': '#BBBBBB', 'node2vec': '#44BB99',
}
# (mode, rewiring) as stored in the sweep CSV -> house short name
FRIENDLY_NAMES = {
    ('none', 'none'): 'static', ('random', 'none'): 'random',
    ('biased', 'same'): 'L-sim', ('biased', 'diff'): 'L-opp',
    ('bridge', 'same'): 'B-sim', ('bridge', 'diff'): 'B-opp',
    ('wtf', 'none'): 'wtf', ('node2vec', 'none'): 'node2vec',
}
ALGO_ORDER = ['L-opp', 'B-opp', 'random', 'L-sim', 'B-sim', 'static', 'node2vec', 'wtf']
# Interquartile bands for three series only (eight would be mud): the two families
# whose contrast the panel is about, plus wtf, whose ensemble mean is not a stable
# summary. wtf on DPA is bimodal, so its mean is sample-dependent - this sweep's
# 30 runs give +0.31 at the default field where the 90-run main campaign gives
# -0.23. The band shows that directly instead of letting the mean line assert a
# value that contradicts Figs 2 and 3.
BAND_ALGOS = ['L-sim', 'L-opp', 'wtf']
EMPHASIS = ['L-sim', 'L-opp']

TOPOLOGY_SETS = {
    'all': None,
    'directed': ['DPAH', 'Twitter'],
    'undirected': ['cl', 'FB'],
}

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.normpath(os.path.join(
    HERE, "..", "..", "Output",
    "heatmap_sweep_phased_sweep_20260707_1943_politicalClimate_tay.csv"))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "Figs", "Sensitivity"))

N_BOOT = 1000
RNG_SEED = 0


def setup_style():
    plt.rcParams.update({
        'font.size': FONT_SIZE, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.labelsize': FONT_SIZE - 1, 'ytick.labelsize': FONT_SIZE - 1,
        'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
    })


def load_data(path, topologies='all'):
    df = pd.read_csv(path)
    df['mode'] = df['mode'].fillna('none').astype(str).str.strip()
    df['rewiring'] = df['rewiring'].fillna('none').astype(str).str.strip()
    keys = list(zip(df['mode'], df['rewiring']))
    unknown = sorted({k for k in keys if k not in FRIENDLY_NAMES})
    if unknown:
        raise ValueError(f"unmapped (mode, rewiring) combinations in {path}: {unknown}")
    df['alg'] = [FRIENDLY_NAMES[k] for k in keys]
    df['state'] = pd.to_numeric(df['state'], errors='coerce')
    df = df.dropna(subset=['state'])
    keep = TOPOLOGY_SETS[topologies]
    if keep is not None:
        df = df[df['topology'].isin(keep)]
    return df


def _first_crossing(phi, y, target=0.0):
    """Field strength at which the curve y first reaches `target`."""
    for i in range(len(y) - 1):
        if y[i] < target <= y[i + 1]:
            return phi[i] + (target - y[i]) * (phi[i + 1] - phi[i]) / (y[i + 1] - y[i])
    return np.nan


def _aligned_fraction(values):
    return (values > 0).mean()


def _phi_star_draws(sub, phi_levels, rng, n_boot=N_BOOT):
    """Bootstrap draws of phi*, resampling runs within each field level.

    phi* is the field at which half the runs end aligned, not the field at which
    the ensemble mean turns positive. The two agree to within this CI here, but
    the occupancy definition stays meaningful when the ensemble is bimodal, which
    is exactly the regime R1.7 objects to summarising with a mean.

    Retaining the draws (rather than just their percentiles) lets the difference
    against the static baseline be read off the same resampling, which is what the
    "needs a stronger field than no rewiring" claim rests on: comparing marginal
    CIs by eye would understate the separation.
    """
    by_phi = [sub.loc[sub['politicalClimate'] == p, 'state'].to_numpy() for p in phi_levels]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        frac = np.array([_aligned_fraction(v[rng.integers(0, len(v), len(v))])
                         for v in by_phi])
        draws[b] = _first_crossing(phi_levels, frac, 0.5)
    return draws


def _ci(draws):
    d = draws[np.isfinite(draws)]
    if len(d) < 0.5 * len(draws):
        return np.nan, np.nan
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def summarise(df):
    """Per-algorithm curve, transition threshold and reach at the default field."""
    phi_levels = np.sort(df['politicalClimate'].unique())
    phi_min, phi_max = phi_levels[0], phi_levels[-1]
    rng = np.random.default_rng(RNG_SEED)

    curves, rows, draws = {}, [], {}
    for alg in [a for a in ALGO_ORDER if a in set(df['alg'])]:
        sub = df[df['alg'] == alg]
        g = sub.groupby('politicalClimate')['state']
        mean = g.mean().reindex(phi_levels).to_numpy()
        q25 = g.quantile(0.25).reindex(phi_levels).to_numpy()
        q75 = g.quantile(0.75).reindex(phi_levels).to_numpy()
        curves[alg] = dict(phi=phi_levels, mean=mean, q25=q25, q75=q75)

        aligned = g.apply(_aligned_fraction).reindex(phi_levels).to_numpy()
        lo_field = sub[sub['politicalClimate'] == phi_min]['state']
        hi_field = sub[sub['politicalClimate'] == phi_max]['state']
        draws[alg] = _phi_star_draws(sub, phi_levels, rng)
        ci_lo, ci_hi = _ci(draws[alg])
        rows.append(dict(
            algorithm=alg,
            topologies=len(sub['topology'].unique()),
            runs_per_field=int(len(hi_field)),
            mean_at_phi_min=lo_field.mean(),
            frac_aligned_at_phi_min=(lo_field > 0).mean(),
            phi_star=_first_crossing(phi_levels, aligned, 0.5),
            phi_star_ci_lo=ci_lo, phi_star_ci_hi=ci_hi,
            phi_star_mean_crossing=_first_crossing(phi_levels, mean),
            mean_at_phi_max=hi_field.mean(),
            median_at_phi_max=hi_field.median(),
            q25_at_phi_max=hi_field.quantile(0.25),
            q75_at_phi_max=hi_field.quantile(0.75),
            frac_aligned_at_phi_max=(hi_field > 0).mean(),
            frac_above_0p9_at_phi_max=(hi_field > 0.9).mean(),
        ))
    stats = pd.DataFrame(rows).sort_values('phi_star').reset_index(drop=True)

    if 'static' in draws:
        base = draws['static']
        diffs = {a: d - base for a, d in draws.items()}
        stats['phi_star_vs_static'] = [np.nanmean(diffs[a]) for a in stats['algorithm']]
        stats['phi_star_vs_static_ci_lo'] = [_ci(diffs[a])[0] for a in stats['algorithm']]
        stats['phi_star_vs_static_ci_hi'] = [_ci(diffs[a])[1] for a in stats['algorithm']]
        stats['p_needs_stronger_field_than_static'] = [
            np.nanmean(diffs[a] > 0) for a in stats['algorithm']]
    return curves, stats, phi_min, phi_max


def _declutter(positions, min_gap):
    """Nudge overlapping label positions apart, preserving their order."""
    order = np.argsort(positions)
    out = np.array(positions, dtype=float)
    for k in range(1, len(order)):
        lo, hi = order[k - 1], order[k]
        if out[hi] - out[lo] < min_gap:
            out[hi] = out[lo] + min_gap
    return out


def _style_axis(ax):
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.set_axisbelow(True)


def draw_response(ax, curves, phi_min, phi_max, end_labels=True):
    ax.axhline(0, color='0.45', lw=0.7, ls='--', zorder=1)

    for alg in BAND_ALGOS:
        if alg not in curves:
            continue
        c = curves[alg]
        # no shaded aligned/opposed half-planes: the wtf band is house grey and
        # would vanish into them.
        ax.fill_between(c['phi'], c['q25'], c['q75'], color=FRIENDLY_COLORS[alg],
                        alpha=0.22 if alg == 'wtf' else 0.15, lw=0, zorder=2)
    for alg, c in curves.items():
        emph = alg in EMPHASIS
        ax.plot(c['phi'], c['mean'], color=FRIENDLY_COLORS[alg], zorder=3,
                lw=1.5 if emph else 1.0,
                marker='o' if emph else None, ms=2.2, mew=0)

    if end_labels:
        # Standalone panel: eight curve ends span less than the height of eight
        # labels, so the labels are decluttered into a column and tied back to
        # their curve with a leader line.
        algs = list(curves)
        ends = [curves[a]['mean'][-1] for a in algs]
        placed = _declutter(ends, min_gap=0.115)
        placed = placed - (placed.max() - max(ends)) / 2
        for alg, y, y_end in zip(algs, placed, ends):
            ax.annotate(alg, xy=(phi_max, y_end), xytext=(phi_max * 1.035, y),
                        color=FRIENDLY_COLORS[alg], fontsize=FONT_SIZE - 1.5,
                        va='center', ha='left', annotation_clip=False,
                        arrowprops=dict(arrowstyle='-', lw=0.5, shrinkA=0, shrinkB=1,
                                        color=FRIENDLY_COLORS[alg], alpha=0.6))
    else:
        # Paired with the map panel, which carries the full colour key: only the
        # two families whose contrast the panel is about are named here.
        for alg, name, dy in [('L-sim', r'$x$(similar)', 0.13),
                              ('L-opp', r'$x$(opposite)', -0.16)]:
            if alg not in curves:
                continue
            c = curves[alg]
            i = np.searchsorted(c['phi'], phi_max * 0.80)
            ax.text(c['phi'][i], c['mean'][i] + dy, name, color=FRIENDLY_COLORS[alg],
                    fontsize=FONT_SIZE - 1, ha='center',
                    va='bottom' if dy > 0 else 'top')

    ax.text(phi_max * 0.017, 0.035, 'aligned', fontsize=FONT_SIZE - 1.5, color='0.5',
            va='bottom')
    ax.text(phi_max * 0.017, -0.035, 'opposed', fontsize=FONT_SIZE - 1.5, color='0.5',
            va='top')
    ax.text(phi_max * 0.988, -1.0, 'default', fontsize=FONT_SIZE - 2, color='0.5',
            rotation=90, va='bottom', ha='right')

    ax.set_xlim(phi_min, phi_max)
    ax.set_ylim(-1.05, 1.0)
    ax.set_xlabel(r'external field $\phi$')
    ax.set_ylabel(r'steady state $\langle a^* \rangle$')
    _style_axis(ax)


def draw_map(ax, stats):
    base = stats[stats['algorithm'] == 'static']
    if not base.empty:
        bx, by = base['phi_star'].iloc[0], base['mean_at_phi_max'].iloc[0]
        ax.axvline(bx, color='0.75', lw=0.7, ls=':', zorder=1)
        ax.axhline(by, color='0.75', lw=0.7, ls=':', zorder=1)

    x = stats['phi_star'].to_numpy()
    y = stats['mean_at_phi_max'].to_numpy()
    xerr = np.vstack([x - stats['phi_star_ci_lo'].to_numpy(),
                      stats['phi_star_ci_hi'].to_numpy() - x])
    # Vertical bars are the interquartile range across runs, not a CI on the mean:
    # R1.7 objects to point estimates precisely where the ensemble is multimodal,
    # so the spread has to be visible here rather than deferred to the SI.
    yerr = np.clip(np.vstack([y - stats['q25_at_phi_max'].to_numpy(),
                              stats['q75_at_phi_max'].to_numpy() - y]), 0, None)
    for i, alg in enumerate(stats['algorithm']):
        ax.errorbar(x[i], y[i], xerr=xerr[:, i:i + 1], yerr=yerr[:, i:i + 1], fmt='none',
                    ecolor=FRIENDLY_COLORS[alg], elinewidth=0.9, capsize=1.6, zorder=2,
                    alpha=0.85)
        ax.scatter(x[i], y[i], s=26, color=FRIENDLY_COLORS[alg], zorder=3,
                   edgecolor='w', linewidth=0.5)

    span_x = np.nanmax(x) - np.nanmin(x)
    pad_x = 0.06 * span_x
    # Label above-right, flipping below for points that would collide with one
    # already placed (the L/B pairs sit almost exactly on top of each other).
    placed = []
    for i, alg in enumerate(stats['algorithm']):
        below = any(abs(x[i] - px) < 0.09 * span_x and abs(y[i] - py) < 0.10
                    for px, py in placed)
        ax.annotate(alg, (x[i], y[i]), xytext=(4, -9 if below else 5),
                    textcoords='offset points', color=FRIENDLY_COLORS[alg],
                    fontsize=FONT_SIZE - 1.5, va='center', ha='left')
        placed.append((x[i], y[i]))

    ax.set_xlim(np.nanmin(x) - 2.2 * pad_x, np.nanmax(x) + 5.5 * pad_x)
    ax.set_ylim(-0.15, 1.0)
    ax.set_xticks(np.arange(0.020, 0.0451, 0.005))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
    ax.set_xlabel(r'transition field $\phi^*$ (half of runs aligned)')
    ax.set_ylabel(r'$\langle a^* \rangle$ at default $\phi$')
    if not base.empty:
        # deliberately descriptive ("worse on both axes"), not a significance claim:
        # only wtf is clearly separated from the baseline threshold, node2vec is
        # marginal. The paired bootstrap lives in the companion stats CSV.
        ax.annotate('worse than no rewiring\non both axes',
                    (bx + pad_x * 0.35, -0.115), fontsize=FONT_SIZE - 2, color='0.5',
                    va='bottom', ha='left')
    _style_axis(ax)


def make_figure(curves, stats, phi_min, phi_max, out_path, single=False):
    if single:
        fig, ax_a = plt.subplots(figsize=(0.55 * TEXTWIDTH, 6.4 * cm))
        draw_response(ax_a, curves, phi_min, phi_max, end_labels=True)
        fig.subplots_adjust(left=0.165, right=0.80, top=0.96, bottom=0.19)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 6.4 * cm))
        draw_response(axes[0], curves, phi_min, phi_max, end_labels=False)
        draw_map(axes[1], stats)
        for ax, letter in zip(axes, 'AB'):
            ax.text(-0.135, 1.04, letter, transform=ax.transAxes,
                    fontweight='bold', fontsize=FONT_SIZE + 1, va='bottom')
        # left/wspace are hand-set rather than tight_layout: the y-labels carry a
        # superscript that runs off the canvas if the margins are any tighter.
        fig.subplots_adjust(left=0.105, right=0.985, top=0.93, bottom=0.19, wspace=0.34)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    png = os.path.splitext(out_path)[0] + ".png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.environ.get("PHI_SWEEP_CSV", DEFAULT_CSV))
    ap.add_argument("--out", default=None)
    ap.add_argument("--single", action="store_true",
                    help="panel A only, sized for width=0.55\\textwidth")
    ap.add_argument("--topologies", choices=list(TOPOLOGY_SETS), default="all")
    args = ap.parse_args()

    setup_style()
    df = load_data(args.csv, args.topologies)
    curves, stats, phi_min, phi_max = summarise(df)

    out = args.out
    if out is None:
        suffix = "_single" if args.single else ""
        if args.topologies != "all":
            suffix += f"_{args.topologies}"
        out = os.path.join(FIG_DIR,
                           f"external_field_response_{date.today():%Y%m%d}{suffix}.pdf")

    png = make_figure(curves, stats, phi_min, phi_max, out, single=args.single)
    stats_path = os.path.splitext(out)[0] + "_stats.csv"
    stats.to_csv(stats_path, index=False)

    print(f"source: {args.csv}")
    print(f"topologies: {args.topologies} -> {sorted(df['topology'].unique())}")
    print(f"field range: {phi_min:g} to {phi_max:g} over {df['politicalClimate'].nunique()} levels")
    print(stats.round(4).to_string(index=False))
    print(f"saved:\n  {out}\n  {png}\n  {stats_path}")


if __name__ == "__main__":
    main()
