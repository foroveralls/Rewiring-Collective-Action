#!/usr/bin/env python3
"""
Minimal single-column panel for reviewer R1.2: "the dependence on this parameter
[the external field phi] should be more central in the main text, especially for
the claim that rewiring promotes consensus".

This is deliberately the smallest device that answers the comment. It is drawn to
sit as panel (b) underneath the single-column run-level violin, sharing that
panel's algorithm x-axis, so it costs a few centimetres of column rather than a
figure slot. The violins above are measured at the default field phi = 0.05, which
is this panel's TOP row: reading downwards is turning the field off.

Two variants of the same statement, chosen with --variant:

    ladder    (default) one column per algorithm, phi up the column, colour = median
              <a*> across runs. The bottom of every column is opposed: no rewiring
              rule aligns the population without the field. The black rule marks
              where that algorithm's median run turns aligned; the grey dashed rules
              bound the field range over which at least 15% of runs settle aligned
              (a* > 0.5) and at least 15% settle opposed (a* < -0.5), i.e. where the
              ensemble is genuinely divided rather than sitting near zero (the R1.7
              concern, shown rather than asserted). A bound that runs off the swept
              range is left open. The magnitude qualifier is load-bearing: a sign-only
              test flags any ensemble straddling zero, which on this data means the
              heterophilic rules get the widest markers despite being the least
              divided. See summarize().

    dumbbell  <a*> on y (same axis as the violins above), algorithm on x. Each stem
              runs from the median at phi = 0 to the median at the default field,
              with one rug tick per swept phi level, so the whole response is in the
              stem: ticks bunched low means nothing happens until the field is
              nearly at its default value. The bar at the top marker is the IQR
              across runs at the default field.

Both carry per-run spread, because R1.7's multimodality complaint was sourced from
the SI phi heatmap - answering R1.2 with a means-only main-text panel would hand
the reviewer a fresh instance of their other comment.

DATA: the corrected political-climate sweep, one row per run, columns
    state,state_std,politicalClimate,rewiring,mode,topology
Default is the `tay` sweep, verified 2026-07-27 to run at rho = 0.10 (see
claude_stuff/Review/si_heatmap_regeneration_2026-07-27.md section 2). The older `lou` and
`dyr` sweeps ran at rho = 0 and must NOT be used. PHI_SWEEP_CSV overrides the path.

Runs are pooled over the four topologies (30 runs each, so 120 per cell; WTF is
defined only on the directed topologies, so 60). --topologies directed|undirected
checks that the pooling is not hiding a disagreement.

Usage:
    python field_strip.py                          # ladder, one column, standalone
    python field_strip.py --variant dumbbell
    python field_strip.py --variant both           # comparison sheet, both stacked
    python field_strip.py --table                  # also write the LaTeX fallback
    python field_strip.py --topologies directed

Sizes are final printed sizes for a 237pt (8.35 cm) column, so include the PDF with
width=\\columnwidth and no further scaling.
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
import matplotlib.patheffects as pe
import seaborn as sns

# ---- style ---------------------------------------------------------------------
cm = 1 / 2.54
COLWIDTH = 8.35 * cm           # oup-authoring-template [contemporary,large]: 237pt
FONT_SIZE = 7

# house palette, keyed by the short names used in the violin figures
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
# same order as violin_run_level.py, so the columns line up when stacked
ALGO_ORDER = ['static', 'random', 'L-sim', 'L-opp', 'B-sim', 'B-opp',
              'wtf', 'node2vec']

TOPOLOGY_SETS = {'all': None, 'directed': ['DPAH', 'Twitter'],
                 'undirected': ['cl', 'FB'],
                 # single topologies, for matching the panel above if it is not pooled
                 'DPAH': ['DPAH'], 'cl': ['cl'], 'Twitter': ['Twitter'], 'FB': ['FB']}

SPLIT_FRAC = 0.15              # share of runs on each side that counts as "runs split"
BRANCH_MIN = 0.5               # |a*| above which a run counts as settled, not near zero

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.normpath(os.path.join(
    HERE, "..", "..", "Output",
    "heatmap_sweep_phased_sweep_20260707_1943_politicalClimate_tay.csv"))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "Figs", "Sensitivity"))


def setup_style():
    plt.rcParams.update({
        'font.size': FONT_SIZE, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
        'xtick.labelsize': FONT_SIZE - 1, 'ytick.labelsize': FONT_SIZE - 1,
        'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
        'hatch.linewidth': 0.3,
    })


# ---- data ----------------------------------------------------------------------
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


def summarize(df):
    """One row per (algorithm, phi): median, IQR, branch occupancy, n."""
    g = df.groupby(['alg', 'politicalClimate'])['state']
    out = pd.DataFrame({
        'median': g.median(), 'mean': g.mean(),
        'q25': g.quantile(0.25), 'q75': g.quantile(0.75),
        'frac_aligned': g.apply(lambda s: float((s > 0).mean())),
        'frac_aligned_settled': g.apply(lambda s: float((s > BRANCH_MIN).mean())),
        'frac_opposed_settled': g.apply(lambda s: float((s < -BRANCH_MIN).mean())),
        'n': g.size(),
    }).reset_index().rename(columns={'politicalClimate': 'phi'})
    out['minority'] = np.minimum(out['frac_aligned'], 1 - out['frac_aligned'])
    # `minority` is sign-based and is kept only for provenance: it cannot tell a genuinely
    # divided ensemble from one whose runs all sit near zero and straddle it. The
    # heterophilic rules cap <a*> near 0.20, so 82-100% of their runs fall within
    # |a*| < BRANCH_MIN across the whole window a sign test flags, i.e. a sign test marks
    # them as the most divided columns in the figure when they are the least. The drawn
    # criterion therefore requires both outcomes to be occupied away from zero.
    out['split'] = np.minimum(out['frac_aligned_settled'],
                              out['frac_opposed_settled']) >= SPLIT_FRAC
    return out


def _grid(summ, col):
    algos = [a for a in ALGO_ORDER if a in set(summ['alg'])]
    return summ.pivot(index='alg', columns='phi', values=col).loc[algos]


def _runs(flags):
    """Contiguous True stretches in a boolean sequence, as (start, stop) indices."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def crossing_phi(summ):
    """Field at which the median run turns aligned, linearly interpolated.

    Deliberately a median (branch-occupancy) crossing rather than a mean crossing:
    in the transition window the ensemble is split, and a mean crossing there is a
    statement about how many runs sit on each branch wearing the costume of a
    typical trajectory.
    """
    med = _grid(summ, 'median')
    phis = med.columns.values
    out = {}
    for alg, row in med.iterrows():
        y = row.values
        out[alg] = np.nan
        for i in range(len(y) - 1):
            if y[i] < 0 <= y[i + 1]:
                out[alg] = phis[i] + (0 - y[i]) * (phis[i + 1] - phis[i]) / (y[i + 1] - y[i])
                break
    return out


# ---- panels --------------------------------------------------------------------
def draw_ladder(ax, summ, cbar=True, xlabels=True):
    """phi (y) x algorithm (x) strip, colour = median <a*>."""
    med = _grid(summ, 'median')
    split = _grid(summ, 'split')
    algos = list(med.index)
    phis = med.columns.values
    step = np.diff(phis).mean()
    edges_y = np.concatenate([phis - step / 2, [phis[-1] + step / 2]])
    edges_x = np.arange(len(algos) + 1) - 0.5

    # one column per algorithm, gutters between them: phi is continuous within a
    # column, the algorithms are categorical across them
    cmap = sns.diverging_palette(20, 220, as_cmap=True, center="light")
    half = 0.45
    halo = [pe.withStroke(linewidth=1.8, foreground='white')]
    crossings = crossing_phi(summ)
    for i, alg in enumerate(algos):
        mesh = ax.pcolormesh([i - half, i + half], edges_y,
                             med.loc[alg].values[:, None], cmap=cmap,
                             vmin=-1, vmax=1, edgecolors='none', zorder=2)
        # Adjacent quads leave hairline seams at every phi edge in vector PDF output,
        # which read as rules across the column. edgecolors='none' does not prevent
        # them; rasterizing the mesh does. The overlaid rules and all text stay vector.
        mesh.set_rasterized(True)
        # bounds of the window over which the ensemble is divided between the aligned
        # and opposed outcomes, taken as one span per algorithm; a bound that runs off
        # the swept range is left open, i.e. undrawn
        runs = _runs(split.loc[alg].values)
        if runs:
            lo, hi = runs[0][0], runs[-1][1]
            for j, y in ((lo, phis[lo] - step / 2), (hi, phis[hi] + step / 2)):
                if 0 < j < len(phis) - 1:
                    ax.plot([i - half, i + half], [y, y], color='0.32', lw=0.6,
                            dashes=(2.2, 1.4), zorder=4)
        # where the median run turns aligned
        phi_c = crossings[alg]
        if np.isfinite(phi_c):
            ax.plot([i - half, i + half], [phi_c, phi_c], color='0.05', lw=1.0,
                    solid_capstyle='butt', zorder=5, path_effects=halo)

    ax.set_ylim(edges_y[0], edges_y[-1])
    ax.set_xlim(edges_x[0], edges_x[-1])
    ax.set_yticks([0.0, 0.025, 0.05])
    ax.set_yticklabels(['0', '0.025', '0.05'])
    ax.set_yticks(phis, minor=True)          # shows the sweep resolution
    ax.tick_params(axis='y', which='minor', size=1.0, width=0.5, color='0.6')
    ax.set_ylabel(r'field $\phi$', labelpad=1.5)
    ax.set_xticks(np.arange(len(algos)))
    ax.set_xticklabels(algos if xlabels else [], rotation=45, ha='right')
    ax.tick_params(axis='x', length=0, pad=1)
    ax.tick_params(axis='y', size=1.8, pad=1.5)
    # a light frame keeps the strip's extent readable where cells are near white
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color('0.75')
    # the top row is the default field, i.e. the condition the violins above show;
    # said in the caption rather than on the panel, which has no room to spare

    if cbar:
        cb = ax.figure.colorbar(mesh, ax=ax, fraction=0.030, pad=0.045,
                                ticks=[-1, 0, 1], aspect=11)
        cb.ax.set_yticklabels(['$-1$', '0', '1'])
        # per-run notation: $a^*$ is one run's steady state, the colour is the median
        # across runs; $\langle\cdot\rangle$ would claim the ensemble mean (MS L87)
        cb.set_label(r'median $a^*$', labelpad=2,
                     fontsize=FONT_SIZE - 1)
        cb.outline.set_visible(False)
        cb.ax.tick_params(width=0.5, size=1.4, pad=1.5,
                          labelsize=FONT_SIZE - 1.5)
        # name the poles once, so the reader does not have to decode the sign
        cb.ax.text(0.5, 1.035, 'aligned', transform=cb.ax.transAxes, ha='center',
                   va='bottom', fontsize=FONT_SIZE - 2, color='0.35')
        cb.ax.text(0.5, -0.035, 'opposed', transform=cb.ax.transAxes, ha='center',
                   va='top', fontsize=FONT_SIZE - 2, color='0.35')
    return mesh


def draw_dumbbell(ax, summ, xlabels=True):
    """<a*> (y) x algorithm (x): phi = 0 to default, with a rug tick per phi."""
    med = _grid(summ, 'median')
    q25, q75 = _grid(summ, 'q25'), _grid(summ, 'q75')
    algos = list(med.index)
    phis = med.columns.values

    ax.axhline(0.0, color='0.55', lw=0.7, ls='--', zorder=1)
    for i, alg in enumerate(algos):
        c = FRIENDLY_COLORS[alg]
        lo, hi = med.loc[alg].iloc[0], med.loc[alg].iloc[-1]
        ax.plot([i, i], [lo, hi], color=c, lw=1.1, alpha=0.55,
                solid_capstyle='round', zorder=2)
        # the response itself: one tick per swept field level
        ax.scatter([i] * len(phis), med.loc[alg].values, marker='_', s=13,
                   color=c, linewidth=0.9, alpha=0.9, zorder=3)
        # spread across runs at the default field
        ax.plot([i, i], [q25.loc[alg].iloc[-1], q75.loc[alg].iloc[-1]],
                color=c, lw=2.6, alpha=0.35, solid_capstyle='round', zorder=2.5)
        ax.scatter([i], [lo], s=11, facecolor='white', edgecolor=c,
                   linewidth=0.9, zorder=4)
        ax.scatter([i], [hi], s=13, color=c, linewidth=0, zorder=4)

    ax.set_ylim(-1.08, 1.42)
    ax.set_xlim(-0.6, len(algos) - 0.4)
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel(r'$a^*$', labelpad=1.5)
    ax.set_xticks(np.arange(len(algos)))
    ax.set_xticklabels(algos if xlabels else [], rotation=45, ha='right')
    ax.tick_params(axis='x', pad=1)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    # marker key on one line, in the headroom above the data
    handles = [
        Line2D([], [], marker='o', color='none', markerfacecolor='white',
               markeredgecolor='0.4', markersize=2.8, label=r'$\phi=0$'),
        Line2D([], [], marker='_', color='0.4', markersize=4,
               linestyle='none', label='swept levels'),
        Line2D([], [], marker='o', color='none', markerfacecolor='0.4',
               markeredgecolor='none', markersize=3.2,
               label=r'default $\phi$ (IQR)'),
    ]
    ax.legend(handles=handles, loc='upper left', ncol=3, frameon=False,
              fontsize=FONT_SIZE - 1.5, handlelength=0.9, handletextpad=0.35,
              columnspacing=0.9, borderpad=0.0, borderaxespad=0.15)


# ---- output --------------------------------------------------------------------
def write_table(summ, path):
    """LaTeX fallback: the same statement as a four-column table."""
    med, q25, q75 = _grid(summ, 'median'), _grid(summ, 'q25'), _grid(summ, 'q75')
    cross = crossing_phi(summ)
    lines = [r'\begin{tabular}{lccc}', r'\toprule',
             r'& $\langle a^* \rangle$ at & transition & $\langle a^* \rangle$ at '
             r'default $\phi$ \\',
             r'rewiring & $\phi = 0$ & field $\phi^*$ & (IQR across runs) \\',
             r'\midrule']
    for alg in med.index:
        lines.append(
            f'{alg} & ${med.loc[alg].iloc[0]:.2f}$ & ${cross[alg]:.3f}$ & '
            f'${med.loc[alg].iloc[-1]:.2f}$ '
            f'(${q25.loc[alg].iloc[-1]:.2f}$--${q75.loc[alg].iloc[-1]:.2f}$) \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"  {path}")


def save(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=400, bbox_inches='tight', pad_inches=0.01)
    png = os.path.splitext(out_path)[0] + ".png"
    fig.savefig(png, dpi=400, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print(f"  {out_path}\n  {png}")


def report(summ):
    cross = crossing_phi(summ)
    med = _grid(summ, 'median')
    print("\nalg        median <a*> phi=0   phi* (median crosses 0)   <a*> default")
    for alg in med.index:
        print(f"  {alg:9s} {med.loc[alg].iloc[0]:>10.2f} {cross[alg]:>20.4f} "
              f"{med.loc[alg].iloc[-1]:>15.2f}")
    n_split = int(summ['split'].sum())
    at_default = summ[(summ['phi'] == summ['phi'].max()) & summ['split']]['alg'].tolist()
    print(f"\n  cells with >={SPLIT_FRAC:.0%} of runs settled on each side "
          f"(|a*| > {BRANCH_MIN}): {n_split} of {len(summ)}")
    print(f"  still divided at the default field: {at_default or 'none'}")
    # the drawn rules bracket the outermost divided cells, so a column whose divided
    # cells are not contiguous carries a rule spanning an undivided one: any caption
    # claim of the form "between the rules, ..." has to be checked against this
    split = _grid(summ, 'split')
    for alg in split.index:
        idx = np.flatnonzero(split.loc[alg].values)
        if len(idx) and idx[-1] - idx[0] + 1 != len(idx):
            gaps = [split.columns[j] for j in range(idx[0], idx[-1] + 1)
                    if not split.loc[alg].values[j]]
            print(f"  NOTE {alg}: divided cells are not contiguous, undivided at phi="
                  f"{', '.join(f'{g:.4f}' for g in gaps)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.environ.get("PHI_SWEEP_CSV", DEFAULT_CSV))
    ap.add_argument("--variant", default="ladder",
                    choices=["ladder", "dumbbell", "both"])
    ap.add_argument("--topologies", default="all", choices=list(TOPOLOGY_SETS))
    ap.add_argument("--height", type=float, default=None,
                    help="panel height in cm (default 3.1 ladder / 3.3 dumbbell)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--table", action="store_true",
                    help="also write the LaTeX table fallback")
    args = ap.parse_args()

    setup_style()
    df = load_data(args.csv, args.topologies)
    summ = summarize(df)
    print(f"{len(df):,} runs  |  {df['alg'].nunique()} algorithms  |  "
          f"{df['politicalClimate'].nunique()} field levels  |  "
          f"topologies={sorted(df['topology'].unique())}")

    suffix = '' if args.topologies == 'all' else f'_{args.topologies}'
    stem = args.out or os.path.join(
        FIG_DIR, f"field_strip_{args.variant}{suffix}_{date.today():%Y%m%d}.pdf")

    if args.variant == 'both':
        h = args.height or 6.8
        fig, axes = plt.subplots(2, 1, figsize=(COLWIDTH, h * cm))
        draw_ladder(axes[0], summ, xlabels=False)
        draw_dumbbell(axes[1], summ)
        for ax, lab in zip(axes, 'ab'):
            ax.set_title(lab, loc='left', fontweight='bold', pad=2)
        fig.tight_layout(h_pad=1.4)
    elif args.variant == 'ladder':
        h = args.height or 3.1
        fig, ax = plt.subplots(figsize=(COLWIDTH, h * cm))
        draw_ladder(ax, summ)
        fig.tight_layout(pad=0.15)
    else:
        h = args.height or 3.3
        fig, ax = plt.subplots(figsize=(COLWIDTH, h * cm))
        draw_dumbbell(ax, summ)
        fig.tight_layout(pad=0.15)

    print("saved:")
    save(fig, stem)
    summ.to_csv(os.path.splitext(stem)[0] + "_stats.csv", index=False)
    print(f"  {os.path.splitext(stem)[0]}_stats.csv")
    if args.table:
        write_table(summ, os.path.splitext(stem)[0] + "_table.tex")
    report(summ)


if __name__ == "__main__":
    main()
