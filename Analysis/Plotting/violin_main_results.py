#!/usr/bin/env python3
"""
Violin plot of the FINAL per-agent opinion distribution, faceted by network type
with rewiring algorithms on the x-axis. Addresses reviewer R1.7: collapsing the
ensemble into means hides bimodality / bistability; the per-agent violins expose it
(most visibly for `wtf`).

DATA: reads the per-agent CSV produced by extract_states_lowmem.py with columns
    scenario,rewiring,type,model_run,final_t,state
The full per-agent extraction is cluster-only (the snapshot pickle OOMs a 15GB box);
this script is data-agnostic and runs identically on the cluster output and on a
synthetic dev sample.

Usage:
    python violin_main_results.py                       # uses Output/per_agent_final_states.csv
    python violin_main_results.py --csv <path> [--out <path.pdf>]

Style constants mirror plots_lines.py / convergence_vs_cooperation.py so the figure
is visually consistent with Figs 2 and 3.
"""
import os
import argparse
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- style (copied from plots_lines.py / convergence_vs_cooperation.py) --------
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
TYPE_ORDER = ['DPAH', 'cl', 'Twitter', 'FB']  # matches convergence_vs_cooperation.py

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.normpath(os.path.join(HERE, "..", "..", "Output",
                                            "per_agent_final_states.csv"))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "Figs", "Distributions"))


def _norm(x):
    x = str(x).strip()
    return 'none' if x in ('', 'None', 'none', 'nan', 'NaN', '<NA>') else x.lower()


def load_data(path):
    # keep_default_na=False so the literal string "None" is preserved, not parsed to NaN
    df = pd.read_csv(path, keep_default_na=False,
                     dtype={'scenario': str, 'rewiring': str, 'type': str})
    df['state'] = pd.to_numeric(df['state'], errors='coerce')
    df = df.dropna(subset=['state'])
    df['grouped'] = df['scenario'].map(_norm) + '_' + df['rewiring'].map(_norm)
    return df


def _style_axis(ax):
    ax.set_ylim(-1.08, 1.08)
    ax.axhline(0.0, color='0.5', lw=0.8, ls='--', alpha=0.8, zorder=1)
    ax.yaxis.grid(True, alpha=0.4, linestyle='--', linewidth=line_params["grid_line_width"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(line_params["axis_line_width"])
    ax.tick_params(axis='both', which='major', width=line_params["tick_major_width"])


def make_violin(df, out_path, preview=False):
    types_present = [t for t in TYPE_ORDER if t in df['type'].unique()]
    n = len(types_present)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(19 * cm, 8.5 * nrows * cm),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax_i, type_ in enumerate(types_present):
        ax = axes[ax_i]
        sub = df[df['type'] == type_]
        algos = [g for g in ALGO_ORDER if (sub['grouped'] == g).any()]
        data = [sub.loc[sub['grouped'] == g, 'state'].to_numpy() for g in algos]
        pos = np.arange(len(algos))

        parts = ax.violinplot(data, positions=pos, widths=0.85,
                              showmeans=False, showmedians=False, showextrema=False)
        for body, g in zip(parts['bodies'], algos):
            body.set_facecolor(PLOT_COLORS[g])
            body.set_edgecolor('black')
            body.set_linewidth(0.6)
            body.set_alpha(0.85)
            # opinions are bounded in [-1, 1]; clip the KDE spill past the bounds
            v = body.get_paths()[0].vertices
            v[:, 1] = np.clip(v[:, 1], -1.0, 1.0)

        # per-run mean overlay (jittered) — shows across-run spread / bistability,
        # the complement to the per-agent KDE (within-run structure). R1.7 named both.
        rng = np.random.default_rng(0)
        for p, g in zip(pos, algos):
            run_means = (sub[sub['grouped'] == g]
                         .groupby('model_run')['state'].mean().to_numpy())
            jit = rng.uniform(-0.13, 0.13, size=len(run_means))
            ax.scatter(p + jit, run_means, s=5, color='black', alpha=0.40,
                       linewidth=0, zorder=3.5)

        # mini box-and-median overlay (q25-q75 line + median dot), drawn on top
        for p, arr in zip(pos, data):
            q1, med, q3 = np.percentile(arr, [25, 50, 75])
            ax.vlines(p, q1, q3, color='black', lw=2.4, zorder=4)
            ax.scatter(p, med, s=14, color='white', edgecolor='black',
                       linewidth=0.7, zorder=5)

        ax.set_xticks(pos)
        ax.set_xticklabels([FRIENDLY_NAMES[g] for g in algos], rotation=45, ha='right')
        ax.set_xlim(-0.6, len(algos) - 0.4)
        ax.set_title(NETWORK_DISPLAY_NAMES.get(type_, type_), fontsize=FONT_SIZE)
        if ax_i % ncols == 0:
            ax.set_ylabel(r'Final opinion $a_i$')
        _style_axis(ax)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if preview:
        fig.suptitle("PREVIEW — SYNTHETIC SAMPLE DATA (not real results)",
                     fontsize=FONT_SIZE - 1, color='firebrick', y=1.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    png = os.path.splitext(out_path)[0] + ".png"
    fig.savefig(png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved:\n  {out_path}\n  {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load_data(args.csv)
    preview = "SAMPLE" in os.path.basename(args.csv).upper()
    if args.out:
        out = args.out
    else:
        suffix = "_SAMPLE_PREVIEW" if preview else ""
        out = os.path.join(FIG_DIR, f"violin_final_opinion_{date.today()}{suffix}.pdf")

    print(f"rows={len(df):,}  types={sorted(df['type'].unique())}  "
          f"algos={sorted(df['grouped'].unique())}")
    make_violin(df, out, preview=preview)


if __name__ == "__main__":
    main()
