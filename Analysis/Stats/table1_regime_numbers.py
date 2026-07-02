#!/usr/bin/env python3
"""
FIRST-PASS numeric backbone for Table 1 (R1.7).

(b) Per-group CIs in each algorithm's RECOMMENDED regime, from the sensitivity
    sweeps (per-run `state`=mean opinion a*, `state_std`=polarisation P*; 30 runs/cell).
    Convergence rate is intentionally OMITTED (sweeps store no trajectories).
    Each group is compared to RANDOM evaluated in the SAME regime.

Also prints the SI default-parameter table (per algorithm x topology) from
Output/per_run_summary.csv (n=90/cell): a*, P*, convergence (t95) median [IQR].

REGIME ASSUMPTIONS ARE A FIRST PASS — author to correct. Available axes:
  stub_pnf sweep: stubbornness in {0..1, 10 steps} x polarisingNode_f {0..1, 10 steps}, phi=0.05 (default)
  phi sweep:      politicalClimate in [0, 0.05] (12 steps), stubbornness=0.6 (default), pNf default
Defaults: phi=0.05, stubbornness=0.6, pNf~0.10. NOTE: no negative/unfavourable phi exists in the sweep.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root (this file lives in Analysis/Stats/)
OUT = os.path.join(ROOT, "Output")
STUB_PNF = os.path.join(OUT, "heatmap_sweep_phased_sweep_20251014_1511_stubbornness_polarisingNode_f_pxc.csv")
PHI = os.path.join(OUT, "heatmap_sweep_phased_sweep_20250908_1826_politicalClimate_lou.csv")
PER_RUN = os.path.join(OUT, "per_run_summary.csv")

sweep = {"stub_pnf": pd.read_csv(STUB_PNF), "phi": pd.read_csv(PHI)}
for d in sweep.values():
    d['mode'] = d['mode'].astype(str)
    d['rewiring'] = d['rewiring'].astype(str)


def ci(x):
    x = np.asarray(x, float)
    return f"{np.median(x):+.2f} [{np.percentile(x,25):+.2f}, {np.percentile(x,75):+.2f}]"


def subset(which, filt, modes_rews):
    """modes_rews: list of (mode, rewiring) where rewiring may be 'nan'."""
    d = sweep[which]
    m = pd.Series(True, index=d.index)
    for col, val in filt.items():
        m &= np.isclose(d[col].astype(float), val)
    sel = pd.Series(False, index=d.index)
    for mode, rew in modes_rews:
        sel |= (d['mode'] == mode) & (d['rewiring'] == rew)
    return d[m & sel]


# Regime config per group: (label, sweep, filter, group scenario combos, regime description)
RANDOM = [('random', 'nan')]
REGIMES = [
    ("x(opposite) [L-opp,B-opp]", "phi", {"politicalClimate": 0.0},
     [('biased', 'diff'), ('bridge', 'diff')],
     "neutral environment phi=0 (stub=0.6 default=med-high, pNf default)"),
    ("x(similar) [L-sim,B-sim]", "stub_pnf", {"polarisingNode_f": 0.0, "stubbornness": 0.5555555555555556},
     [('biased', 'same'), ('bridge', 'same')],
     "low divergers pNf=0, medium stubbornness=0.56 (phi=0.05 favourable=default)"),
    ("WTF", "stub_pnf", {"stubbornness": 0.8888888888888888, "polarisingNode_f": 0.1111111111111111},
     [('wtf', 'nan')],
     "high stubbornness=0.89, pNf=0.11~default (phi=0.05; UNFAVOURABLE phi UNAVAILABLE in sweep)"),
    ("N2V", "phi", {"politicalClimate": 0.05},
     [('node2vec', 'nan')],
     "default conditions phi=0.05 ('uncertain conditions' is vague - placeholder)"),
]

print("=" * 100)
print("(b) TABLE 1 — per-group a* and P* in recommended regime, vs RANDOM in the SAME regime")
print("    median [IQR], pooled across topologies & 30 runs/cell. Convergence OMITTED (no trajectories).")
print("=" * 100)
for label, wh, filt, combos, desc in REGIMES:
    g = subset(wh, filt, combos)
    r = subset(wh, filt, RANDOM)
    print(f"\n{label}   [regime: {desc}]")
    print(f"    n(group)={len(g):4d}  a* = {ci(g.state):26s}  P* = {ci(g.state_std)}")
    print(f"    n(rand) ={len(r):4d}  a* = {ci(r.state):26s}  P* = {ci(r.state_std)}   (random, same regime)")
    da = np.median(g.state) - np.median(r.state)
    dp = np.median(g.state_std) - np.median(r.state_std)
    print(f"    -> a* {'HIGHER' if da>0 else 'LOWER'} ({da:+.2f}),  P* {'HIGHER' if dp>0 else 'LOWER'} ({dp:+.2f}) vs random")

# Standalone Random reference at default (phi=0.05)
r0 = subset("phi", {"politicalClimate": 0.05}, RANDOM)
print(f"\nRandom (reference, default phi=0.05)  n={len(r0)}  a* = {ci(r0.state)}  P* = {ci(r0.state_std)}")

print("\n" + "=" * 100)
print("SI DEFAULT-PARAMETER TABLE — per algorithm x topology, from per_run_summary.csv (n=90/cell)")
print("    a* = cooperativity, P* = final_polarization, conv = speed_t95 ;  median [IQR]")
print("=" * 100)
pr = pd.read_csv(PER_RUN)
pr['rewiring'] = pr['rewiring'].fillna('none'); pr['scenario'] = pr['scenario'].fillna('none')
NAME = {'none_none': 'static', 'random_none': 'random', 'biased_same': 'L-sim',
        'biased_diff': 'L-opp', 'bridge_same': 'B-sim', 'bridge_diff': 'B-opp',
        'wtf_none': 'wtf', 'node2vec_none': 'node2vec'}
ORDER = ['static', 'random', 'L-sim', 'L-opp', 'B-sim', 'B-opp', 'wtf', 'node2vec']
TOPO = ['DPAH', 'cl', 'Twitter', 'FB']
pr['algo'] = (pr['scenario'] + '_' + pr['rewiring']).map(lambda g: NAME.get(g, g))
rows = []
for algo in ORDER:
    for topo in TOPO:
        s = pr[(pr['algo'] == algo) & (pr['type'] == topo)]
        if s.empty:
            continue
        rows.append({'algorithm': algo, 'topology': topo, 'n': len(s),
                     'a*': ci(s.cooperativity), 'P*': ci(s.final_polarization),
                     'conv(t95)': ci(s.speed_t95)})
si = pd.DataFrame(rows)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 60); pd.set_option('display.max_colwidth', 30)
print(si.to_string(index=False))
si.to_csv(os.path.join(OUT, "SI_default_params_table.csv"), index=False)
print(f"\nsaved {os.path.join(OUT, 'SI_default_params_table.csv')}")

# ---- emit accurate markdown + LaTeX for the SI default table into a draft MD ----
def md_table(df, cols):
    out = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return out

lt = si.copy()
lt.columns = ['Algorithm', 'Topology', '$n$', r'$\langle a^* \rangle$', r'$\langle P^* \rangle$', 'Conv. (t95)']
latex = lt.to_latex(index=False, escape=False, column_format='llccccc',
                    caption='Default-parameter steady-state metrics per algorithm and topology '
                            '(median [IQR] across 90 runs). $\\langle a^* \\rangle$ = mean opinion, '
                            '$\\langle P^* \\rangle$ = polarisation, Conv. = t95 convergence rate.',
                    label='tab:si_default_metrics')

gen_md = os.path.join(ROOT, "Submission", "Review", "table1_tables_generated.md")
with open(gen_md, "w") as f:
    f.write("# Auto-generated tables (accurate values) — for the R1.7 Table 1 draft\n\n")
    f.write("> Generated by `Analysis/Stats/table1_regime_numbers.py`. Do NOT hand-edit; rerun the script.\n\n")
    f.write("## SI default-parameter table (markdown)\n\n")
    f.write(md_table(si, ['algorithm', 'topology', 'n', 'a*', 'P*', 'conv(t95)']))
    f.write("\n## SI default-parameter table (LaTeX, paste-ready)\n\n```latex\n")
    f.write(latex)
    f.write("```\n")
print(f"saved {gen_md}")
