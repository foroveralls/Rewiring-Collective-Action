#!/usr/bin/env python3
"""Fig. 3 audit: confirm the plotted coordinates are rank percentiles, and
quantify how far the raw values sit from the numbers quoted in the caption/text."""
import sys, os
import numpy as np, pandas as pd

sys.path.insert(0, os.path.abspath("Analysis/Plotting"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cvc", "Analysis/Plotting/convergence_vs_cooperation.py")
cvc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cvc)

CSV = ("Output/default_run_avg_N_800_n_90_pNf_0_pc_0.05_"
       "sweep_20251014_1704_phased_run_gme_2025-10-15.csv")
print(f"input: {CSV}")
data = pd.read_csv(CSV)
print(f"rows: {len(data):,}  cols: {list(data.columns)[:12]}")

rows = []
for method in ("inflection", "t95"):
    m = cvc.calculate_metrics(data.copy(), method=method)
    m = m.rename(columns={"speed": f"speed_{method}"})
    rows.append(m.set_index(["scenario", "topology"])[[f"speed_{method}"]]
                if method == "t95" else
                m.set_index(["scenario", "topology"])[[f"speed_{method}", "cooperativity"]])
df = rows[0].join(rows[1]).reset_index()

# The exact transform the figure applies (convergence_vs_cooperation.py L253-254)
df["coop_rank"] = df["cooperativity"].rank(pct=True)
df["speed_rank_infl"] = df["speed_inflection"].rank(pct=True)
df["speed_rank_t95"] = df["speed_t95"].rank(pct=True)

# Pareto membership on RAW values (find_pareto_front always uses raw)
pf_infl = cvc.find_pareto_front(
    df.rename(columns={"speed_inflection": "speed"})[["speed", "cooperativity"]])
df["pareto_global_infl"] = pf_infl.values

df = df.sort_values("coop_rank", ascending=False)
pd.set_option("display.width", 200, "display.max_rows", 100)
print("\n=== ALL 30 CONDITIONS (inflection metric) ===")
print(df[["scenario", "topology", "cooperativity", "coop_rank",
          "speed_inflection", "speed_rank_infl", "pareto_global_infl"]]
      .to_string(index=False,
                 formatters={"cooperativity": "{:+.4f}".format,
                             "coop_rank": "{:.3f}".format,
                             "speed_inflection": "{:.4f}".format,
                             "speed_rank_infl": "{:.3f}".format}))

print("\n=== THE CAPTION'S CLAIM: B-sim on FB ===")
r = df[(df.scenario == "B-sim") & (df.topology == "FB")].iloc[0]
print(f"  caption says:  <a*> ~ 0.84,  convergence rate ~ 0.85")
print(f"  coop RANK percentile  = {r.coop_rank:.3f}   <-- what is plotted")
print(f"  speed RANK percentile = {r.speed_rank_infl:.3f}   <-- what is plotted")
print(f"  RAW <a*>              = {r.cooperativity:+.4f}")
print(f"  RAW inflection rate   = {r.speed_inflection:.4f}  (x1e-3 per step)")

print("\n=== CONSENSUS THRESHOLD CHECK (paper defines consensus as <a*> >= 0.8) ===")
print(f"  conditions with RANK  >= 0.8 : {(df.coop_rank >= 0.8).sum()} of {len(df)}")
print(f"  conditions with RAW   >= 0.8 : {(df.cooperativity >= 0.8).sum()} of {len(df)}")
above_raw = df[df.cooperativity >= 0.8]
print("  raw >= 0.8:", ", ".join(f"{a}/{b} ({c:+.3f})" for a, b, c in
      zip(above_raw.scenario, above_raw.topology, above_raw.cooperativity)) or "  (none)")
print(f"  RAW <a*> range over all 30: {df.cooperativity.min():+.4f} .. {df.cooperativity.max():+.4f}")

print("\n=== L185 CLAIM: B-sim 'rapid convergence (>0.70) or high mean opinion (>0.8) or both (FB)' ===")
for topo in ["DPAH", "cl", "Twitter", "FB"]:
    s = df[(df.scenario == "B-sim") & (df.topology == topo)]
    if len(s):
        s = s.iloc[0]
        print(f"  B-sim/{topo:8s} rank(speed)={s.speed_rank_infl:.2f} rank(a*)={s.coop_rank:.2f} "
              f"| RAW rate={s.speed_inflection:.4f} RAW a*={s.cooperativity:+.4f}")

print("\n=== L185 CLAIM: x(opposite) 'converge quickly on Twitter (~0.95), CSF (~1.0), FB (~1)' ===")
for sc in ["B-opp", "L-opp"]:
    for topo in ["Twitter", "cl", "FB"]:
        s = df[(df.scenario == sc) & (df.topology == topo)]
        if len(s):
            s = s.iloc[0]
            print(f"  {sc}/{topo:8s} rank(speed)={s.speed_rank_infl:.2f} "
                  f"| RAW rate={s.speed_inflection:.4f} RAW a*={s.cooperativity:+.4f}")

print("\n=== L185 CLAIM: wtf 'slower convergence than 83% of other scenarios' ===")
for topo in ["DPAH", "Twitter", "cl", "FB"]:
    s = df[(df.scenario == "wtf") & (df.topology == topo)]
    if len(s):
        s = s.iloc[0]
        pct_slower_than = 100 * (1 - s.speed_rank_infl)
        print(f"  wtf/{topo:8s} rank(speed)={s.speed_rank_infl:.3f} -> slower than "
              f"{pct_slower_than:.0f}% of scenarios | RAW rate={s.speed_inflection:.4f} "
              f"RAW a*={s.cooperativity:+.4f}")

print("\n=== DOES THE ORDERING SURVIVE? (rank is a monotone transform of raw) ===")
from scipy.stats import spearmanr
print(f"  spearman(raw a*, rank a*)       = {spearmanr(df.cooperativity, df.coop_rank).statistic:.6f}")
print(f"  spearman(raw rate, rank rate)   = {spearmanr(df.speed_inflection, df.speed_rank_infl).statistic:.6f}")
print(f"  sign agreement: rank axis cannot show sign; raw a* < 0 for "
      f"{(df.cooperativity < 0).sum()} of {len(df)} conditions:")
neg = df[df.cooperativity < 0]
for a, b, c, d in zip(neg.scenario, neg.topology, neg.cooperativity, neg.coop_rank):
    print(f"    {a}/{b}: raw {c:+.4f}  ->  plotted as percentile {d:.3f}")

df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fig3_raw_vs_rank.csv"), index=False)
print("\nsaved: fig3_raw_vs_rank.csv")
