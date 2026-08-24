# -*- coding: utf-8 -*-
"""
Equivalence test: pre-optimization model vs the current (optimized) model.

`Analysis/models_checks.py` was refactored for speed on 2026-07-02 (commit
8c8e230). This script re-runs that verification from scratch, so the claim "results are
statistically equivalent" can be checked at any time rather than trusted.

The pre-optimization file is pulled straight out of git (default 8c8e230^,
i.e. the last state before the refactor) and imported side by side with the
working-tree version, so nothing has to be kept in sync by hand.

Two stages, mirroring where the optimizations can and cannot preserve the RNG
stream:

  --mode exact     Paired, seed-matched, bit-level. With WEIGHT_BATCH=1 and the
                   network cache off, the optimized code should consume the
                   identical random stream, so trajectories, per-agent final
                   states and final edge sets must match to float noise. Covers
                   the deterministic algorithms (None/random/biased/bridge).
                   A failure here is a genuine behaviour change.

  --mode ensemble  Production settings (batched weight draws + network cache),
                   which deliberately reorder RNG consumption, so runs can only
                   agree in distribution. Independent ensembles per
                   implementation are compared per metric with a difference
                   test (Welch, Mann-Whitney, KS, Holm-corrected) AND a TOST
                   equivalence test, since "p > 0.05" on its own is not
                   evidence of sameness. Covers wtf as well, which was never
                   bit-reproducible (rustworkx pagerank near-ties).

Small N/steps are fine here: the question is whether the two implementations
sample the same distribution, not whether the sweep horizons are converged.

Run from the Analysis/ directory (relative paths, as in the sweep scripts):

    python optimization_equivalence_test.py --smoke              # ~2 min sanity check
    python optimization_equivalence_test.py                      # exact + ensemble, defaults
    python optimization_equivalence_test.py --mode ensemble --runs 40 --n 400 --steps 8000
    python optimization_equivalence_test.py --mode exact --scenarios bridge/diff

Outputs (../Output/):
    optimization_equivalence_runs_<date>.csv     per-run metrics, both implementations
    optimization_equivalence_stats_<date>.csv    per scenario x metric test table
    optimization_equivalence_exact_<date>.csv    per scenario bit-level differences
"""

import os
import sys
import time
import argparse
import tempfile
import subprocess
import importlib.util
import multiprocessing
from datetime import date
from itertools import repeat

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

import matplotlib
matplotlib.use("Agg")  # the model imports pyplot; keep it headless

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ANALYSIS_DIR, ".."))

OPT_COMMIT = "8c8e230"          # "optimizations and new graph implementations", 2026-07-02
LEGACY_REV = OPT_COMMIT + "^"   # last state of models_checks.py before the refactor

# (algorithm, rewiringMode). wtf is distribution-only: its recommendations were
# never bit-reproducible, so it is excluded from the exact stage.
SCENARIOS = [("None", "None"), ("random", "None"),
             ("biased", "same"), ("biased", "diff"),
             ("bridge", "same"), ("bridge", "diff"),
             ("wtf", "None")]
EXACT_EXCLUDE = {"wtf", "node2vec"}

# empirical topologies come with a fixed size, so --n is ignored for these
TOP_FILES = {"Twitter": ("twitter_graph_N_789.gpickle", 789),
             "FB": ("FB_graph_N_786.gpickle", 786)}

# metrics compared in the ensemble stage; the floor keeps TOST margins sane when
# a metric has (near) zero spread, e.g. degree-conserving rewiring
METRICS = {"state_final": 0.02, "state_tail": 0.02, "ratio_tail": 0.01,
           "state_sd_final": 0.01, "t95": 1.0,
           "avg_degree": 0.05, "sd_degree": 0.05, "n_edges": 1.0}

IMPLS = {}  # name -> module; populated before the pool forks, inherited by workers


# %% loading the two implementations
def git_file(rev):
    """models_checks.py as of `rev`, or None if that revision is unavailable."""
    r = subprocess.run(["git", "-C", REPO_ROOT, "show", f"{rev}:Analysis/models_checks.py"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def load_legacy(rev):
    """Import the pre-refactor models_checks.py straight from git history."""
    src = git_file(rev)
    if src is None:
        raise SystemExit(f"cannot read Analysis/models_checks.py at revision '{rev}' "
                         f"(expected the pre-optimization state, default {LEGACY_REV})")
    tmpdir = tempfile.mkdtemp(prefix="models_checks_legacy_")
    path = os.path.join(tmpdir, "models_checks_legacy.py")
    with open(path, "w") as f:
        f.write(src)

    # the legacy file adds its own parent to sys.path, which is now a temp dir,
    # so the repo root has to be there for `from Auxillary import ...`
    for p in (REPO_ROOT, ANALYSIS_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)

    spec = importlib.util.spec_from_file_location("models_checks_legacy", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["models_checks_legacy"] = mod
    spec.loader.exec_module(mod)
    return mod, path


def working_tree_is_optimized():
    """True if the working-tree model still matches the optimized commit."""
    committed = git_file(OPT_COMMIT)
    with open(os.path.join(ANALYSIS_DIR, "models_checks.py")) as f:
        return committed is not None and f.read() == committed


# %% per-run metrics
def time_to_fraction(states, tail_mean, frac=0.95):
    """Steps until 95% of the total drift from the initial to the plateau value.

    NaN when the trajectory barely moves: the crossing index is then dominated
    by noise and the value flips sign for no meaningful reason.
    """
    s = np.asarray(states, dtype=float)
    span = tail_mean - s[0]
    if abs(span) < 0.05:
        return np.nan
    reached = (s - s[0]) / span >= frac
    return float(np.argmax(reached)) if reached.any() else np.nan


def summarize(model, tail_frac=0.1):
    """Reduce a finished model to the numbers the paper's results depend on."""
    states = np.asarray(model.states, dtype=float)
    ratio = np.asarray(model.ratio, dtype=float)
    sds = np.asarray(model.statesds, dtype=float)
    k = max(1, int(len(states) * tail_frac))
    degrees = np.array([d for _, d in model.graph.degree()], dtype=float)
    tail = float(states[-k:].mean())
    return {"state_final": float(states[-1]),
            "state_tail": tail,
            "ratio_final": float(ratio[-1]),
            "ratio_tail": float(ratio[-k:].mean()),
            "state_sd_final": float(sds[-1]),
            "t95": time_to_fraction(states, tail),
            "avg_degree": float(degrees.mean()),
            "sd_degree": float(degrees.std()),
            "max_degree": float(degrees.max()),
            "n_edges": float(model.graph.number_of_edges())}


def make_args(base_args, algo, mode, topo, nwsize, steps):
    top_file, fixed_n = TOP_FILES.get(topo, (None, None))
    return {**base_args, "rewiringAlgorithm": algo, "rewiringMode": mode,
            "type": topo, "top_file": top_file,
            "nwsize": fixed_n if fixed_n else nwsize, "timesteps": steps,
            "plot": False, "save_snapshots": False, "seed": 42}


# %% ensemble stage
def _run_pair(i, sim_args):
    """One run index, both implementations, in the same worker.

    simulate() seeds from (scenario, run index, pid), so running both here gives
    each implementation the identical seed for run i: the ensembles are paired
    on the seed even though production settings make the streams diverge.
    """
    out = {}
    for name in ("legacy", "current"):
        t0 = time.perf_counter()
        model = IMPLS[name].simulate(i, sim_args)
        out[name] = summarize(model)
        out[name]["seconds"] = time.perf_counter() - t0
    return i, out


def run_ensemble(scenarios, base_args, runs, topo, nwsize, steps, jobs):
    rows = []
    ctx = multiprocessing.get_context("fork")  # workers inherit both loaded modules
    pool = ctx.Pool(processes=jobs) if jobs > 1 else None
    try:
        for algo, mode in scenarios:
            sim_args = make_args(base_args, algo, mode, topo, nwsize, steps)
            t0 = time.time()
            work = zip(range(runs), repeat(sim_args))
            results = pool.starmap(_run_pair, work) if pool else [_run_pair(*w) for w in work]
            for i, out in results:
                for name, metrics in out.items():
                    rows.append({"impl": name, "algo": algo, "mode": mode,
                                 "scenario": f"{algo}/{mode}", "run": i, **metrics})
            print(f"  {algo}/{mode}: {runs} runs x2 in {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    return pd.DataFrame(rows)


# %% statistics
def tost(x, y, margin):
    """Two one-sided Welch tests: p < alpha means |true difference| < margin."""
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    se = np.sqrt(vx / nx + vy / ny)
    if se == 0:
        return 0.0 if abs(np.mean(x) - np.mean(y)) < margin else 1.0
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    d = np.mean(x) - np.mean(y)
    p_lower = stats.t.sf((d + margin) / se, df)   # H0: d <= -margin
    p_upper = stats.t.cdf((d - margin) / se, df)  # H0: d >= +margin
    return float(max(p_lower, p_upper))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (order preserved)."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    kept = p[ok]
    n = len(kept)
    adj = np.full_like(p, np.nan)
    out = np.empty(n)
    running = 0.0
    for rank, j in enumerate(np.argsort(kept)):
        running = max(running, (n - rank) * kept[j])  # step-down, monotone in rank
        out[j] = min(1.0, running)
    adj[ok] = out
    return adj


def compare(df, tost_sd):
    """Per scenario x metric: difference tests + TOST equivalence."""
    rows = []
    for scenario in df["scenario"].unique():
        sub = df[df["scenario"] == scenario]
        for metric, floor in METRICS.items():
            x = sub.loc[sub["impl"] == "legacy", metric].dropna().to_numpy()
            y = sub.loc[sub["impl"] == "current", metric].dropna().to_numpy()
            if len(x) < 3 or len(y) < 3:
                continue
            sd_pooled = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
            margin = max(tost_sd * sd_pooled, floor)
            diff = float(np.mean(y) - np.mean(x))  # current - legacy
            se = np.sqrt(np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y))
            p_welch = stats.ttest_ind(x, y, equal_var=False).pvalue if se > 0 else 1.0
            p_mwu = stats.mannwhitneyu(x, y).pvalue if sd_pooled > 0 else 1.0
            p_ks = stats.ks_2samp(x, y).pvalue if sd_pooled > 0 else 1.0
            rows.append({
                "scenario": scenario, "metric": metric,
                "mean_legacy": float(np.mean(x)), "sd_legacy": float(np.std(x, ddof=1)),
                "mean_current": float(np.mean(y)), "sd_current": float(np.std(y, ddof=1)),
                "diff": diff,
                "ci95_lo": diff - 1.96 * se, "ci95_hi": diff + 1.96 * se,
                "cohens_d": diff / sd_pooled if sd_pooled > 0 else 0.0,
                "p_welch": float(p_welch), "p_mwu": float(p_mwu), "p_ks": float(p_ks),
                "tost_margin": margin, "p_tost": tost(x, y, margin)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # difference tests are the family we correct: many scenario x metric cells,
    # all under the same null that the refactor changed nothing
    out["p_welch_holm"] = holm(out["p_welch"].to_numpy())
    out["p_ks_holm"] = holm(out["p_ks"].to_numpy())
    # a metric that is constant within each arm (e.g. edge count under
    # degree-conserving rewiring) has no test statistic, so any gap between the
    # two constants is a deterministic difference rather than an undecided one
    deterministic_gap = ((out["sd_legacy"] == 0) & (out["sd_current"] == 0)
                         & (out["diff"].abs() > 0))
    out["verdict"] = np.where(
        (out["p_welch_holm"] < 0.05) | (out["p_ks_holm"] < 0.05) | deterministic_gap,
        "DIFFERENT",
        np.where(out["p_tost"] < 0.05, "EQUIVALENT", "INCONCLUSIVE"))
    return out


# %% exact stage
def edge_key(graph):
    if nx.is_directed(graph):
        return set(graph.edges())
    return {frozenset(e) for e in graph.edges()}


def agent_states(model):
    return np.array([model.graph.nodes[n]['agent'].state for n in sorted(model.graph.nodes)])


def run_exact(scenarios, base_args, runs, topo, nwsize, steps, tol):
    """Seed-matched paired runs with the RNG-reordering optimizations disabled.

    WEIGHT_BATCH=1 and USE_NETWORK_CACHE=False restore the original order of
    random draws, so changes 1-3 of the refactor (incremental statistics, the
    rustworkx mirror, the bridge hoisting) must reproduce the old run exactly.
    """
    cur = IMPLS["current"]
    saved = (cur.WEIGHT_BATCH, cur.USE_NETWORK_CACHE)
    cur.WEIGHT_BATCH, cur.USE_NETWORK_CACHE = 1, False
    rows = []
    try:
        for algo, mode in scenarios:
            if algo in EXACT_EXCLUDE:
                continue
            sim_args = make_args(base_args, algo, mode, topo, nwsize, steps)
            for i in range(runs):
                # same process, same run index -> simulate() derives the same seed
                old = IMPLS["legacy"].simulate(i, sim_args)
                new = cur.simulate(i, sim_args)
                d_state = np.abs(np.asarray(old.states) - np.asarray(new.states)).max()
                d_ratio = np.abs(np.asarray(old.ratio) - np.asarray(new.ratio)).max()
                d_sd = np.abs(np.asarray(old.statesds) - np.asarray(new.statesds)).max()
                d_agents = np.abs(agent_states(old) - agent_states(new)).max()
                same_edges = edge_key(old.graph) == edge_key(new.graph)
                ok = max(d_state, d_ratio, d_sd, d_agents) < tol and same_edges
                rows.append({"scenario": f"{algo}/{mode}", "run": i,
                             "max_dstate": d_state, "max_dratio": d_ratio,
                             "max_dsd": d_sd, "max_dagent": d_agents,
                             "identical_edges": same_edges,
                             "n_edges_legacy": old.graph.number_of_edges(),
                             "n_edges_current": new.graph.number_of_edges(),
                             "verdict": "PASS" if ok else "FAIL"})
                print(f"  {algo}/{mode} run {i}: max|dstate|={d_state:.2e} "
                      f"max|dagent|={d_agents:.2e} edges_match={same_edges} "
                      f"-> {rows[-1]['verdict']}", flush=True)
    finally:
        cur.WEIGHT_BATCH, cur.USE_NETWORK_CACHE = saved
    return pd.DataFrame(rows)


# %% reporting
def report_ensemble(runs_df, stats_df):
    print("\n" + "=" * 78)
    print("ENSEMBLE STAGE (production settings; distributional equivalence)")
    print("=" * 78)
    cols = ["mean_legacy", "mean_current", "diff", "cohens_d",
            "p_welch_holm", "p_ks_holm", "p_tost", "verdict"]
    for scenario in stats_df["scenario"].unique():
        sub = stats_df[stats_df["scenario"] == scenario].set_index("metric")
        print(f"\n{scenario}")
        print(sub[cols].to_string(float_format=lambda v: f"{v:.4g}"))

    speed = runs_df.groupby(["scenario", "impl"])["seconds"].mean().unstack()
    if {"legacy", "current"} <= set(speed.columns):
        speed["speedup"] = speed["legacy"] / speed["current"]
        print("\nWall time per run (s) and speedup at this N/steps:")
        print(speed.round(2).to_string())

    n_diff = (stats_df["verdict"] == "DIFFERENT").sum()
    n_equiv = (stats_df["verdict"] == "EQUIVALENT").sum()
    n_incon = (stats_df["verdict"] == "INCONCLUSIVE").sum()
    print(f"\nSummary: {n_equiv} equivalent, {n_incon} inconclusive, {n_diff} different "
          f"(of {len(stats_df)} scenario x metric tests)")
    if n_diff:
        print("\nCells flagged DIFFERENT (inspect these):")
        print(stats_df[stats_df["verdict"] == "DIFFERENT"][
            ["scenario", "metric", "mean_legacy", "mean_current", "diff",
             "cohens_d", "p_welch_holm", "p_ks_holm"]].to_string(index=False))
    if n_incon:
        print("INCONCLUSIVE = no difference detected, but the ensemble is too small "
              "to certify equivalence at the chosen margin; raise --runs.")
    return n_diff


def report_exact(exact_df):
    print("\n" + "=" * 78)
    print("EXACT STAGE (seed-matched, WEIGHT_BATCH=1, cache off)")
    print("=" * 78)
    if exact_df.empty:
        print("no deterministic scenarios selected")
        return 0
    per_scenario = exact_df.groupby("scenario").agg(
        runs=("run", "count"), max_dstate=("max_dstate", "max"),
        max_dagent=("max_dagent", "max"), all_edges_match=("identical_edges", "all"),
        failures=("verdict", lambda v: (v == "FAIL").sum()))
    print(per_scenario.to_string())
    n_fail = int((exact_df["verdict"] == "FAIL").sum())
    print(f"\n{len(exact_df) - n_fail}/{len(exact_df)} paired runs bit-identical")
    return n_fail


# %% main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["exact", "ensemble", "both"], default="both")
    ap.add_argument("--runs", type=int, default=24, help="runs per implementation (ensemble)")
    ap.add_argument("--exact-runs", type=int, default=2, help="paired runs per scenario (exact)")
    ap.add_argument("--n", type=int, default=200, help="network size (small is fine here)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--topology", default="cl", help="cl, cl_nh, DPAH, sf, rand")
    ap.add_argument("--scenarios", help="comma-separated algo/mode, e.g. bridge/diff,wtf/None")
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (0 = codebase default)")
    # parameter point: the heatmap corrections moved mainly the high-stubbornness
    # rows, so it is worth being able to re-test away from the campaign baseline
    ap.add_argument("--stubbornness", type=float, help="override s (default: model baseline)")
    ap.add_argument("--phi", type=float, help="override politicalClimate")
    ap.add_argument("--pnf", type=float, help="override polarisingNode_f")
    ap.add_argument("--tost-sd", type=float, default=0.5,
                    help="equivalence margin in pooled SDs (floored per metric)")
    ap.add_argument("--tol", type=float, default=1e-9, help="exact-stage float tolerance")
    ap.add_argument("--legacy-rev", default=LEGACY_REV, help="git rev of the pre-refactor model")
    ap.add_argument("--smoke", action="store_true", help="fast end-to-end check")
    ap.add_argument("--outdir", default="../Output")
    a = ap.parse_args()

    if a.smoke:
        a.runs, a.exact_runs, a.n, a.steps = 6, 1, 100, 600
        if not a.scenarios:
            a.scenarios = "None/None,biased/diff,wtf/None"

    scenarios = SCENARIOS
    if a.scenarios:  # "algo/mode", or bare "algo" for the modeless algorithms
        scenarios = [tuple((s.split("/") + ["None"])[:2]) for s in a.scenarios.split(",")]

    os.chdir(ANALYSIS_DIR)  # the model resolves data paths relative to Analysis/
    from general_param_sweep import get_optimal_process_count
    import models_checks as current
    legacy, legacy_path = load_legacy(a.legacy_rev)
    IMPLS["current"], IMPLS["legacy"] = current, legacy

    jobs = a.jobs if a.jobs > 0 else max(1, get_optimal_process_count())
    base_args = dict(current.getargs())  # copy: getargs hands back the module's own dict
    if a.stubbornness is not None:
        base_args["stubbornness"] = a.stubbornness
    if a.phi is not None:
        base_args["politicalClimate"] = base_args["newPoliticalClimate"] = a.phi
    if a.pnf is not None:
        base_args["polarisingNode_f"] = a.pnf
    stamp = date.today()

    print("=" * 78)
    print("MODEL OPTIMIZATION EQUIVALENCE TEST" + ("  (smoke)" if a.smoke else ""))
    print("=" * 78)
    print(f"legacy : {a.legacy_rev} -> {legacy_path}")
    print(f"current: Analysis/models_checks.py"
          f"{'' if working_tree_is_optimized() else '  [MODIFIED vs ' + OPT_COMMIT + ']'}")
    print(f"topology={a.topology} N={a.n} steps={a.steps} "
          f"stubbornness={base_args['stubbornness']} phi={base_args['politicalClimate']} "
          f"pNf={base_args['polarisingNode_f']}")
    print(f"scenarios: {', '.join(f'{s}/{m}' for s, m in scenarios)}")

    failures = 0
    if a.mode in ("exact", "both"):
        print(f"\nExact stage: {a.exact_runs} paired run(s) per deterministic scenario")
        exact_df = run_exact(scenarios, base_args, a.exact_runs, a.topology,
                             a.n, a.steps, a.tol)
        if not exact_df.empty:
            path = f"{a.outdir}/optimization_equivalence_exact_{stamp}.csv"
            exact_df.to_csv(path, index=False)
            failures += report_exact(exact_df)
            print(f"-> {path}")

    if a.mode in ("ensemble", "both"):
        print(f"\nEnsemble stage: {a.runs} runs per implementation, {jobs} process(es)")
        runs_df = run_ensemble(scenarios, base_args, a.runs, a.topology,
                               a.n, a.steps, jobs)
        stats_df = compare(runs_df, a.tost_sd)
        p_runs = f"{a.outdir}/optimization_equivalence_runs_{stamp}.csv"
        p_stats = f"{a.outdir}/optimization_equivalence_stats_{stamp}.csv"
        runs_df.to_csv(p_runs, index=False)
        stats_df.to_csv(p_stats, index=False)
        failures += report_ensemble(runs_df, stats_df)
        print(f"-> {p_runs}\n-> {p_stats}")

    print("\n" + ("VERDICT: no evidence that the optimization changed the model."
                  if failures == 0 else
                  f"VERDICT: {failures} check(s) flagged - see the tables above."))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
