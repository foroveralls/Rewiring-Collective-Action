#!/usr/bin/env python3
"""
LOW-MEMORY extractor for PER-RUN **inflection** convergence rates from the
full-resolution "individual" campaign CSV.

WHY: reviewer R1.7 asks for uncertainty on the main figures. Fig. 3 (the Pareto
plot, `Analysis/Plotting/convergence_vs_cooperation.py`) gained IQR error bars in
June 2026, but the error-bar branch is gated on `method == 't95'` because the only
per-run file that existed, `Output/per_run_summary.csv`, stores `speed_t95` and
nothing else. The submitted manuscript's Fig. 3 is the **inflection** metric
(`-v2.tex` L110 loads `pareto_speed_cooperativity_inflection_2026-03-27.pdf`), so
adding the error bars silently swapped the figure's speed metric. Reverting to
inflection means the error bars need per-run *inflection* rates, which have never
been computed. This script computes them. Full context, cluster command and
smoke-test numbers: `claude_stuff/Review/per_run_inflection_2026-07-30.md`.

FULL RESOLUTION IS NON-NEGOTIABLE -- do not "optimise" this into the existing
downsampling pass. `Analysis/Stats/summarize_individual_csv.py` streams the same
file keeping only `t % 100 == 0`, i.e. ~1,600 points per run. `find_inflection`
requires its inflection index to satisfy `5000 < i < 20000` in **raw array-index**
units, which a 1,600-point series can never reach: it returned `False` for 8/8
test conditions and `calculate_inflection_convergence_speed` then silently returns
0.0. Rescaling the two thresholds by the downsample factor recovers the inflection
*location* almost exactly (5693 -> 5600, 6520 -> 6500) but shifts the *rate* by
+/-30% (0.107 -> 0.074, 0.040 -> 0.052), because `regwin=10` then spans +/-1,000 raw
steps instead of +/-10. So the rate can only be computed on the untouched
full-resolution series.

METRIC FIDELITY: `find_inflection` and `estimate_convergence_rate` are **imported**
from `Analysis/Plotting/convergence_vs_cooperation.py` (by explicit path, because
`Plotting/` is not a package and this script lives in `Analysis/`), so the per-run
rate cannot drift from the ensemble rate the figure plots. The 3-line wrapper
`calculate_inflection_convergence_speed` is inlined rather than called, only to
avoid running the 600-sigma Gaussian filter twice per run; `--verify-wrapper K`
cross-checks the inlined path against the real wrapper on the first K runs and
prints both. The odd bits of the published definition are deliberately preserved:
`gaussian_filter1d(seq, 600)`, the absolute 5000/20000 index bounds, `regwin=10`,
the `*1000` scaling, and `rate = -b1 / (traj[loc] - 1)` (that denominator is a
normalisation toward a = 1; it is strange but it is what is published).

REGWIN DIAGNOSTIC (added 2026-07-31, and the reason to run this at all now):
`estimate_convergence_rate` fits a straight line over `traj[loc-regwin : loc+regwin+1]`,
i.e. **21 timesteps out of 160,000** at the published `regwin=10`. On the ENSEMBLE
trajectory that is fine -- `avg_state` is already a mean over 90 runs, so the noise
is gone before the regression sees it. On a SINGLE run it is a 21-step slope of a
raw stochastic series, and the smoke test found **23% of per-run rates coming out
negative** with a per-run IQR (~0.18) that is 2-3x the entire spread of the 30
ensemble values (0.089). If that spread is estimator noise rather than run-to-run
variation, then drawing it -- as an error bar on Fig. 3 or, worse, as a violin body
in the main text -- misrepresents uncertainty instead of disclosing it, which is the
opposite of what R1.7 asks for.

So `--regwins` recomputes the rate at several window widths in the SAME pass (the
cost here is the sequential read, not the regression) and the `_META.txt` sidecar
prints, per regwin, the fraction of negative rates and the median per-condition IQR.
**Read those two lines to decide**: if both collapse as the window widens, the spread
was estimator noise; if the IQR holds roughly flat, it is genuine run-to-run
variation and belongs in the figure.

**`speed_inflection` is always the published `regwin=10` value and must stay the
canonical column** -- the extra `speed_inflection_rw*` columns are diagnostics. Do
NOT swap a wider window into Fig. 3's ensemble rate without deciding that
deliberately: it shifts the plotted rates and can move ranks and the Pareto front.
Using a wider window for the PER-RUN estimate only is defensible (the ensemble
series has been averaged, a single run has not) but must be stated in the caption.

MEMORY: chunked read, and a run's trajectory is buffered only while its rows are
still arriving. A finished run is detected as "key absent from this chunk", which
is safe for any block-contiguous layout (verified: the pte rerun has each run in
one ascending contiguous block) and is checked anyway -- every row carries
`t_complete`, and a key that re-opens after being flushed is reported loudly.
One run at 270k steps is ~4 MB of buffer, so peak RSS is dominated by the pandas
chunk, not the buffers.

OUTPUT: one row per (type, scenario, rewiring, model_run), written and flushed as
each run completes, so a killed job still leaves usable partial output. There is
no --resume: the cost here is the sequential scan of the input, which resuming
cannot skip. A `_META.txt` sidecar (named after --out) records input path + mtime,
row counts, runtime, and the inflection-failure tally.

  speed_inflection      published rate (regwin=10), or **NaN** when no inflection
                        is found (use this for the IQR; NaN runs must be excluded
                        and the count disclosed)
  speed_inflection_rw*  the same rate at each --regwins width; `_rw10` duplicates
                        speed_inflection exactly (asserted at write time) so a
                        downstream loop can treat all widths uniformly. DIAGNOSTIC
                        ONLY -- see the REGWIN section above
  speed_inflection_raw  exactly what `calculate_inflection_convergence_speed`
                        returns, i.e. 0.0 on failure -- kept for fidelity/audit,
                        NOT for the IQR (a 0.0 masquerades as "slowest")
  inflection_found      1/0
  inflection_idx        raw array index of the inflection, -1 if none
  inflection_value      trajectory[idx], the `-b1/(x-1)` denominator's input
  slope_b1              raw regression slope before the /(x-1) and *1000
  speed_t95_fullres     the t95 metric on the FULL-resolution series, for
                        cross-checking against per_run_summary.csv (which has it
                        at step=100)
  cooperativity         mean of the last 10% of the trajectory (same definition
                        as `calculate_metrics`), so Fig. 3 can take both axes
                        from this one file
  n_points,t_min,t_max,t_complete,n_dup_t,max_dt   series-integrity diagnostics

Cluster run on the merged gme campaign (~28 GB, 30 conditions x 90 runs).
The default --regwins is what the R1.7 decision needs; it costs no extra I/O:
  cd Analysis && /home/jpoveralls/miniconda3/envs/collective_rewiring/bin/python \
      extract_inflection_lowmem.py \
      --in ../Output/default_run_individual_N_800_n_90_pNf_0_pc_0.05_sweep_20251014_1704_phased_run_gme_2025-10-15.csv \
      --out ../Output/per_run_inflection.csv

Then read the REGWIN SPREAD DIAGNOSTIC block at the end of
../Output/per_run_inflection_META.txt before plotting anything.

Local smoke test (1.5 GB pte rerun: bridge/diff/FB only, 90 runs at 270k):
  cd Analysis && python extract_inflection_lowmem.py \
      --in ../Output/default_run_individual_N_800_n_90_pNf_0_pc_0.05_sweep_20260724_1637_phased_run_pte_2026-07-24.csv \
      --out ../Output/per_run_inflection_SMOKE.csv
"""
import argparse
import csv
import importlib.util
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("MPLBACKEND", "Agg")  # the imported plotting module pulls in pyplot

import numpy as np
import pandas as pd

# Derived, not hardcoded: this script also runs on the cluster, where an absolute
# local path would not exist.
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "..", "Output")
CVC_PATH = os.path.join(ANALYSIS_DIR, "Plotting", "convergence_vs_cooperation.py")

KEY_COLS = ["type", "scenario", "rewiring", "model_run"]
USECOLS = ["t", "avg_state", "model_run", "scenario", "rewiring", "type"]
OUT_FIELDS = KEY_COLS + [
    "speed_inflection", "speed_inflection_raw", "inflection_found", "inflection_idx",
    "inflection_value", "slope_b1", "speed_t95_fullres", "cooperativity",
    "n_points", "t_min", "t_max", "t_complete", "n_dup_t", "max_dt",
]
CANONICAL_REGWIN = 10   # the published window; speed_inflection is always this one


def regwin_col(rw):
    return f"speed_inflection_rw{rw}"


def load_cvc(path):
    """Import Analysis/Plotting/convergence_vs_cooperation.py by explicit path.

    Plotting/ is not a package, so a normal import is not available from here.
    Importing (rather than copying) is the point: the per-run rate must be the
    same function the figure applies to the ensemble mean.
    """
    spec = importlib.util.spec_from_file_location("_cvc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("find_inflection", "estimate_convergence_rate",
               "calculate_inflection_convergence_speed", "calculate_t95_convergence_speed"):
        if not hasattr(mod, fn):
            raise RuntimeError(f"{path} has no {fn}() -- the metric definition moved")
    return mod


def _rate_at(cvc, y, loc, regwin):
    """Published rate at an arbitrary regression half-width, or NaN.

    Guards only the array bounds. Everything else is left exactly as published:
    `estimate_convergence_rate` builds `x = np.arange(len(trajec) - 1)` and slices
    both x and y by the same window, so x is one element short of y -- preserved on
    purpose, it is the definition Fig. 3 uses.
    """
    if loc - regwin < 0 or loc + regwin + 1 > y.size - 1:
        return np.nan
    try:
        r = cvc.estimate_convergence_rate(y, loc, regwin=regwin) * 1000.0
    except Exception:
        return np.nan
    return float(r) if np.isfinite(r) else np.nan


def run_metrics(cvc, t, y, verify_wrapper=False, regwins=()):
    """Per-run metrics for one full-resolution trajectory (t ascending, y aligned)."""
    n = t.size
    dt = np.diff(t) if n > 1 else np.array([0])
    t_min, t_max = int(t[0]), int(t[-1])
    n_dup = int((dt == 0).sum())
    out = {
        "n_points": int(n), "t_min": t_min, "t_max": t_max,
        "n_dup_t": n_dup, "max_dt": int(dt.max()),
        "t_complete": int(n == (t_max - t_min + 1) and n_dup == 0),
    }

    win = max(1, int(n * 0.1))
    out["cooperativity"] = float(np.mean(y[-win:]))
    out["speed_t95_fullres"] = float(cvc.calculate_t95_convergence_speed(y))

    loc = cvc.find_inflection(y)
    if loc:
        out["inflection_found"] = 1
        out["inflection_idx"] = int(loc)
        out["inflection_value"] = float(y[loc])
        try:
            rate = cvc.estimate_convergence_rate(y, loc) * 1000.0
        except Exception:
            rate = np.nan
        if np.isnan(rate):
            # the wrapper's own fallback; the inflection was found but the fit failed
            out["speed_inflection"] = np.nan
            out["speed_inflection_raw"] = 0.0
            out["slope_b1"] = np.nan
        else:
            out["speed_inflection"] = float(rate)
            out["speed_inflection_raw"] = float(rate)
            out["slope_b1"] = float(-rate / 1000.0 * (y[loc] - 1.0))
    else:
        out["inflection_found"] = 0
        out["inflection_idx"] = -1
        out["inflection_value"] = np.nan
        out["slope_b1"] = np.nan
        out["speed_inflection"] = np.nan   # excluded from the IQR
        out["speed_inflection_raw"] = 0.0  # what the published wrapper returns

    # Diagnostic widths. No inflection means no rate at any width.
    for rw in regwins:
        out[regwin_col(rw)] = _rate_at(cvc, y, loc, rw) if loc else np.nan
    if CANONICAL_REGWIN in regwins and loc:
        # the published column and its diagnostic twin must be the same number
        a, b = out["speed_inflection"], out[regwin_col(CANONICAL_REGWIN)]
        if not (np.isnan(a) and np.isnan(b)) and not np.isclose(a, b, rtol=0, atol=0):
            raise RuntimeError(
                f"regwin={CANONICAL_REGWIN} diverged from the published path: "
                f"{b!r} vs {a!r} -- the metric definition moved")

    if verify_wrapper:
        ref = float(cvc.calculate_inflection_convergence_speed(y))
        out["_wrapper_ref"] = ref
        out["_wrapper_match"] = bool(np.isclose(ref, out["speed_inflection_raw"],
                                                rtol=0, atol=0))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Per-run inflection convergence rates from the full-resolution individual CSV.")
    ap.add_argument("--in", dest="inp", required=True,
                    help="individual (per-run) campaign CSV; required on purpose, so the "
                         "campaign is a deliberate choice (Figs 2/3 = the merged gme file)")
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "per_run_inflection.csv"),
                    help="per-run CSV to write; the _META.txt sidecar follows its name")
    ap.add_argument("--chunk", type=int, default=4_000_000,
                    help="rows per read_csv chunk (memory knob; 4M ~ 350 MB peak)")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="stop after roughly this many input rows (0 = whole file); for quick "
                         "smoke tests only. The runs still open at the cut are written from a "
                         "TRUNCATED series -- discard the last few rows of that output")
    ap.add_argument("--verify-wrapper", type=int, default=3, metavar="K",
                    help="cross-check the inlined rate against "
                         "calculate_inflection_convergence_speed() on the first K runs (0 = off)")
    ap.add_argument("--regwins", default="10,50,200,1000",
                    help="comma-separated regression half-widths to ALSO emit, as "
                         "speed_inflection_rw<N>, for the estimator-noise check (see the "
                         "REGWIN section of the docstring). Costs no extra I/O. "
                         "'' disables. speed_inflection always stays regwin=10")
    args = ap.parse_args()

    regwins = sorted({int(x) for x in args.regwins.split(",") if x.strip()})
    bad = [r for r in regwins if r < 1]
    if bad:
        ap.error(f"--regwins must be positive, got {bad}")
    out_fields = OUT_FIELDS + [regwin_col(r) for r in regwins]

    inp = os.path.abspath(args.inp)
    out_csv = os.path.abspath(args.out)
    meta_txt = os.path.splitext(out_csv)[0] + "_META.txt"

    t0 = time.time()
    cvc = load_cvc(CVC_PATH)
    in_bytes = os.path.getsize(inp)
    in_mtime = datetime.fromtimestamp(os.path.getmtime(inp)).isoformat(timespec="seconds")
    print(f"[{0.0:7.1f}s] metric functions imported from {CVC_PATH}", flush=True)
    print(f"[{0.0:7.1f}s] reading {inp}  ({in_bytes/1e9:.2f} GB, mtime {in_mtime}, "
          f"chunk={args.chunk:,})", flush=True)
    print(f"[{0.0:7.1f}s] diagnostic regwins: {regwins or '(none)'}  "
          f"(published metric stays regwin={CANONICAL_REGWIN})", flush=True)

    buf = {}           # open key -> [(t_arr, y_arr), ...]
    flushed = set()
    reopened = []
    truncated = False
    rows_in = 0
    runs_out = 0
    fails = []
    incomplete = []
    wrapper_checks = []

    rate_rows = []     # (key, {regwin: rate}) for the REGWIN SPREAD DIAGNOSTIC

    fout = open(out_csv, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=out_fields, extrasaction="ignore")
    writer.writeheader()

    def flush_key(key):
        nonlocal runs_out
        parts = buf.pop(key)
        t = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        order = np.argsort(t, kind="stable")   # never assume the source is sorted
        t, y = t[order], y[order]
        verify = args.verify_wrapper > 0 and len(wrapper_checks) < args.verify_wrapper
        m = run_metrics(cvc, t, y, verify_wrapper=verify, regwins=regwins)
        row = dict(zip(KEY_COLS, key))
        row.update(m)
        rate_rows.append((key[:3], {rw: m.get(regwin_col(rw), np.nan) for rw in regwins}))
        writer.writerow(row)
        fout.flush()
        runs_out += 1
        flushed.add(key)
        if verify:
            wrapper_checks.append((key, m["_wrapper_ref"], m["speed_inflection_raw"],
                                   m["_wrapper_match"]))
            print(f"[{time.time()-t0:7.1f}s]   wrapper check {key}: "
                  f"inlined={m['speed_inflection_raw']!r} wrapper={m['_wrapper_ref']!r} "
                  f"match={m['_wrapper_match']}", flush=True)
        if not m["inflection_found"]:
            fails.append(key)
        if not m["t_complete"]:
            incomplete.append((key, m["n_points"], m["t_min"], m["t_max"],
                               m["n_dup_t"], m["max_dt"]))
        print(f"[{time.time()-t0:7.1f}s] run {runs_out:5d}  {'/'.join(map(str, key))}  "
              f"n={m['n_points']:,} t=[{m['t_min']},{m['t_max']}] "
              f"infl={m['inflection_idx']} rate={m['speed_inflection']} "
              f"coop={m['cooperativity']:+.4f}", flush=True)

    reader = pd.read_csv(
        inp, chunksize=args.chunk, usecols=USECOLS,
        dtype={"t": "int64", "avg_state": "float64", "model_run": "int64",
               "scenario": "category", "rewiring": "category", "type": "category"},
        keep_default_na=False,   # keeps the literal string "None" as "None", like the source
    )

    for ci, ch in enumerate(reader):
        rows_in += len(ch)
        seen = set()
        for key, sub in ch.groupby(KEY_COLS, sort=False, observed=True):
            key = (str(key[0]), str(key[1]), str(key[2]), int(key[3]))
            seen.add(key)
            if key in flushed:
                reopened.append(key)
                print(f"[{time.time()-t0:7.1f}s] !! WARNING key re-opened after flush: "
                      f"{key} -- input is NOT block-contiguous; its row is only a partial "
                      f"series. Re-run with a larger --chunk or sort the input.", flush=True)
                continue
            buf.setdefault(key, []).append(
                (sub["t"].to_numpy(copy=True), sub["avg_state"].to_numpy(copy=True)))
        for key in [k for k in buf if k not in seen]:
            flush_key(key)
        print(f"[{time.time()-t0:7.1f}s] chunk {ci+1}: rows_in={rows_in:,} "
              f"({rows_in/max(1e-9, time.time()-t0):,.0f} rows/s) open_runs={len(buf)} "
              f"runs_done={runs_out}", flush=True)
        if args.max_rows and rows_in >= args.max_rows:
            truncated = True
            print(f"[{time.time()-t0:7.1f}s] --max-rows reached; stopping early", flush=True)
            break

    for key in sorted(buf):   # tail
        flush_key(key)
    fout.close()

    elapsed = time.time() - t0
    n_fail = len(fails)
    meta = [
        f"input:            {inp}",
        f"input size:       {in_bytes:,} bytes ({in_bytes/1e9:.2f} GB)",
        f"input mtime:      {in_mtime}",
        f"output:           {out_csv}",
        f"generated:        {datetime.now().isoformat(timespec='seconds')}",
        f"metric source:    {CVC_PATH} (find_inflection + estimate_convergence_rate, imported)",
        f"resolution:       FULL (no downsampling -- required, see script docstring)",
        f"chunk:            {args.chunk:,} rows",
        f"rows read:        {rows_in:,}",
        f"runs written:     {runs_out}",
        f"runtime:          {elapsed:.1f} s ({elapsed/60:.1f} min); "
        f"{rows_in/max(1e-9, elapsed):,.0f} rows/s"
        + ("  (--max-rows truncated the read, so no MB/s figure)" if truncated else
           f", {in_bytes/1e6/max(1e-9, elapsed):,.1f} MB/s"),
        f"truncated by --max-rows: {truncated}"
        + (" -> the runs still open at the cut were written from a PARTIAL series" if truncated else ""),
        f"inflection FAILED: {n_fail} / {runs_out} runs "
        f"({100.0*n_fail/max(1, runs_out):.1f}%) -> speed_inflection = NaN "
        f"(speed_inflection_raw = 0.0, matching the published wrapper)",
        f"incomplete series: {len(incomplete)} (t_complete = 0)",
        f"keys re-opened after flush: {len(reopened)} (non-zero => input not block-contiguous, "
        f"affected rows are partial series)",
    ]
    if regwins:
        meta += ["", "=" * 72,
                 "REGWIN SPREAD DIAGNOSTIC -- read this before drawing any per-run",
                 "convergence bar or violin (R1.7). See the REGWIN section of the docstring.",
                 "=" * 72,
                 "",
                 "The question: is the per-run spread real run-to-run variation, or is it",
                 "noise from fitting a straight line over 2*regwin+1 steps of a single",
                 "stochastic trajectory? Compare against the spread of the 30 ENSEMBLE",
                 "values, which is 0.0890 end to end (inflection, merged gme campaign).",
                 "",
                 "  If %neg and median IQR both COLLAPSE as regwin grows -> estimator noise.",
                 "    Do not draw the regwin=10 per-run spread. Either report it as a number",
                 "    with the caveat, or use a wider window for the PER-RUN estimate only",
                 "    and say so in the caption.",
                 "  If median IQR stays roughly FLAT -> genuine run-to-run variation.",
                 "    It belongs in the figure; the violin is the right instrument for it.",
                 "",
                 f"{'regwin':>8}  {'n valid':>8}  {'% neg':>7}  {'median':>9}  "
                 f"{'med cond IQR':>13}  {'vs 0.0890':>10}",
                 "-" * 72]
        ENSEMBLE_SPREAD = 0.0890
        for rw in regwins:
            vals = np.array([d[rw] for _, d in rate_rows], dtype=float)
            ok = np.isfinite(vals)
            n_ok = int(ok.sum())
            if not n_ok:
                meta.append(f"{rw:>8}  {0:>8}  {'--':>7}  {'--':>9}  {'--':>13}  {'--':>10}")
                continue
            pct_neg = 100.0 * float((vals[ok] < 0).sum()) / n_ok
            med = float(np.median(vals[ok]))
            # per-condition IQR, then the median over conditions: one condition's
            # dispersion is what a single violin body / error bar would show
            per_cond = {}
            for (cond, d) in rate_rows:
                v = d[rw]
                if np.isfinite(v):
                    per_cond.setdefault(cond, []).append(v)
            iqrs = [float(np.percentile(v, 75) - np.percentile(v, 25))
                    for v in per_cond.values() if len(v) >= 10]
            med_iqr = float(np.median(iqrs)) if iqrs else float("nan")
            meta.append(f"{rw:>8}  {n_ok:>8}  {pct_neg:>6.1f}%  {med:>9.5f}  "
                        f"{med_iqr:>13.5f}  {med_iqr/ENSEMBLE_SPREAD:>9.2f}x")
        meta += ["-" * 72,
                 "'med cond IQR' = median over conditions of the per-run IQR within a",
                 "condition. 'vs 0.0890' is that as a multiple of the full ensemble spread;",
                 "at regwin=10 the smoke test put it at 2-3x, which is what saturates Fig. 3's",
                 "rank-space error bars and would dominate a violin body.",
                 "",
                 f"NOTE: speed_inflection is regwin={CANONICAL_REGWIN} and is the published",
                 "metric. The rw columns are diagnostics. Changing Fig. 3's ENSEMBLE rate to a",
                 "wider window shifts the plotted rates and can move ranks and the Pareto",
                 "front -- that is a separate, deliberate decision.",
                 ""]

    if wrapper_checks:
        meta.append("wrapper cross-checks (inlined vs calculate_inflection_convergence_speed):")
        meta += [f"  {k}: wrapper={w!r} inlined={i!r} match={m}" for k, w, i, m in wrapper_checks]
    if fails:
        meta.append("failed runs (no inflection in 5000 < i < 20000):")
        meta += [f"  {'/'.join(map(str, k))}" for k in fails]
    if incomplete:
        meta.append("incomplete series (key, n_points, t_min, t_max, n_dup_t, max_dt):")
        meta += [f"  {'/'.join(map(str, k))} {n} {a} {b} {d} {g}"
                 for k, n, a, b, d, g in incomplete]
    if reopened:
        meta.append(f"re-opened keys: {sorted(set(reopened))}")

    with open(meta_txt, "w") as mf:
        mf.write("\n".join(meta) + "\n")

    print("\n".join(meta), flush=True)
    print(f"[{elapsed:7.1f}s] DONE runs={runs_out} -> {out_csv}", flush=True)
    if reopened:
        sys.exit(3)


if __name__ == "__main__":
    main()
