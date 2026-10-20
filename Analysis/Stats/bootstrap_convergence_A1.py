#!/usr/bin/env python3
"""
Bootstrap CIs for the ensemble convergence metrics of Fig. 3 (round-2 R1b.3, analysis A1).

For each (type, scenario, rewiring) condition of the canonical default campaign, resample
runs with replacement (B = 1000), recompute the ensemble mean trajectory <a>(t) and from it:

  rate     the published Fig. 3 inflection rate, exactly as `convergence_vs_cooperation.py`
           computes it on the avg file: Gaussian smoothing (sigma 600), first sign change of
           the second derivative with 5000 < i < 20000, OLS slope over +/-10 raw steps, then
           rate = -slope / (<a>(t_infl) - 1) * 1000. Resamples with no inflection are counted
           as failures, not as 0.0 (the published wrapper returns 0.0).
  t_half   first step at which <a> has covered half the net change, from <a>(0) to the
           mean of the last 10% of the trajectory (the published final-opinion window).
  t_zero   first step with <a> > 0; NaN if never.
  a_final  mean of the last 10% of <a>(t), the published final mean opinion.

Two stages, because the input is 28 GB and the cost is the sequential read:

  --extract    stream the individual CSV once, write one float32 .npy per run to --cache
  --bootstrap  per condition, load runs, bootstrap, write the summary CSV + draws .npz

Smoothing and differencing are linear, so they are applied once per run and the bootstrap
averages the smoothed runs (smooth(mean) == mean(smooth)); --check verifies this and that the
full-sample point estimate reproduces the avg-file value.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Plotting'))
from convergence_vs_cooperation import calculate_inflection_convergence_speed  # noqa: E402

SIGMA = 600
RADIUS = int(4.0 * SIGMA + 0.5)        # scipy's default truncate=4.0
INF_MIN, INF_MAX = 5000, 20000
REGWIN = 10
# columns of the smoothed series kept for the inflection search: the diff index i covers
# INF_MIN..INF_MAX, and np.gradient twice needs 2 neighbours on each side
LO, HI = INF_MIN - 4, INF_MAX + 4


def key_dir(cache, key):
    return os.path.join(cache, '__'.join(key))


def extract(path, cache, chunk):
    os.makedirs(cache, exist_ok=True)
    cols = ['t', 'avg_state', 'model_run', 'scenario', 'rewiring', 'type']
    open_runs, flushed, n_rows, t0 = {}, set(), 0, time.time()

    def flush(k):
        t = np.concatenate(open_runs[k]['t'])
        a = np.concatenate(open_runs[k]['a'])
        if not np.array_equal(t, np.arange(len(t))):
            raise SystemExit(f'non-contiguous t in {k}')
        cond, run = k[:3], k[3]
        os.makedirs(key_dir(cache, cond), exist_ok=True)
        np.save(os.path.join(key_dir(cache, cond), f'run_{run:03d}.npy'), a.astype(np.float32))
        flushed.add(k)
        del open_runs[k]

    reader = pd.read_csv(path, usecols=cols, chunksize=chunk, keep_default_na=False,
                         dtype={'t': np.int64, 'avg_state': np.float64, 'model_run': np.int64,
                                'scenario': str, 'rewiring': str, 'type': str})
    for df in reader:
        n_rows += len(df)
        seen = set()
        for (typ, sc, rw, run), g in df.groupby(['type', 'scenario', 'rewiring', 'model_run'],
                                                sort=False):
            k = (typ, sc, rw, int(run))
            if k in flushed:
                raise SystemExit(f'key re-opened after flush: {k}')
            seen.add(k)
            buf = open_runs.setdefault(k, {'t': [], 'a': []})
            buf['t'].append(g['t'].to_numpy())
            buf['a'].append(g['avg_state'].to_numpy())
        for k in [k for k in open_runs if k not in seen]:
            flush(k)
        print(f'{n_rows:>12,} rows  {len(flushed):>5} runs  {time.time() - t0:7.0f} s', flush=True)
    for k in list(open_runs):
        flush(k)
    print(f'done: {n_rows:,} rows, {len(flushed)} runs, {time.time() - t0:.0f} s')


def gauss_kernel():
    x = np.arange(-RADIUS, RADIUS + 1)
    k = np.exp(-0.5 * (x / SIGMA) ** 2)
    return k / k.sum()


def smooth_window(R):
    """gaussian_filter1d(R, 600, axis=1) (mode='reflect'), columns LO..HI only."""
    a, b = LO - RADIUS, HI + RADIUS
    if a < 0:
        pad = np.pad(R, ((0, 0), (-a, 0)), mode='symmetric')[:, :b - a]
    else:
        pad = R[:, a:b]
    return fftconvolve(pad, gauss_kernel()[None, :], mode='valid', axes=1)


def metrics(M, MS):
    """Metrics for each row of M (raw ensemble means) and MS (smoothed, columns LO..HI)."""
    nB, T = M.shape
    d2 = np.gradient(np.gradient(MS, axis=1), axis=1)
    # published: infls = where(diff(sign(d2))); first i >= 5000; accept if 5000 < i < 20000
    i0 = INF_MIN - LO
    ch = np.diff(np.sign(d2), axis=1)[:, i0:i0 + (INF_MAX - INF_MIN)] != 0
    has = ch.any(axis=1)
    loc = np.where(has, ch.argmax(axis=1) + INF_MIN, -1)
    ok = has & (loc > INF_MIN) & (loc < INF_MAX)

    rate = np.full(nB, np.nan)
    x = np.arange(-REGWIN, REGWIN + 1, dtype=float)
    for j in np.flatnonzero(ok):
        y = M[j, loc[j] - REGWIN: loc[j] + REGWIN + 1]
        b1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
        rate[j] = -b1 / (M[j, loc[j]] - 1) * 1000

    a0 = M[:, 0]
    af = M[:, -int(T * 0.1):].mean(axis=1)
    target = a0 + 0.5 * (af - a0)
    hit = np.sign(af - a0)[:, None] * (M - target[:, None]) >= 0
    t_half = hit.argmax(axis=1).astype(float)

    pos = M > 0
    t_zero = np.where(pos.any(axis=1), pos.argmax(axis=1), np.nan).astype(float)
    return rate, np.where(ok, loc, np.nan), t_half, t_zero, af


def load(cache, cond):
    d = os.path.join(cache, cond)
    files = sorted(f for f in os.listdir(d) if f.endswith('.npy'))
    runs = [np.load(os.path.join(d, f)).astype(np.float64) for f in files]
    if len({len(r) for r in runs}) != 1:
        raise SystemExit(f'{cond}: runs of unequal length')
    return np.vstack(runs)


def ci(v):
    v = v[np.isfinite(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def bootstrap(cache, out, B, seed, batch, avg_path):
    avg = None
    if avg_path:
        avg = pd.read_csv(avg_path, keep_default_na=False,
                          dtype={'scenario': str, 'rewiring': str, 'type': str})
    rng = np.random.default_rng(seed)
    rows, draws = [], {}
    for cond in sorted(os.listdir(cache)):
        t0 = time.time()
        typ, sc, rw = cond.split('__')
        R = load(cache, cond)
        n, T = R.shape
        S = smooth_window(R)
        point = [m[0] for m in metrics(R.mean(0, keepdims=True), S.mean(0, keepdims=True))]

        res = [[] for _ in range(5)]
        for s in range(0, B, batch):
            W = rng.multinomial(n, np.full(n, 1.0 / n), size=min(batch, B - s)) / n
            for acc, m in zip(res, metrics(W @ R, W @ S)):
                acc.append(m)
        rate, t_inf, t_half, t_zero, af = (np.concatenate(r) for r in res)
        draws[cond] = np.vstack([rate, t_inf, t_half, t_zero, af])

        row = dict(type=typ, scenario=sc, rewiring=rw, n_runs=n, T=T,
                   rate=point[0], rate_lo=ci(rate)[0], rate_hi=ci(rate)[1],
                   rate_fail=np.mean(~np.isfinite(rate)), t_infl=point[1],
                   t_half=point[2], t_half_lo=ci(t_half)[0], t_half_hi=ci(t_half)[1],
                   t_zero=point[3], t_zero_lo=ci(t_zero)[0], t_zero_hi=ci(t_zero)[1],
                   t_zero_never=np.mean(~np.isfinite(t_zero)),
                   a_final=point[4], a_final_lo=ci(af)[0], a_final_hi=ci(af)[1])
        if avg is not None:
            g = avg[(avg.type == typ) & (avg.scenario == sc) & (avg.rewiring == rw)]
            row['rate_avgfile'] = calculate_inflection_convergence_speed(
                g.sort_values('t')['avg_state'].to_numpy())
        rows.append(row)
        print(f'{cond:40s} n={n} T={T} rate={point[0]:.4f} [{row["rate_lo"]:.4f}, '
              f'{row["rate_hi"]:.4f}] fail={row["rate_fail"]:.3f} '
              f'avgfile={row.get("rate_avgfile", np.nan):.4f}  {time.time() - t0:.0f} s', flush=True)

    pd.DataFrame(rows).to_csv(out, index=False)
    np.savez_compressed(os.path.splitext(out)[0] + '_draws.npz',
                        order=np.array(['rate', 't_infl', 't_half', 't_zero', 'a_final']), **draws)
    print(f'wrote {out}')


def check(cache):
    """smooth(mean) == mean(smooth) on the window, and it matches scipy's gaussian_filter1d."""
    cond = sorted(os.listdir(cache))[0]
    R = load(cache, cond)
    ref = gaussian_filter1d(R.mean(0), SIGMA)[LO:HI]
    ours = smooth_window(R).mean(0)
    print(f'{cond}: max |mean(smooth) - gaussian_filter1d(mean)| = {np.abs(ours - ref).max():.2e}')


A0 = -0.20      # initial mean opinion (models_checks.py:71)
MIN_RISE = 0.05  # conditions rising less than this from A0 have no meaningful t_half (DPA WTF)


def _speeds(g, D):
    """(name, point, draws) per speed metric, larger = faster. A time that is undefined (<a>
    never turns positive) counts as slowest, -inf, rather than being dropped."""
    def inv(t):
        return np.where(np.isfinite(t), -t, -np.inf)
    return [('rate', g.rate.values, D[:, 0]),
            ('t_half', inv(g.t_half.values), inv(D[:, 2])),
            ('t_zero', inv(g.t_zero.values), inv(D[:, 3]))]


def _load(out):
    d = pd.read_csv(out, keep_default_na=False, na_values=[''])
    z = np.load(os.path.splitext(out)[0] + '_draws.npz')
    return d, z


def _draws(z, g):
    return np.stack([z['__'.join([r.type, r.scenario, r.rewiring])] for r in g.itertuples()])


def pareto(out):
    """Per topology, with and without static: Pareto front of (speed, a_final) on the point
    estimates, the share of bootstrap resamples in which each algorithm is on the front, and the
    share in which it is the fastest. Speed = published rate, 1/t_half or 1/t_zero. Conditions are
    resampled independently, so draw b of one condition is paired with draw b of the others
    arbitrarily; that is fine for independent runs."""
    d, z = _load(out)

    def front(speed, a):
        return np.array([not np.any((speed >= speed[i]) & (a >= a[i]) & ((speed > speed[i]) | (a > a[i])))
                         for i in range(len(a))])

    rows = []
    for static in (False, True):
        for typ, g in (d if static else d[d.scenario != 'None']).groupby('type'):
            g = g.reset_index(drop=True)
            D = _draws(z, g)
            for name, s_pt, s_bs in _speeds(g, D):
                pt = front(s_pt, g.a_final.values)
                pr = np.mean([front(s_bs[:, b], D[:, 4, b]) for b in range(D.shape[2])], axis=0)
                fastest = np.mean(s_bs == s_bs.max(axis=0), axis=1)
                rows += [dict(type=typ, scenario=r.scenario, rewiring=r.rewiring, speed=name,
                              with_static=static, on_front=bool(pt[i]), p_front=pr[i],
                              is_fastest=bool(s_pt[i] == s_pt.max()), p_fastest=fastest[i])
                         for i, r in g.iterrows()]
    path = os.path.splitext(out)[0] + '_pareto.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'wrote {path}')


def summary(out):
    """Trade-off and x(opposite)-vs-x(similar) numbers quoted in the D3 notes.

    Trade-off: Spearman between time (t_half or t_zero) and a_final across conditions, positive
    = slower algorithms end higher. Per topology (static included) and pooled; conditions with
    a_final - A0 < MIN_RISE are excluded. Interval = 2.5-97.5 percentiles over draws.
    x(opp) vs x(sim): t_zero speed-up of local/bridge(opposite) over local/bridge(similar),
    1 - T_opp / T_sim, as (i) ratio of means pooled over topologies, (ii) mean of per-topology
    ratios, (iii) per topology."""
    from scipy.stats import spearmanr
    d, z = _load(out)
    d = d[d.a_final - A0 >= MIN_RISE].reset_index(drop=True)
    lines = []

    def ci(v):
        return f'{np.median(v):.2f} [{np.nanpercentile(v, 2.5):.2f}, {np.nanpercentile(v, 97.5):.2f}]'

    for col, k in [('t_half', 2), ('t_zero', 3)]:
        for typ, g in list(d.groupby('type')) + [('pooled', d)]:
            g = g.reset_index(drop=True)
            D = _draws(z, g)
            pt = spearmanr(g[col], g.a_final).correlation
            bs = []
            for b in range(D.shape[2]):
                ok = np.isfinite(D[:, k, b])
                bs.append(spearmanr(D[ok, k, b], D[ok, 4, b]).correlation)
            lines.append(f'spearman({col}, a_final) {typ:8s} n={len(g):2d}  point {pt:.2f}  draws {ci(np.array(bs))}')

    opp = d[d.scenario.isin(['biased', 'bridge']) & (d.rewiring == 'diff')]
    sim = d[d.scenario.isin(['biased', 'bridge']) & (d.rewiring == 'same')]
    Do, Ds = _draws(z, opp)[:, 3], _draws(z, sim)[:, 3]
    lines.append(f'x(opp) vs x(sim) t_zero speed-up, pooled ratio of means: '
                 f'point {1 - opp.t_zero.mean() / sim.t_zero.mean():.3f}  '
                 f'draws {ci(1 - Do.mean(0) / Ds.mean(0))}')
    tops = sorted(opp.type.unique())
    per = np.array([1 - Do[(opp.type == t).values].mean(0) / Ds[(sim.type == t).values].mean(0) for t in tops])
    lines.append(f'x(opp) vs x(sim) t_zero speed-up, mean of per-topology ratios: draws {ci(per.mean(0))}')
    for t, v in zip(tops, per):
        lines.append(f'  {t:8s} {ci(v)}')

    path = os.path.splitext(out)[0] + '_summary.txt'
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'wrote {path}')


def perrun(cache, out):
    """Per-run t_zero (first step with that run's a > 0) for SI Fig. S6.

    Same criterion as the ensemble t_zero, applied to each run. It is not an estimate of the
    ensemble value: the ensemble mean waits for slow runs, so on DPA the per-run median sits
    1-5k steps earlier. Runs that never turn positive are NaN and must be disclosed.
    """
    rows = []
    for cond in sorted(os.listdir(cache)):
        typ, sc, rw = cond.split('__')
        d = os.path.join(cache, cond)
        for f in sorted(f for f in os.listdir(d) if f.endswith('.npy')):
            pos = np.load(os.path.join(d, f)) > 0
            rows.append(dict(type=typ, scenario=sc, rewiring=rw, model_run=int(f[4:7]),
                             T=len(pos), t_zero=float(pos.argmax()) if pos.any() else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    never = df.groupby(['type', 'scenario', 'rewiring']).t_zero.apply(lambda v: v.isna().sum())
    print(f'wrote {out}: {len(df)} runs, {df.t_zero.isna().sum()} never positive')
    print(never[never > 0].to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--extract', metavar='CSV')
    p.add_argument('--bootstrap', action='store_true')
    p.add_argument('--check', action='store_true')
    p.add_argument('--pareto', action='store_true', help='Pareto-front and fastest-algorithm shares from --out and its draws')
    p.add_argument('--summary', action='store_true', help='trade-off correlations and x(opp) vs x(sim) speed-up')
    p.add_argument('--perrun', metavar='CSV', help='write per-run t_zero (SI Fig. S6) from --cache')
    p.add_argument('--cache', default=os.path.join(HERE, '..', '..', 'Output', 'convergence_cache'))
    p.add_argument('--out', default=os.path.join(HERE, '..', '..', 'Output', 'convergence_bootstrap_A1.csv'))
    p.add_argument('--avg', help='avg CSV, to cross-check the full-sample rate against the published path')
    p.add_argument('--chunk', type=int, default=4_000_000)
    p.add_argument('-B', type=int, default=1000)
    p.add_argument('--batch', type=int, default=50)
    p.add_argument('--seed', type=int, default=20261016)
    a = p.parse_args()
    if a.extract:
        extract(a.extract, a.cache, a.chunk)
    if a.check:
        check(a.cache)
    if a.bootstrap:
        bootstrap(a.cache, a.out, a.B, a.seed, a.batch, a.avg)
    if a.pareto:
        pareto(a.out)
    if a.summary:
        summary(a.out)
    if a.perrun:
        perrun(a.cache, a.perrun)
