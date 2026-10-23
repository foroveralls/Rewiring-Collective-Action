#!/usr/bin/env python3
"""
Bootstrap intervals for the Fig. 4 field thresholds (round-2 R1b.6, analysis A2).

The threshold phi* is the published one, `field_strip.crossing_phi`: per algorithm, the median
a* across runs at each of the 12 swept field values, then the first upward zero crossing of that
median (y[i] < 0 <= y[i+1]), linearly interpolated between the two grid values. It is undefined
(NaN) if the median never crosses; such resamples are counted, not dropped silently.

Resampling: runs are drawn with replacement within each (algorithm, topology, phi) cell (30 runs),
B times with a fixed seed. One set of draws per cell is shared by every estimand, so pooled,
per-topology and directed/undirected values and all differences are computed on the same draw
(differences are paired by draw; the runs themselves are independent across algorithms).
Pooling is the published one: concatenate the topology cells (120 runs per cell, 60 for WTF,
which exists only on DPAH and Twitter) and take the median.

Reported, each as full-sample point + 2.5-97.5 percentile interval over defined draws +
share of undefined draws:
  (a) phi* per algorithm, pooled over topologies
  (b) phi* per algorithm per topology
  (c) phi* on directed (DPAH + Twitter) and undirected (cl + FB) pools, and the shift
      directed - undirected
  (d) differences phi*(alg) - phi*(static), pooled and per topology, plus a few recommender
      contrasts; "separated" = the difference interval excludes 0.
x-opp / x-sim pool the local and bridge variants into one ensemble (240 runs per cell pooled);
they are a sensitivity check, the figure draws L-opp and B-opp as separate columns.

DATA: the corrected political-climate sweep (`tay`, rho = 0.10, w = 0.6). PHI_SWEEP_CSV overrides.

Usage:
    python phi_threshold_bootstrap_A2.py              # B = 1000, writes the summary txt
    python phi_threshold_bootstrap_A2.py -B 200 --out /tmp/x.txt
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Plotting'))
import field_strip as fs  # noqa: E402

TOPS = ['DPAH', 'Twitter', 'cl', 'FB']
POOLS = {'pooled': TOPS, 'directed': ['DPAH', 'Twitter'], 'undirected': ['cl', 'FB'],
         'DPAH': ['DPAH'], 'Twitter': ['Twitter'], 'cl': ['cl'], 'FB': ['FB']}
COMPOSITE = {'x-opp': ['L-opp', 'B-opp'], 'x-sim': ['L-sim', 'B-sim']}
OUT = os.path.join(HERE, '..', '..', 'Output', 'phi_threshold_bootstrap_A2_summary.txt')


def crossing(Y, phis):
    """Vectorised `field_strip.crossing_phi` over rows of Y (draws x phi)."""
    up = (Y[:, :-1] < 0) & (Y[:, 1:] >= 0)
    ok = up.any(1)
    i = up.argmax(1)
    r = np.arange(len(Y))
    y0, y1 = Y[r, i], Y[r, i + 1]
    out = phis[i] + (0 - y0) * (phis[i + 1] - phis[i]) / (y1 - y0)
    out[~ok] = np.nan
    return out, up.sum(1)


def build(df):
    """cells[(alg, top)] = array (n_phi, n_runs) of a*, phi ascending."""
    phis = np.sort(df['politicalClimate'].unique())
    cells = {}
    for (alg, top), g in df.groupby(['alg', 'topology']):
        cells[(alg, top)] = np.vstack([g.loc[g['politicalClimate'] == p, 'state'].to_numpy()
                                       for p in phis])
    return phis, cells


def run(df, B, seed):
    phis, cells = build(df)
    rng = np.random.default_rng(seed)
    # one index draw per (alg, top, phi) cell, shared by all estimands
    idx = {k: rng.integers(0, v.shape[1], size=(B, v.shape[0], v.shape[1]))
           for k, v in sorted(cells.items())}
    algs = [a for a in fs.ALGO_ORDER if any(k[0] == a for k in cells)] + list(COMPOSITE)

    est, multi = {}, {}
    for alg in algs:
        base = COMPOSITE.get(alg, [alg])
        for pool, tops in POOLS.items():
            keys = [(a, t) for a in base for t in tops if (a, t) in cells]
            if not keys:
                continue
            full = np.concatenate([cells[k] for k in keys], axis=1)          # phi x n
            pt, _ = crossing(np.median(full, 1)[None, :], phis)
            # resampled: (B, phi, n) per cell, concatenated along runs
            R = np.concatenate([np.take_along_axis(np.broadcast_to(cells[k], (B,) + cells[k].shape),
                                                   idx[k], axis=2) for k in keys], axis=2)
            d, n_up = crossing(np.median(R, 2), phis)
            est[(alg, pool)] = (pt[0], d)
            multi[(alg, pool)] = np.mean(n_up > 1)
    return phis, est, multi


def fmt(pt, d, nd=4):
    ok = np.isfinite(d)
    if not ok.any():
        return f'{pt:.{nd}f} [undefined in all draws]'
    lo, hi = np.percentile(d[ok], [2.5, 97.5])
    s = f'{pt:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]'
    if (~ok).any():
        s += f'  undefined {np.mean(~ok):.1%}'
    return s


def diff_line(est, a, b, pool):
    if (a, pool) not in est or (b, pool) not in est:
        return None
    pa, da = est[(a, pool)]
    pb, db = est[(b, pool)]
    d = da - db
    ok = np.isfinite(d)
    lo, hi = np.percentile(d[ok], [2.5, 97.5])
    sep = 'separated' if (hi < 0 or lo > 0) else 'not separated'
    return (f'  {a:8s} - {b:8s} {pool:10s} {pa - pb:+.4f} [{lo:+.4f}, {hi:+.4f}]  '
            f'P(diff<0)={np.mean(d[ok] < 0):.3f}  undefined {np.mean(~ok):.1%}  {sep}')


def check_point(df, est):
    """Full-sample estimates must equal field_strip's published crossings."""
    for pool, key in [('pooled', 'all'), ('directed', 'directed'), ('undirected', 'undirected')]:
        ref = fs.crossing_phi(fs.summarize(df[df['topology'].isin(POOLS[pool])]))
        for alg, v in ref.items():
            assert np.isclose(est[(alg, pool)][0], v, equal_nan=True), (alg, pool, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default=os.environ.get('PHI_SWEEP_CSV', fs.DEFAULT_CSV))
    ap.add_argument('-B', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    df = fs.load_data(a.csv)
    phis, est, multi = run(df, a.B, a.seed)
    check_point(df, est)
    algs = list(dict.fromkeys(k[0] for k in est))

    L = [f'phi* bootstrap (R1b.6 / A2). csv={os.path.basename(a.csv)}  B={a.B}  seed={a.seed}',
         f'{len(df)} runs; grid step {np.diff(phis).mean():.5f}; phi* = first upward zero crossing '
         f'of the median a*, linearly interpolated (field_strip.crossing_phi).',
         'Point = full sample; [2.5, 97.5] percentiles over defined draws; "undefined" = share of '
         'draws whose median never crosses.', '']
    L.append('(a) pooled over topologies (120 runs per cell; WTF 60; x-opp/x-sim 240)')
    for alg in algs:
        L.append(f'  {alg:8s} {fmt(*est[(alg, "pooled")])}')
    L += ['', '(b) per topology (30 runs per cell)']
    for alg in algs:
        L.append(f'  {alg:8s} ' + '  '.join(f'{t}: {fmt(*est[(alg, t)])}'
                                           for t in TOPS if (alg, t) in est))
    L += ['', '(c) directed (DPAH+Twitter) vs undirected (cl+FB), 60 runs per cell; shift = dir - undir']
    for alg in algs:
        if (alg, 'undirected') not in est:
            L.append(f'  {alg:8s} directed {fmt(*est[(alg, "directed")])}  (no undirected runs)')
            continue
        (pd_, dd), (pu, du) = est[(alg, 'directed')], est[(alg, 'undirected')]
        L.append(f'  {alg:8s} directed {fmt(pd_, dd)}  undirected {fmt(pu, du)}  '
                 f'shift {fmt(pd_ - pu, dd - du)}')
    shifts = {alg: est[(alg, 'directed')][0] - est[(alg, 'undirected')][0]
              for alg in algs if (alg, 'undirected') in est}
    big = max(shifts, key=lambda k: abs(shifts[k]))
    L.append(f'  largest |shift|: {big} {shifts[big]:+.4f}')

    L += ['', '(d) differences in phi*; "separated" = 95% interval of the difference excludes 0']
    for alg in ['L-opp', 'B-opp', 'x-opp', 'random', 'L-sim', 'B-sim', 'x-sim', 'wtf', 'node2vec']:
        for pool in ['pooled'] + TOPS + ['directed']:
            line = diff_line(est, alg, 'static', pool)
            if line:
                L.append(line)
        L.append('')
    L.append('  recommender contrasts')
    for x, y, pool in [('node2vec', 'B-sim', 'pooled'), ('node2vec', 'L-sim', 'pooled'),
                       ('wtf', 'node2vec', 'pooled'), ('wtf', 'node2vec', 'directed'),
                       ('wtf', 'B-sim', 'pooled'), ('wtf', 'static', 'directed'),
                       ('L-opp', 'random', 'pooled'), ('B-opp', 'random', 'pooled')]:
        L.append(diff_line(est, x, y, pool))

    L += ['', 'draws with more than one upward crossing of the median (non-monotone ladder), '
          'nonzero only:']
    L += [f'  {k[0]:8s} {k[1]:10s} {v:.1%}' for k, v in multi.items() if v > 0] or ['  none']

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
