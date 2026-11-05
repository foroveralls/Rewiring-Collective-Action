#!/bin/bash
# Runs added in the minor revision (Oct 2026): WTF at its long horizon, and the rewiring-rate (p_join)
# sweep. WTF runs switch between opposed and aligned states, so WTF runs 500k steps (run_phased.py
# TIMESTEPS_OVERRIDE) and its sweep cells come from wtf_long_sweep.py.
# NETWORK_CACHE=0 WEIGHT_BATCH=1 is the main-campaign setting (see models_checks.py).
# About 20 h with 54 worker processes, most of it the p_join sweep (step 3) and the (rho, w) cells (step 4).
set -euo pipefail
cd "$(dirname "$0")"
export NETWORK_CACHE=0 WEIGHT_BATCH=1

# 1. WTF in the main campaign: 90 runs per network, 500k steps
python run_phased.py --conditions wtf:None:DPAH,wtf:None:Twitter

# 2. WTF horizon check (SI, "WTF horizon"): 1M steps; compare each run's [450k, 500k) and [900k, 1M) windows
python run_phased.py --conditions wtf:None:DPAH,wtf:None:Twitter --timesteps 1000000

# 3. p_join sweep on DPA and CSF, 90 runs per condition. p_join 0.25 runs twice the horizon (same expected
#    number of new links); WTF (directed, so DPA only) at p_join 0.5 is step 1.
C=""
for t in DPAH cl; do
    for s in random:None biased:same biased:diff bridge:same bridge:diff node2vec:None; do C="$C,$s:$t"; done
done
C=${C#,}
for p in 0.5 0.75 1.0; do python run_phased.py --conditions "$C" --pjoin $p; done
python run_phased.py --conditions "$C" --pjoin 0.25 --timesteps 320000
for p in 0.75 1.0; do python run_phased.py --conditions wtf:None:DPAH --pjoin $p; done
python run_phased.py --conditions wtf:None:DPAH --pjoin 0.25 --timesteps 1000000

# 4. WTF cells of the phi and (rho, w) sweeps; the script sets NETWORK_CACHE and WEIGHT_BATCH per cell
unset NETWORK_CACHE WEIGHT_BATCH
python wtf_long_sweep.py phi
python wtf_long_sweep.py rhow
