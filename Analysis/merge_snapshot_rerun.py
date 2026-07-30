#!/usr/bin/env python3
"""Merge a single-condition rerun's snapshots into the master snapshot pickle.

The pickle counterpart of merge_campaign_rerun.py: that script fixes the
campaign CSVs, this one fixes ../Output/all_snapshots_<sweep_id>_<date>.pkl.gz,
the separate artifact used for per-agent extraction (extract_states_lowmem.py).
It replaces one "<scenario>_<rewiring>_<topology>" entry of the master dict with
the corresponding entry from a fresh rerun, e.g. bridge/diff/FB at the corrected
270k horizon (see claude_stuff/Infrastructure/convergence_diagnostic_criteria_2026-07-03.md
section 9). The original is backed up first, since Output/ is gitignored.

The master dict is
    {"<algo>_<mode>_<topology>": {"metadata": {...}, "snapshots": {run: {t: graph}}}}
(built in run_phased.py's snapshot consolidation step), so the merge is a
single top-level key swap. Graph objects are never introspected, and are
unpickled normally rather than through extract_states_lowmem.py's state-only
stub: the point of this script is to produce a corrected *full fidelity* master
pickle, so the netin/networkx graphs must survive the round trip intact.

CLUSTER-ONLY / BIG-RAM. Keeping the graphs means the old master has to be
loaded whole: pickle rebuilds the entire object graph in the unpickler's memo
before load() returns, so there is no streaming or partial-read escape (the
same OOM that motivated extract_states_lowmem.py's stub - a plain load of the
~1.2 GB gzipped master kills a ~15 GB box). Run this where the RAM is, right
before the extraction step. The rerun pickle is a single condition and is cheap
to load anywhere.

Usage:
    python merge_snapshot_rerun.py \
        --old-pkl ../Output/all_snapshots_sweep_20250905_1501_phased_run_afk_2025-09-05.pkl.gz \
        --new-pkl ../Output/all_snapshots_sweep_20260724_1637_phased_run_pte_2026-07-24.pkl.gz
"""
import argparse
import gzip
import os
import pickle
import shutil
import sys
from datetime import date

# The pickled graphs reference models_checks.Agent, so Analysis/ must be
# importable no matter where this is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_master(path, label):
    print(f"loading {label} {path} ...", flush=True)
    with gzip.open(path, "rb") as f:
        master = pickle.load(f)
    if not isinstance(master, dict):
        raise ValueError(f"{path} does not hold a master snapshot dict (got {type(master)})")
    print(f"  {len(master)} condition(s): {sorted(master)}", flush=True)
    return master


def n_runs(entry):
    """Number of simulation runs stored under a master-dict entry."""
    return len(entry.get("snapshots", {})) if isinstance(entry, dict) else 0


def backup_path_for(old_path):
    if old_path.endswith(".pkl.gz"):
        stem = old_path[: -len(".pkl.gz")]
    else:
        stem = os.path.splitext(old_path)[0]
    return f"{stem}_prererun_backup_{date.today()}.pkl.gz"


def merge_snapshots(old_path, new_path, scenario, rewiring, topology):
    """Rewrite old_path with the target condition's entry taken from new_path."""
    target = f"{scenario}_{rewiring}_{topology}"

    new_master = load_master(new_path, "rerun pickle")
    if target not in new_master:
        raise ValueError(
            f"No '{target}' entry in {new_path}; it has: {sorted(new_master)}")
    new_entry = new_master[target]
    added = n_runs(new_entry)
    if len(new_master) > 1:
        print(f"  note: ignoring {len(new_master) - 1} condition(s) in {new_path} "
              f"outside the target")

    old_master = load_master(old_path, "master pickle")
    if target in old_master:
        dropped = n_runs(old_master[target])
    else:
        dropped = 0
        print(f"  WARNING: '{target}' is not in {old_path}; adding it as a new "
              f"condition rather than replacing one")

    old_master[target] = new_entry

    tmp_path = old_path + ".merge_tmp"
    try:
        with gzip.open(tmp_path, "wb") as f:
            pickle.dump(old_master, f)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    backup_path = backup_path_for(old_path)
    shutil.copy2(old_path, backup_path)
    os.replace(tmp_path, old_path)
    print(f"{old_path}: '{target}' dropped {dropped} old run(s), added {added} "
          f"new run(s); now {len(old_master)} condition(s) (backup: {backup_path})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old-pkl", required=True, help="existing all_snapshots_*.pkl.gz to update in place")
    p.add_argument("--new-pkl", required=True, help="all_snapshots_*.pkl.gz from the single-condition rerun")
    p.add_argument("--scenario", default="bridge")
    p.add_argument("--rewiring", default="diff")
    p.add_argument("--topology", default="FB")
    args = p.parse_args()

    merge_snapshots(args.old_pkl, args.new_pkl, args.scenario, args.rewiring, args.topology)


if __name__ == "__main__":
    main()
