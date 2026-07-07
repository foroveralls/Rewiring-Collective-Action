#!/usr/bin/env python3
"""Merge a single-condition rerun (from run_phased.py) into the main campaign CSVs.

Replaces the rows for one (scenario, rewiring, topology) condition in an
existing default_run_avg / default_run_individual pair with the rows from a
fresh rerun, e.g. bridge/diff/FB at the corrected 270k horizon (see
claude_stuff/convergence_diagnostic_criteria_2026-07-03.md section 9). Backs
up the originals before overwriting, since Output/ is gitignored.

Usage:
    python merge_campaign_rerun.py \
        --old-avg ../Output/default_run_avg_N_800_n_90_pNf_0_pc_0.05_sweep_20251014_1704_phased_run_gme_2025-10-15.csv \
        --old-individual ../Output/default_run_individual_N_800_n_90_pNf_0_pc_0.05_sweep_20251014_1704_phased_run_gme_2025-10-15.csv \
        --new-avg ../Output/default_run_avg_N_800_n_90_pNf_0_pc_0.05_sweep_<newid>_<date>.csv \
        --new-individual ../Output/default_run_individual_N_800_n_90_pNf_0_pc_0.05_sweep_<newid>_<date>.csv
"""
import argparse
import shutil
from datetime import date

import pandas as pd


def merge_condition(old_path, new_path, scenario, rewiring, topology):
    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)

    new_mask = (new_df["scenario"] == scenario) & (new_df["rewiring"] == rewiring) & (new_df["type"] == topology)
    new_rows = new_df[new_mask]
    if new_rows.empty:
        raise ValueError(f"No {scenario}/{rewiring}/{topology} rows found in {new_path}")
    if len(new_rows) != len(new_df):
        print(f"  note: ignoring {len(new_df) - len(new_rows)} row(s) in {new_path} outside the target condition")

    old_mask = (old_df["scenario"] == scenario) & (old_df["rewiring"] == rewiring) & (old_df["type"] == topology)
    dropped = int(old_mask.sum())
    merged = pd.concat([old_df[~old_mask], new_rows], ignore_index=True)

    backup_path = old_path.replace(".csv", f"_prererun_backup_{date.today()}.csv")
    shutil.copy2(old_path, backup_path)
    merged.to_csv(old_path, index=False)
    print(f"{old_path}: dropped {dropped} old row(s), added {len(new_rows)} new row(s) "
          f"(backup: {backup_path})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old-avg", required=True, help="existing default_run_avg_*.csv to update in place")
    p.add_argument("--old-individual", required=True, help="existing default_run_individual_*.csv to update in place")
    p.add_argument("--new-avg", required=True, help="default_run_avg_*.csv from the single-condition rerun")
    p.add_argument("--new-individual", required=True, help="default_run_individual_*.csv from the single-condition rerun")
    p.add_argument("--scenario", default="bridge")
    p.add_argument("--rewiring", default="diff")
    p.add_argument("--topology", default="FB")
    args = p.parse_args()

    merge_condition(args.old_avg, args.new_avg, args.scenario, args.rewiring, args.topology)
    merge_condition(args.old_individual, args.new_individual, args.scenario, args.rewiring, args.topology)


if __name__ == "__main__":
    main()
