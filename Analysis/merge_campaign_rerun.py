#!/usr/bin/env python3
"""Merge a single-condition rerun (from run_phased.py) into the main campaign CSVs.

Replaces the rows for one (scenario, rewiring, topology) condition in an
existing default_run_avg / default_run_individual pair with the rows from a
fresh rerun, e.g. bridge/diff/FB at the corrected 270k horizon (see
claude_stuff/convergence_diagnostic_criteria_2026-07-03.md section 9). Backs
up the originals before overwriting, since Output/ is gitignored.

The merge streams line by line rather than loading the CSVs into a DataFrame:
the individual-run file is ~28 GB (>> available RAM), and a text pass also
preserves the literal string "None" (a real scenario/rewiring value here),
which pandas' default NA handling would otherwise turn into a blank cell.

Usage:
    python merge_campaign_rerun.py \
        --old-avg ../Output/default_run_avg_N_800_n_90_pNf_0_pc_0.05_sweep_20251014_1704_phased_run_gme_2025-10-15.csv \
        --old-individual ../Output/default_run_individual_N_800_n_90_pNf_0_pc_0.05_sweep_20251014_1704_phased_run_gme_2025-10-15.csv \
        --new-avg ../Output/default_run_avg_N_800_n_90_pNf_0_pc_0.05_sweep_<newid>_<date>.csv \
        --new-individual ../Output/default_run_individual_N_800_n_90_pNf_0_pc_0.05_sweep_<newid>_<date>.csv
"""
import argparse
import os
import shutil
from datetime import date


def _key_indices(header_cols, source):
    for name in ("scenario", "rewiring", "type"):
        if name not in header_cols:
            raise ValueError(f"{source} has no '{name}' column (header: {header_cols})")
    return header_cols.index("scenario"), header_cols.index("rewiring"), header_cols.index("type")


def merge_condition(old_path, new_path, scenario, rewiring, topology):
    """Rewrite old_path with the target condition's rows replaced by new_path's.

    Streams both files line by line (constant memory), so the multi-GB
    individual-run CSV never has to fit in RAM, and the literal "None" values
    are copied through verbatim instead of being coerced to blanks.
    """
    target = (scenario, rewiring, topology)

    with open(old_path, "r") as f:
        old_header = f.readline()
    old_cols = old_header.rstrip("\r\n").split(",")
    o_s, o_r, o_t = _key_indices(old_cols, old_path)

    tmp_path = old_path + ".merge_tmp"
    dropped = added = new_total = 0
    try:
        with open(old_path, "r") as fin, open(tmp_path, "w") as fout:
            fout.write(old_header)
            next(fin)  # skip the header we already wrote
            for line in fin:
                parts = line.rstrip("\r\n").split(",")
                if (parts[o_s], parts[o_r], parts[o_t]) == target:
                    dropped += 1
                else:
                    fout.write(line)

            with open(new_path, "r") as fnew:
                new_cols = fnew.readline().rstrip("\r\n").split(",")
                n_s, n_r, n_t = _key_indices(new_cols, new_path)
                reorder = new_cols != old_cols
                if reorder:
                    if sorted(new_cols) != sorted(old_cols):
                        raise ValueError(
                            f"Column mismatch: {new_path} {new_cols} vs {old_path} {old_cols}")
                    perm = [new_cols.index(c) for c in old_cols]
                for line in fnew:
                    parts = line.rstrip("\r\n").split(",")
                    new_total += 1
                    if (parts[n_s], parts[n_r], parts[n_t]) == target:
                        added += 1
                        if reorder:
                            fout.write(",".join(parts[k] for k in perm) + "\n")
                        else:
                            fout.write(line if line.endswith("\n") else line + "\n")
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if added == 0:
        os.remove(tmp_path)
        raise ValueError(f"No {scenario}/{rewiring}/{topology} rows found in {new_path}")
    if new_total != added:
        print(f"  note: ignoring {new_total - added} row(s) in {new_path} outside the target condition")

    backup_path = old_path.replace(".csv", f"_prererun_backup_{date.today()}.csv")
    shutil.copy2(old_path, backup_path)
    os.replace(tmp_path, old_path)
    print(f"{old_path}: dropped {dropped} old row(s), added {added} new row(s) "
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
