#!/usr/bin/env python3
"""Splice a heatmap re-run patch into the master pnf x stubbornness heatmap CSV.

Replaces whole (topology, mode, rewiring, stubbornness) blocks in the master with
the corresponding rows from a patch produced by rerun_subset_sweep.py. The patch
covers every pnf value (and all 30 sims) for each block it touches, so block-level
replacement keeps the grid complete and the total row count unchanged.

The master heatmap is small (~8 MB, 90k rows) so this loads it into pandas - unlike
the 28 GB individual-run campaign file that merge_campaign_rerun.py has to stream.
Both CSVs are read with keep_default_na=False so the literal string "None" (a real
value in the `mode` and `rewiring` columns here) is not coerced to NaN, which would
break the join keys.

Usage:
    python splice_heatmap_rerun.py \
        --master ../Output/heatmap_sweep_phased_sweep_20251014_1511_stubbornness_polarisingNode_f_pxc.csv \
        --patch  ../Output/heatmap_sweep_phased_RERUN_<id>.csv
    # overwrites the master in place after backing it up; pass --out to redirect.
"""
import argparse
import shutil
from datetime import date

import pandas as pd

# stubbornness is rounded into a join key so float repr differences can't cause a
# block to silently miss. Both files come from np.linspace(0,1,10) so they already
# match to full precision; the round is belt-and-braces.
KEY = ["topology", "mode", "rewiring", "stub_key"]
NUMERIC = ["state", "state_std", "stubbornness", "polarisingNode_f"]


def load(path):
    df = pd.read_csv(path, keep_default_na=False, na_values=[])
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col])
    df["stub_key"] = df["stubbornness"].round(6)
    return df


def splice(master_path, patch_path, out_path=None):
    master = load(master_path)
    patch = load(patch_path)

    cols = [c for c in master.columns if c != "stub_key"]
    if cols != [c for c in patch.columns if c != "stub_key"]:
        raise ValueError(
            f"column mismatch:\n  master {list(master.columns)}\n  patch  {list(patch.columns)}")

    blocks = patch[KEY].drop_duplicates()
    merged = master.merge(blocks, on=KEY, how="left", indicator=True)
    kept = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    dropped = len(master) - len(kept)

    result = pd.concat([kept, patch], ignore_index=True)[cols]

    if len(result) != len(master):
        print(f"WARNING: row count changed {len(master)} -> {len(result)}. A patched "
              f"block has a different pnf/sim count than the master, or the patch "
              f"introduced blocks absent from the master.")

    if out_path is None:
        backup = master_path.replace(".csv", f"_prererun_backup_{date.today()}.csv")
        shutil.copy2(master_path, backup)
        out_path = master_path
        print(f"backup written: {backup}")

    result.to_csv(out_path, index=False)
    print(f"blocks replaced: {len(blocks)} | master rows dropped: {dropped} | "
          f"patch rows added: {len(patch)} | final rows: {len(result)}")
    print(f"written: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", required=True, help="master heatmap CSV to update")
    p.add_argument("--patch", required=True,
                   help="RERUN patch CSV from rerun_subset_sweep.py")
    p.add_argument("--out", default=None,
                   help="output path (default: overwrite master after backup)")
    args = p.parse_args()
    splice(args.master, args.patch, args.out)


if __name__ == "__main__":
    main()
