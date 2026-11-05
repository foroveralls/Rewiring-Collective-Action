#!/usr/bin/env python3
"""
SI default-parameter table: steady-state mean opinion a* and polarization P* per
algorithm x topology, median [IQR] across the 90 runs of the main campaign.

Input:  Output/per_run_summary.csv (Analysis/Stats/summarize_individual_csv.py)
Output: Output/SI_default_params_table.csv and Output/SI_default_params_table.tex
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "Output")
PER_RUN = os.path.join(OUT, "per_run_summary.csv")

NAME = {'none_none': 'Static', 'random_none': 'Random', 'biased_same': 'Local (similar)',
        'biased_diff': 'Local (opposite)', 'bridge_same': 'Bridge (similar)',
        'bridge_diff': 'Bridge (opposite)', 'wtf_none': 'WTF', 'node2vec_none': 'N2V'}
TOPO = {'DPAH': 'DPA', 'cl': 'CSF', 'Twitter': 'Twitter', 'FB': 'FB'}


def ci(x):
    x = np.asarray(x, float)
    # Minus signs only: P* is non-negative by construction, and an explicit "+"
    # on a level (as opposed to a difference) is not the convention in this field.
    return f"{np.median(x):.2f} [{np.percentile(x, 25):.2f}, {np.percentile(x, 75):.2f}]"


def main():
    pr = pd.read_csv(PER_RUN)
    pr['rewiring'] = pr['rewiring'].fillna('none')
    pr['scenario'] = pr['scenario'].fillna('none')
    pr['grouped'] = pr['scenario'] + '_' + pr['rewiring']
    rows = []
    for g, algo in NAME.items():
        for t, topo in TOPO.items():
            s = pr[(pr['grouped'] == g) & (pr['type'] == t)]
            if s.empty:
                continue
            rows.append({'Algorithm': algo, 'Topology': topo, '$n$': len(s),
                         r'$\langle a^* \rangle$': ci(s.cooperativity),
                         r'$\langle P^* \rangle$': ci(s.final_polarization)})
    si = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(si.to_string(index=False))

    stem = os.path.join(OUT, "SI_default_params_table")
    si.to_csv(stem + ".csv", index=False)
    with open(stem + ".tex", "w") as f:
        f.write(si.to_latex(index=False, escape=False, column_format='llccc'))
    print(f"\nsaved {stem}.csv\nsaved {stem}.tex")


if __name__ == "__main__":
    main()
