#!/usr/bin/env python3
"""
run_bf_scan.py
Wrapper that scans over BF values for the mlsp=0 slice and produces
the inputs needed for the ATLAS-style BF vs mchi exclusion plot.

For each BF scan point it:
  1. Calls make_datacards.py  --bf {bf} --xsec 1d --mlsp 0 --era {era}
  2. Calls run_combine.py     --bf {bf} --xsec 1d --mlsp 0 --era {era}

Results land in:
  combine/limits/{era}/limits_{era}_bf{X}_1dxsec.csv   (one per bf)

Then plot with:
  python3 scripts/plot.py --plot bf --era run2run3

Usage:
    python3 scripts/run_bf_scan.py --era run2run3
    python3 scripts/run_bf_scan.py --era run2run3 --n-bf 20 --jobs 4
    python3 scripts/run_bf_scan.py --era run2run3 --bf-min 0.3 --bf-max 1.0
    python3 scripts/run_bf_scan.py --era run2run3 --mchi 300  # single mass test
"""

import os
import sys
import subprocess
import argparse
import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_BF_MIN = 0.05
DEFAULT_BF_MAX = 1.00
DEFAULT_N_BF   = 20


def run_script(script, extra_args, dry_run=False):
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, script)] + extra_args
    print("  $", " ".join(cmd))
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=os.path.dirname(SCRIPTS_DIR))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Scan BF for exclusion plot")
    parser.add_argument("--era",    choices=["run2", "run3", "run2run3"],
                        default="run2run3")
    parser.add_argument("--xsec",   choices=["1d", "2d"], default="1d",
                        help="Use 1d for the BF exclusion plot (default)")
    parser.add_argument("--bf-min", type=float, default=DEFAULT_BF_MIN)
    parser.add_argument("--bf-max", type=float, default=DEFAULT_BF_MAX)
    parser.add_argument("--n-bf",   type=int,   default=DEFAULT_N_BF,
                        help="Number of BF scan points")
    parser.add_argument("--mchi",   type=int,   default=None,
                        help="Restrict to one mchi (for testing)")
    parser.add_argument("--jobs",   type=int,   default=1,
                        help="Parallel jobs passed to run_combine.py")
    parser.add_argument("--force",  action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bf_vals = np.linspace(args.bf_min, args.bf_max, args.n_bf)
    print(f"BF scan: {args.n_bf} points  [{args.bf_min:.2f} → {args.bf_max:.2f}]")
    print(f"era={args.era}  xsec={args.xsec}  mlsp=0\n")

    # For run2run3: make_datacards.py must be called for run2 and run3 separately
    # (each card points to its own era's workspaces). run_combine.py --era run2run3
    # then merges them with combineCards.py before running combine.
    datacard_eras = ["run2", "run3"] if args.era == "run2run3" else [args.era]

    n_ok = n_fail = 0
    for bf in bf_vals:
        bf_str = f"{bf:.4f}"
        print(f"\n── BF = {bf_str} ──────────────────────────────────────")

        common_base = [
            "--bf",   bf_str,
            "--xsec", args.xsec,
            "--mlsp", "0",
        ]
        if args.mchi:
            common_base += ["--mchi", str(args.mchi)]
        if args.force:
            common_base += ["--force"]

        # Step 1: write datacards for each era separately
        cards_ok = True
        for dc_era in datacard_eras:
            ok = run_script("make_datacards.py",
                            ["--era", dc_era] + common_base,
                            dry_run=args.dry_run)
            if not ok:
                print(f"  [FAIL] make_datacards.py --era {dc_era} at bf={bf_str}")
                cards_ok = False
                break
        if not cards_ok:
            n_fail += 1
            continue

        # Step 2: run combine (uses combineCards.py internally for run2run3)
        ok2 = run_script("run_combine.py",
                         ["--era", args.era] + common_base + ["--jobs", str(args.jobs)],
                         dry_run=args.dry_run)
        if not ok2:
            print(f"  [FAIL] run_combine.py at bf={bf_str}")
            n_fail += 1
            continue

        n_ok += 1

    print(f"\n{'='*50}")
    print(f"Done: {n_ok}/{len(bf_vals)} BF points OK  ({n_fail} failed)")
    print(f"\nNext: python3 scripts/plot.py --plot bf --era {args.era} --xsec {args.xsec}")


if __name__ == "__main__":
    main()