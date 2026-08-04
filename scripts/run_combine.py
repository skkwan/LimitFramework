#!/usr/bin/env python3
"""
run_combine.py
Runs Combine AsymptoticLimits on all datacards and collects results into a CSV.

Single era (--era run2 or --era run3):
  Runs combine directly on the per-era datacards.

Combined (--era run2run3):
  For each mass point, calls combineCards.py to merge the run2 and run3
  datacards into a single combined datacard:
    run2_SRHH, run2_SRZH, run3_SRHH, run3_SRZH  (imax=4)
  Then runs combine on the merged card.
  Requires run2 and run3 datacards to exist — run make_datacards.py for
  both eras first.

Usage:
    python3 scripts/run_combine.py --era run2    --bf 1.0 --xsec 2d
    python3 scripts/run_combine.py --era run3    --bf 1.0 --xsec 2d
    python3 scripts/run_combine.py --era run2run3 --bf 1.0 --xsec 2d
    python3 scripts/run_combine.py --era run2run3 --jobs 4
    python3 scripts/run_combine.py --mchi 300 --mlsp 0 --era run2run3 --force
"""

import os
import sys
import re
import csv
import json
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
import config


# ── Combined card builder ─────────────────────────────────────────────────────

def make_combined_card(mchi, mlsp, bf, xsec_mode):
    """
    Call combineCards.py to merge run2 and run3 datacards for one mass point.
    Saves combined card to combine/datacards/run2run3/.
    Returns (card_path, None) on success or (None, error_str) on failure.

    combineCards.py renames channels as run2_SRHH, run2_SRZH, run3_SRHH, run3_SRZH
    and keeps absolute workspace paths intact so no workspace changes are needed.
    """
    card_run2 = config.datacard_path(mchi, mlsp, 'run2', bf, xsec_mode)
    card_run3 = config.datacard_path(mchi, mlsp, 'run3', bf, xsec_mode)

    missing = []
    if not os.path.exists(card_run2): missing.append(f"run2: {card_run2}")
    if not os.path.exists(card_run3): missing.append(f"run3: {card_run3}")
    if missing:
        return None, "missing cards: " + "; ".join(missing)

    combined = config.datacard_path(mchi, mlsp, 'run2run3', bf, xsec_mode)
    os.makedirs(os.path.dirname(combined), exist_ok=True)

    cmd = ["combineCards.py", f"run2={card_run2}", f"run3={card_run3}"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return None, "combineCards.py not found — source Combine environment"
    except subprocess.TimeoutExpired:
        return None, "combineCards.py timed out"

    if result.returncode != 0:
        return None, f"combineCards.py failed: {result.stderr[:300]}"

    with open(combined, 'w') as f:
        f.write(result.stdout)
    return combined, None


# ── Combine runner ────────────────────────────────────────────────────────────

def run_one(args):
    """Run Combine on one mass point. Returns result dict."""
    mchi, mlsp, era_tag, bf, xsec_mode, out_dir, force = args

    bf_str = f"bf{bf:.2f}".replace('.', 'p')
    result_json = os.path.join(out_dir,
                               f"result_mchi{mchi}_mlsp{mlsp}_{bf_str}_{xsec_mode}.json")
    if not force and os.path.exists(result_json):
        with open(result_json) as f:
            return json.load(f)

    # Build or locate the datacard
    if era_tag == 'run2run3':
        card_path, err = make_combined_card(mchi, mlsp, bf, xsec_mode)
        if card_path is None:
            return {'mchi': mchi, 'mlsp': mlsp, 'status': 'no_card', 'detail': err}
    else:
        card_path = config.datacard_path(mchi, mlsp, era_tag, bf, xsec_mode)
        if not os.path.exists(card_path):
            return {'mchi': mchi, 'mlsp': mlsp,
                    'status': 'no_card', 'detail': str(card_path)}

    tag = f"_mchi{mchi}_mlsp{mlsp}_{era_tag}"
    cmd = [
        "combine", "-M", "AsymptoticLimits",
        card_path,
        "--run", "expected",
        "-t", "-1",
        "--expectSignal", "0",
        "-n", tag,
        "--noFitAsimov",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=out_dir, timeout=600)
    except subprocess.TimeoutExpired:
        return {'mchi': mchi, 'mlsp': mlsp, 'status': 'timeout'}
    except FileNotFoundError:
        return {'mchi': mchi, 'mlsp': mlsp, 'status': 'combine_not_found'}

    if proc.returncode != 0:
        return {'mchi': mchi, 'mlsp': mlsp, 'status': 'failed',
                'stderr': proc.stderr[-600:]}

    limits = parse_limits(proc.stdout)
    if not limits:
        return {'mchi': mchi, 'mlsp': mlsp, 'status': 'parse_failed',
                'stdout': proc.stdout[-300:]}

    result = {
        'mchi': mchi, 'mlsp': mlsp,
        'era': era_tag, 'bf': bf, 'xsec': xsec_mode,
        'status': 'ok',
        **limits,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(result_json, 'w') as f:
        json.dump(result, f, indent=2)
    return result


def parse_limits(stdout):
    pat = re.compile(r'Expected\s+([\d.]+)%:\s+r\s+<\s+([\d.eE+\-]+)')
    key_map = {'2.5': 'exp_m2', '16.0': 'exp_m1', '50.0': 'exp',
               '84.0': 'exp_p1', '97.5': 'exp_p2'}
    limits = {}
    for m in pat.finditer(stdout):
        key = key_map.get(m.group(1))
        if key:
            limits[key] = float(m.group(2))
    return limits if len(limits) == 5 else {}


# ── CSV writer ────────────────────────────────────────────────────────────────

def save_csv(results, csv_path):
    cols = ['mchi', 'mlsp', 'era', 'bf', 'xsec', 'status',
            'exp_m2', 'exp_m1', 'exp', 'exp_p1', 'exp_p2']
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in sorted(results, key=lambda x: (x.get('mchi', 0), x.get('mlsp', 0))):
            w.writerow(r)
    print(f"  Saved: {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Combine limits")
    parser.add_argument("--era",   choices=["run2", "run3", "run2run3", "both"],
                        default="both",
                        help="'run2run3' merges cards with combineCards.py first")
    parser.add_argument("--bf",    type=float, default=1.0)
    parser.add_argument("--xsec",  choices=["2d", "1d"], default="2d")
    parser.add_argument("--mchi",  type=int, default=None)
    parser.add_argument("--mlsp",  type=int, default=None)
    parser.add_argument("--jobs",  type=int, default=1,
                        help="Parallel jobs (default: 1)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    eras = ["run2", "run3", "run2run3"] if args.era == "both" else [args.era]

    if args.bf == 1.0:
        mass_points = sorted(config.SIGNAL_GRID['HH'])
    else:
        mass_points = sorted(set(config.SIGNAL_GRID['HH']) & set(config.SIGNAL_GRID['ZH']))
    
    if args.mchi is not None:
        mass_points = [(mc, ml) for mc, ml in mass_points if mc == args.mchi]
    if args.mlsp is not None:
        mass_points = [(mc, ml) for mc, ml in mass_points if ml == args.mlsp]

    for era in eras:
        out_dir = os.path.join(config.LIMITS_DIR, era)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== era={era}  bf={args.bf}  xsec={args.xsec} ===")
        if era == 'run2run3':
            print("  Will merge per-era cards via combineCards.py for each mass point")

        job_args = [
            (mchi, mlsp, era, args.bf, args.xsec, out_dir, args.force)
            for mchi, mlsp in mass_points
        ]
        print(f"Running {len(job_args)} jobs (parallel={args.jobs})", flush=True)

        results = []
        n_ok = n_skip = n_fail = 0

        def handle(r):
            nonlocal n_ok, n_skip, n_fail
            results.append(r)
            status = r.get('status', '?')
            mc, ml = r.get('mchi', '?'), r.get('mlsp', '?')
            if status == 'ok':
                n_ok += 1
                print(f"  OK  mchi={mc:4d} mlsp={ml:4d}  "
                      f"exp={r['exp']:.4f}  "
                      f"[-2s={r['exp_m2']:.4f}  +2s={r['exp_p2']:.4f}]",
                      flush=True)
            elif status in ('no_card', 'skip'):
                n_skip += 1
                detail = r.get('detail', '')
                print(f"  --  mchi={mc} mlsp={ml}  "
                      f"({status}{': '+detail[:80] if detail else ''})", flush=True)
            else:
                n_fail += 1
                print(f"  !!  mchi={mc} mlsp={ml}  FAILED: {status}", flush=True)
                if 'stderr' in r:
                    print(f"      {r['stderr'][-300:]}")

        if args.jobs == 1:
            for ja in job_args:
                handle(run_one(ja))
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futures = {ex.submit(run_one, ja): ja for ja in job_args}
                for fut in as_completed(futures):
                    handle(fut.result())

        print(f"\n  Done: {n_ok} OK  {n_skip} skipped  {n_fail} failed")

        ok_results = [r for r in results if r.get('status') == 'ok']
        if ok_results:
            bf_str = f"bf{args.bf:.2f}".replace('.', 'p')
            csv_path = os.path.join(
                config.LIMITS_DIR,
                f"limits_{era}_{bf_str}_{args.xsec}xsec.csv")
            save_csv(ok_results, csv_path)
            print(f"\n  Plot: python3 scripts/plot.py --csv {csv_path} --era {era}")


if __name__ == "__main__":
    main()