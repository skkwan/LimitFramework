#!/usr/bin/env python3
"""
run_combine.py
Runs Combine AsymptoticLimits on every datacard in a directory and collects
the expected limits into a CSV.

Datacards must follow the make_datacards.py naming:
    datacard_mchi{mchi}_mlsp{mlsp}_{era}_{model}.txt

--mchi / --mlsp restrict the run to datacards with that mass (either or
both). A restricted run updates only its own mass points in the CSV and
keeps the other rows; an unrestricted run rewrites the CSV.

--m bino --mlsp all runs every mlsp in card_dir (the full 2D plane) and
writes one CSV, limits_{era}_bino_mlspall.csv, for plot.py --plot 2d.
--m gmsb --mlsp all is the same as --m gmsb (mlsp = 1 GeV only).

Usage:
    python3 run_combine.py ../combine/datacards/run2/gmsb --m gmsb --jobs 4
    python3 run_combine.py ../combine/datacards/run2/bino --m bino --mlsp 100 --jobs 4
    python3 run_combine.py ../combine/datacards/run2/bino --m bino --mlsp all --jobs 8
    python3 run_combine.py ../combine/datacards/run2/gmsb --m gmsb --mchi 650 --force
"""

import os
import sys
import re
import csv
import glob
import json
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import config

CARD_RE = re.compile(r'^datacard_mchi(\d+)_mlsp(\d+)_(\w+)_(gmsb|bino)\.txt$')


# ── Datacard discovery ────────────────────────────────────────────────────────

def find_cards(card_dir, model):
    """Return (mchi, mlsp, era, path) for each datacard in card_dir for this model."""
    cards = []
    for path in sorted(glob.glob(os.path.join(card_dir, 'datacard_*.txt'))):
        m = CARD_RE.match(os.path.basename(path))
        if m and m.group(4) == model:
            cards.append((int(m.group(1)), int(m.group(2)), m.group(3), path))
    return cards


# ── Combine runner ────────────────────────────────────────────────────────────

def run_one(args):
    """Run Combine on one datacard. Returns result dict."""
    mchi, mlsp, era, model, card_path, out_dir, force = args

    result_json = os.path.join(out_dir,
                               f"result_mchi{mchi}_mlsp{mlsp}_{model}.json")
    if not force and os.path.exists(result_json):
        with open(result_json) as f:
            return json.load(f)

    cmd = [
        "combine", "-M", "AsymptoticLimits",
        card_path,
        "--run", "expected",
        "-t", "-1",
        "--expectSignal", "0",
        "-n", f"_mchi{mchi}_mlsp{mlsp}_{era}",
        "--noFitAsimov",
    ]
    try:
        # print(cmd)
        # printable = ' '.join(cmd)
        # print(printable)
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=out_dir, timeout=600)
    except subprocess.TimeoutExpired:
        return {'mchi': mchi, 'mlsp': mlsp, 'status': 'timeout'}
    except FileNotFoundError:
        return {'mchi': mchi, 'mlsp': mlsp, 'status': 'combine command not found: try running cmsenv'}

    if proc.returncode != 0:
        return {'mchi': mchi, 'mlsp': mlsp, 'status': 'failed',
                'stderr': proc.stderr[-600:]}

    limits = parse_limits(proc.stdout)
    if not limits:
        return {'mchi': mchi, 'mlsp': mlsp, 'status': 'parse_failed',
                'stdout': proc.stdout[-300:]}

    result = {
        'mchi': mchi, 'mlsp': mlsp,
        'era': era, 'model': model,
        'status': 'ok',
        **limits,
    }
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
    cols = ['mchi', 'mlsp', 'era', 'model', 'status',
            'exp_m2', 'exp_m1', 'exp', 'exp_p1', 'exp_p2']
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in sorted(results, key=lambda x: (x['mchi'], x['mlsp'])):
            w.writerow(r)
    print(f"  Saved: {csv_path}")


def merge_csv(results, csv_path, replaced):
    """Rows of the existing csv_path whose (mchi, mlsp) is not in replaced, plus results."""
    if not os.path.exists(csv_path):
        return results
    with open(csv_path, newline='') as f:
        old = [{**row, 'mchi': int(row['mchi']), 'mlsp': int(row['mlsp'])}
               for row in csv.DictReader(f)]
    return [r for r in old if (r['mchi'], r['mlsp']) not in replaced] + results


# ── Entry point ───────────────────────────────────────────────────────────────

def mlsp_arg(s):
    """--mlsp value: an int, or 'all' (every mlsp in card_dir; gmsb: mlsp=1)."""
    return s if s == "all" else int(s)


def main():
    parser = argparse.ArgumentParser(description="Run Combine limits")
    parser.add_argument("card_dir", help="Directory of datacards, e.g. combine/datacards/run2/gmsb")
    parser.add_argument("--m",     dest="model", choices=config.MODELS, required=True,
                        help="'gmsb' implies mlsp=1; 'bino' requires --mlsp")
    parser.add_argument("--mchi",  type=int, default=None,
                        help="Only run datacards with this mchi")
    parser.add_argument("--mlsp",  type=mlsp_arg, default=None,
                        help="Only run datacards with this mlsp; 'all' runs every mlsp "
                             "in card_dir, the full 2D plane (gmsb: only mlsp=1)")
    parser.add_argument("--jobs",  type=int, default=1,
                        help="Parallel jobs (default: 1)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mlsp == "all" and args.model == "gmsb":
        args.mlsp = config.GMSB_MLSP  # gmsb's only mlsp
    scan_2d = args.mlsp == "all"
    if not scan_2d:
        try:
            args.mlsp = config.model_mlsp(args.model, args.mlsp)
        except ValueError as e:
            parser.error(str(e))

    card_dir = os.path.abspath(args.card_dir)
    cards = find_cards(card_dir, args.model)
    if args.mchi is not None:
        cards = [c for c in cards if c[0] == args.mchi]
    if not scan_2d:
        cards = [c for c in cards if c[1] == args.mlsp]
    subset = args.mchi is not None or not scan_2d
    if not cards:
        where = f" with mchi={args.mchi} mlsp={args.mlsp}" if subset else ""
        sys.exit(f"ERROR: no {args.model} datacards{where} found in {card_dir}")

    eras = {c[2] for c in cards}
    if len(eras) != 1:
        sys.exit(f"ERROR: datacards in {card_dir} mix eras {sorted(eras)}")
    era = eras.pop()

    out_dir = os.path.join(config.LIMITS_DIR, era)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {card_dir}  era={era}  m={args.model}  mlsp={args.mlsp} ===")
    print(f"Running {len(cards)} jobs (parallel={args.jobs})", flush=True)

    job_args = [(mchi, mlsp, era, args.model, path, out_dir, args.force)
                for mchi, mlsp, _, path in cards]

    ok_results = []
    n_fail = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for r in ex.map(run_one, job_args):
            mc, ml = r['mchi'], r['mlsp']
            if r['status'] == 'ok':
                ok_results.append(r)
                print(f"  OK  mchi={mc:4d} mlsp={ml:4d}  "
                      f"exp={r['exp']:.4f}  "
                      f"[-2s={r['exp_m2']:.4f}  +2s={r['exp_p2']:.4f}]",
                      flush=True)
            else:
                n_fail += 1
                print(f"  !!  mchi={mc} mlsp={ml}  FAILED: {r['status']}", flush=True)
                if 'stderr' in r:
                    print(f"      {r['stderr'][-300:]}")

    print(f"\n  Done: {len(ok_results)} OK  {n_fail} failed")

    if ok_results:
        csv_path = os.path.join(config.LIMITS_DIR,
                                f"limits_{era}_{args.model}_mlsp{args.mlsp}.csv")
        rows = ok_results
        if subset:
            # Keep the CSV's rows for mass points outside this run
            rows = merge_csv(ok_results, csv_path, {(c[0], c[1]) for c in cards})
        save_csv(rows, csv_path)
        if scan_2d:
            print(f"\n  Plot: python3 plot.py --csv {csv_path} --era {era} --m {args.model} --plot 2d")
        else:
            print(f"\n  Plot: python3 plot.py --csv {csv_path} --era {era} --m {args.model} --plot 1d --mlsp {args.mlsp}")


if __name__ == "__main__":
    main()
