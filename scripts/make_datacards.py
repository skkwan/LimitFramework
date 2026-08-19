#!/usr/bin/env python3
"""
make_datacards.py
Writes Combine datacards for each (mchi, mlsp, era) mass point.

Single signal process (TChiZH), single signal region (SR) -- unlike the
H(gg)+H/Z(bb) channel this framework was originally scaffolded for, there's
no HH/ZH branching-fraction mixing and no SRHH/SRZH split here.

Xsec mode (--xsec):
  '2d' (default): yields as-is from CSV (uses sigma_2D per mass point)
  '1d':           yields scaled by sigma_1D/sigma_2D per mass point
                  Use for 1D limit plots where theory is sigma_1D (full higgsino)

Usage:
    python3 scripts/make_datacards.py --era run2 --xsec 2d
    python3 scripts/make_datacards.py --era run2 --xsec 1d
    python3 scripts/make_datacards.py --era run2 --mchi 650 --mlsp 1 --force
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config


# ── Era settings ──────────────────────────────────────────────────────────────

LUMI_UNC  = {'run2': 1.02, 'run3': 1.02, 'run2run3': 1.02}
YIELD_COL = {'run2': 'yield_run2', 'run3': 'yield_run3_with2025', 'run2run3': 'yield_total_with2025'}


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_signal_csv():
    """Load CSV, remove duplicates, keep only final cut per (mchi, mlsp)."""
    if not os.path.exists(config.SIGNAL_CSV):
        raise FileNotFoundError(f"signal_master.csv not found: {config.SIGNAL_CSV}")
    df = pd.read_csv(config.SIGNAL_CSV)
    df = df.drop_duplicates(subset=['mchi', 'mlsp', 'cut_index']).copy()
    idx = df.groupby(['mchi', 'mlsp'])['cut_index'].idxmax()
    return df.loc[idx].reset_index(drop=True)


def get_yield(df, mchi, mlsp, era_tag):
    """Raw yield from CSV for one (mchi, mlsp)."""
    col  = YIELD_COL[era_tag]
    rows = df[(df['mchi'] == mchi) & (df['mlsp'] == mlsp)]
    return 0.0 if rows.empty else float(rows[col].iloc[0])


def get_xsec_ratio(df, mchi):
    """
    sigma_1d / sigma_2d for this mchi (~5 for most masses).
    For --xsec 1d: multiply the yield by this to convert from
    sigma_2D-based to sigma_1D-based expected events.
    """
    rows = df[df['mchi'] == mchi]
    if rows.empty:
        return 1.0
    xsec_2d = float(rows['xsec_2d'].iloc[0])
    xsec_1d = float(rows['xsec_1d'].iloc[0])
    return xsec_1d / xsec_2d if xsec_2d > 0 else 1.0


# ── Path helpers ──────────────────────────────────────────────────────────────

def sig_ws_path(mchi, mlsp, era_tag):
    return os.path.abspath(config.sig_fit_ws(mchi, mlsp, era_tag))

def bkg_ws_path(era_tag):
    return os.path.abspath(config.combined_bkg_ws(era_tag))


# ── Datacard writer ───────────────────────────────────────────────────────────

def write_datacard(mchi, mlsp, era_tag, df, xsec_mode, force=False):
    out_path = config.datacard_path(mchi, mlsp, era_tag, xsec_mode)

    if not force and os.path.exists(out_path):
        print(f"  EXISTS (skip): {os.path.basename(out_path)}")
        return True

    r_raw = get_yield(df, mchi, mlsp, era_tag)
    xsec_factor = get_xsec_ratio(df, mchi) if xsec_mode == '1d' else 1.0
    r = r_raw * xsec_factor

    if r <= 0:
        print(f"  SKIP mchi={mchi} mlsp={mlsp}: yield zero after scaling")
        return False

    sig_ws   = sig_ws_path(mchi, mlsp, era_tag)
    sig_name = f"ws_sig_mchi{mchi}_mlsp{mlsp}"
    sig_pdf  = f"sig_pdf_mchi{mchi}_mlsp{mlsp}_{era_tag}"

    bkg_ws   = bkg_ws_path(era_tag)
    bkg_name = f"ws_bkg_{era_tag}"

    missing = [(p, l) for p, l in [(sig_ws, 'sig'), (bkg_ws, 'bkg')] if not os.path.exists(p)]
    if missing:
        print(f"  SKIP mchi={mchi} mlsp={mlsp}: missing: {[l for _, l in missing]}")
        return False

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # CombineHarvester's datacard parser joins the shapes path onto the
    # datacard's own directory even when it's already absolute, so the
    # shapes lines must use paths relative to out_dir.
    sig_ws_rel = os.path.relpath(sig_ws, out_dir)
    bkg_ws_rel = os.path.relpath(bkg_ws, out_dir)

    with open(out_path, 'w') as f:
        f.write(f"# Higgsino Z(ll)H(bb) datacard\n")
        f.write(f"# mchi={mchi} mlsp={mlsp} era={era_tag} xsec={xsec_mode}\n")
        f.write(f"# Yield: sig={r:.4f}\n")
        f.write(f"# Run: combine -M AsymptoticLimits <card> -t -1 --expectSignal 0\n")

        f.write(f"\nimax 1\njmax *\nkmax *\n\n---\n\n")
        f.write(f"shapes sig      SR  {sig_ws_rel}  {sig_name}:{sig_pdf}\n")
        f.write(f"shapes bkg      SR  {bkg_ws_rel}  {bkg_name}:bkg_total_{era_tag}\n")
        f.write(f"shapes data_obs SR  {bkg_ws_rel}  {bkg_name}:data_obs\n")
        f.write(f"\n---\n\n")
        f.write(f"bin         SR\n")
        f.write(f"observation -1\n")
        f.write(f"\n---\n\n")
        f.write(f"bin      SR   SR\n")
        f.write(f"process  sig  bkg\n")
        f.write(f"process  0    1\n")
        f.write(f"rate     {r:.6f}  1.0\n")

    print(f"  Wrote: {os.path.basename(out_path)}  [xsec_factor={xsec_factor:.2f}  sig={r:.3f}]")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Write Combine datacards")
    parser.add_argument("--era",   choices=["run2", "run3", "run2run3", "both"],
                        default="both")
    parser.add_argument("--xsec",  choices=["2d", "1d"], default="2d",
                        help="'2d': yields as-is from CSV (default); "
                             "'1d': scale by sigma_1d/sigma_2d for 1D limit plots")
    parser.add_argument("--mchi",  type=int, default=None)
    parser.add_argument("--mlsp",  type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    eras = ["run2", "run3"] if args.era == "both" else [args.era]

    print(f"Settings: xsec={args.xsec}", flush=True)
    print("Loading signal_master.csv...", flush=True)
    df = load_signal_csv()
    print(f"  {len(df)} rows at final cut", flush=True)

    mass_points = sorted(config.SIGNAL_GRID)
    if args.mchi is not None:
        mass_points = [(mc, ml) for mc, ml in mass_points if mc == args.mchi]
    if args.mlsp is not None:
        mass_points = [(mc, ml) for mc, ml in mass_points if ml == args.mlsp]
    print(f"Mass points: {len(mass_points)}", flush=True)

    for era in eras:
        print(f"\n=== era={era} ===")
        n_ok = n_skip = n_fail = 0
        for mchi, mlsp in mass_points:
            try:
                ok = write_datacard(mchi, mlsp, era, df, xsec_mode=args.xsec, force=args.force)
                if ok:  n_ok   += 1
                else:   n_skip += 1
            except Exception as e:
                print(f"  ERROR mchi={mchi} mlsp={mlsp}: {e}")
                import traceback; traceback.print_exc()
                n_fail += 1

        print(f"\n  Done: {n_ok} written  {n_skip} skipped  {n_fail} failed")
        if n_ok > 0 and mass_points:
            card_dir = os.path.join(config.CARDS_DIR, era)
            print(f"\n  Validate:")
            print(f"    cd {card_dir}")
            example = config.datacard_path(mass_points[0][0], mass_points[0][1], era, args.xsec)
            print(f"    ValidateDatacards.py {os.path.basename(example)}")
            print(f"    combine -M AsymptoticLimits {os.path.basename(example)} "
                  f"-t -1 --expectSignal 0")


if __name__ == "__main__":
    main()
