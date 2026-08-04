#!/usr/bin/env python3
"""
make_datacards.py
Writes Combine datacards for each (mchi, mlsp, era) mass point.

Signal structure: sig_HH (process 0) and sig_ZH (process -1) in both SRs.
Combine scales both by the same mu.  6 columns per datacard:
  SRHH_sig_HH  SRHH_sig_ZH  SRHH_bkg  SRZH_sig_HH  SRZH_sig_ZH  SRZH_bkg

BF scaling (--bf):
  CSV yield_HH assumes B(chi->H) = 1   -> scale by bf^2
  CSV yield_ZH assumes B(chi->H) = 0.5 -> scale by 4*bf*(1-bf)
  bf=1.0 (default): pure HH, ZH yields zero
  bf=0.5:           equal mixing, HH scaled x0.25, ZH unscaled

Xsec mode (--xsec):
  '2d' (default): yields as-is from CSV (uses sigma_2D per mass point)
  '1d':           yields scaled by sigma_1D/sigma_2D (~5) per mass point
                  Use for 1D limit plots where theory is sigma_1D (full higgsino)

Usage:
    python3 scripts/make_datacards.py --era run2 --bf 1.0 --xsec 2d  # 2D limits, pure HH
    python3 scripts/make_datacards.py --era run2 --bf 0.5 --xsec 2d  # 2D limits, equal mixing
    python3 scripts/make_datacards.py --era run2 --bf 1.0 --xsec 1d  # 1D limits, pure HH
    python3 scripts/make_datacards.py --era run2 --mchi 300 --mlsp 0 --force
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
    """Load CSV, deduplicate, keep only final cut per (region, sigtype, mchi, mlsp)."""
    if not os.path.exists(config.SIGNAL_CSV):
        raise FileNotFoundError(f"signal_master.csv not found: {config.SIGNAL_CSV}")
    df = pd.read_csv(config.SIGNAL_CSV)
    df = df.drop_duplicates(subset=['region', 'sigtype', 'mchi', 'mlsp', 'cut_index']).copy()
    idx = df.groupby(['region', 'sigtype', 'mchi', 'mlsp'])['cut_index'].idxmax()
    return df.loc[idx].reset_index(drop=True)


def get_yield(df, sigtype, region, mchi, mlsp, era_tag):
    """Raw yield from CSV for one (sigtype, region, mchi, mlsp)."""
    col  = YIELD_COL[era_tag]
    mask = ((df['sigtype'] == sigtype) & (df['region'] == region) &
            (df['mchi'] == mchi) & (df['mlsp'] == mlsp))
    rows = df[mask]
    return 0.0 if rows.empty else float(rows[col].iloc[0])


def get_xsec_ratio(df, mchi):
    """
    sigma_1d / sigma_2d for this mchi (~5 for most masses).
    For --xsec 1d: multiply all yields by this to convert from
    sigma_2D-based to sigma_1D-based expected events.
    """
    rows = df[df['mchi'] == mchi]
    if rows.empty:
        return 1.0
    xsec_2d = float(rows['xsec_2d'].iloc[0])
    xsec_1d = float(rows['xsec_1d'].iloc[0])
    return xsec_1d / xsec_2d if xsec_2d > 0 else 1.0


# ── BF scaling ────────────────────────────────────────────────────────────────

def apply_bf(r_HH_raw, r_ZH_raw, bf):
    """
    Scale yields for B(chi->H) = bf.
      yield_HH = yield_HH_csv * bf^2           (CSV assumes bf=1)
      yield_ZH = yield_ZH_csv * 4*bf*(1-bf)    (CSV assumes bf=0.5)

    bf=1.0 -> HH scale=1,    ZH scale=0    (pure HH, no ZH signal)
    bf=0.5 -> HH scale=0.25, ZH scale=1    (equal mixing, ZH as-is)
    """
    return r_HH_raw * bf**2, r_ZH_raw * 4.0 * bf * (1.0 - bf)


# ── Path helpers ──────────────────────────────────────────────────────────────

def sig_ws_path(sigtype, mchi, mlsp, era_tag):
    return os.path.abspath(config.sig_fit_ws(sigtype, mchi, mlsp, era_tag))

def bkg_ws_path(era_tag):
    return os.path.abspath(config.combined_bkg_ws(era_tag))


# ── Datacard writer ───────────────────────────────────────────────────────────

def write_datacard(mchi, mlsp, era_tag, df, bf, xsec_mode, force=False):
    out_path = config.datacard_path(mchi, mlsp, era_tag, bf, xsec_mode)

    if not force and os.path.exists(out_path):
        print(f"  EXISTS (skip): {os.path.basename(out_path)}")
        return True

    # ── Raw yields from CSV ──────────────────────────────────────────────────
    r_HH_SRHH_raw = get_yield(df, 'HH', 'hhregion', mchi, mlsp, era_tag)
    r_ZH_SRHH_raw = get_yield(df, 'ZH', 'hhregion', mchi, mlsp, era_tag)
    r_HH_SRZH_raw = get_yield(df, 'HH', 'zhregion', mchi, mlsp, era_tag)
    r_ZH_SRZH_raw = get_yield(df, 'ZH', 'zhregion', mchi, mlsp, era_tag)

    # ── BF scaling ──────────────────────────────────────────────────────────
    r_HH_SRHH, r_ZH_SRHH = apply_bf(r_HH_SRHH_raw, r_ZH_SRHH_raw, bf)
    r_HH_SRZH, r_ZH_SRZH = apply_bf(r_HH_SRZH_raw, r_ZH_SRZH_raw, bf)

    # ── Xsec scaling ────────────────────────────────────────────────────────
    xsec_factor = get_xsec_ratio(df, mchi) if xsec_mode == '1d' else 1.0
    r_HH_SRHH *= xsec_factor;  r_ZH_SRHH *= xsec_factor
    r_HH_SRZH *= xsec_factor;  r_ZH_SRZH *= xsec_factor

    if r_HH_SRHH <= 0 and r_ZH_SRZH <= 0:
        print(f"  SKIP mchi={mchi} mlsp={mlsp}: primary yields zero after scaling")
        return False

    # ── Determine whether to include ZH process ─────────────────────────────
    # Must be done before workspace check so we don't require ZH ws when not needed
    include_ZH = (r_ZH_SRHH + r_ZH_SRZH) > 1e-6

    # ── Workspace paths ──────────────────────────────────────────────────────
    bkg_ws   = bkg_ws_path(era_tag)
    bkg_name = f"ws_bkg_{era_tag}"

    sig_HH_ws   = sig_ws_path('HH', mchi, mlsp, era_tag)
    sig_HH_name = f"ws_sig_HH_mchi{mchi}_mlsp{mlsp}"
    sig_HH_pdf  = f"sig_pdf_HH_mchi{mchi}_mlsp{mlsp}"

    sig_ZH_ws   = sig_ws_path('ZH', mchi, mlsp, era_tag)
    sig_ZH_name = f"ws_sig_ZH_mchi{mchi}_mlsp{mlsp}"
    sig_ZH_pdf  = f"sig_pdf_ZH_mchi{mchi}_mlsp{mlsp}"

    to_check = [(bkg_ws, 'bkg'), (sig_HH_ws, 'sig_HH')]
    if include_ZH:
        to_check.append((sig_ZH_ws, 'sig_ZH'))
    missing = [(p, l) for p, l in to_check if not os.path.exists(p)]
    if missing:
        print(f"  SKIP mchi={mchi} mlsp={mlsp}: missing: {[l for _,l in missing]}")
        return False

    lumi = LUMI_UNC[era_tag]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(f"# Higgsino datacard\n")
        f.write(f"# mchi={mchi} mlsp={mlsp} era={era_tag} bf={bf:.3f} xsec={xsec_mode}\n")
        f.write(f"# BF: HH x{bf**2:.4f}, ZH x{4*bf*(1-bf):.4f}  sig_ZH included={include_ZH}\n")
        f.write(f"# Yields: HH_SRHH={r_HH_SRHH:.4f} ZH_SRHH={r_ZH_SRHH:.4f} "
                f"HH_SRZH={r_HH_SRZH:.4f} ZH_SRZH={r_ZH_SRZH:.4f}\n")
        f.write(f"# Run: combine -M AsymptoticLimits <card> -t -1 --expectSignal 0\n")

        if include_ZH:
            f.write(f"\nimax 2\njmax *\nkmax *\n\n---\n\n")
            f.write(f"shapes sig_HH    SRHH  {sig_HH_ws}  {sig_HH_name}:{sig_HH_pdf}\n")
            f.write(f"shapes sig_HH    SRZH  {sig_HH_ws}  {sig_HH_name}:{sig_HH_pdf}\n")
            f.write(f"shapes sig_ZH    SRHH  {sig_ZH_ws}  {sig_ZH_name}:{sig_ZH_pdf}\n")
            f.write(f"shapes sig_ZH    SRZH  {sig_ZH_ws}  {sig_ZH_name}:{sig_ZH_pdf}\n")
            f.write(f"shapes bkg_SRHH  SRHH  {bkg_ws}  {bkg_name}:bkg_total_SRHH_{era_tag}\n")
            f.write(f"shapes bkg_SRZH  SRZH  {bkg_ws}  {bkg_name}:bkg_total_SRZH_{era_tag}\n")
            f.write(f"shapes data_obs  SRHH  {bkg_ws}  {bkg_name}:data_obs_SRHH\n")
            f.write(f"shapes data_obs  SRZH  {bkg_ws}  {bkg_name}:data_obs_SRZH\n")
            f.write(f"\n---\n\n")
            f.write(f"bin         SRHH    SRZH\n")
            f.write(f"observation  -1      -1\n")
            f.write(f"\n---\n\n")
            f.write(f"bin      SRHH      SRHH      SRHH      SRZH      SRZH      SRZH\n")
            f.write(f"process  sig_HH    sig_ZH    bkg_SRHH  sig_HH    sig_ZH    bkg_SRZH\n")
            f.write(f"process  0         -1        1         0         -1        2\n")
            f.write(f"rate     {r_HH_SRHH:.6f}  {r_ZH_SRHH:.6f}  1.0  "
                    f"{r_HH_SRZH:.6f}  {r_ZH_SRZH:.6f}  1.0\n")
            #f.write(f"\n---\n\n")
            #f.write(f"lumi_{era_tag}  lnN  "
            #        f"{lumi:.4f}  {lumi:.4f}  -  {lumi:.4f}  {lumi:.4f}  -\n")
        else:
            f.write(f"\nimax 2\njmax *\nkmax *\n\n---\n\n")
            f.write(f"shapes sig_HH    SRHH  {sig_HH_ws}  {sig_HH_name}:{sig_HH_pdf}\n")
            f.write(f"shapes sig_HH    SRZH  {sig_HH_ws}  {sig_HH_name}:{sig_HH_pdf}\n")
            f.write(f"shapes bkg_SRHH  SRHH  {bkg_ws}  {bkg_name}:bkg_total_SRHH_{era_tag}\n")
            f.write(f"shapes bkg_SRZH  SRZH  {bkg_ws}  {bkg_name}:bkg_total_SRZH_{era_tag}\n")
            f.write(f"shapes data_obs  SRHH  {bkg_ws}  {bkg_name}:data_obs_SRHH\n")
            f.write(f"shapes data_obs  SRZH  {bkg_ws}  {bkg_name}:data_obs_SRZH\n")
            f.write(f"\n---\n\n")
            f.write(f"bin         SRHH    SRZH\n")
            f.write(f"observation  -1      -1\n")
            f.write(f"\n---\n\n")
            f.write(f"bin      SRHH      SRHH      SRZH      SRZH\n")
            f.write(f"process  sig_HH    bkg_SRHH  sig_HH    bkg_SRZH\n")
            f.write(f"process  0         1         0         2\n")
            f.write(f"rate     {r_HH_SRHH:.6f}  1.0  {r_HH_SRZH:.6f}  1.0\n")
            #f.write(f"\n---\n\n")
            #f.write(f"lumi_{era_tag}  lnN  {lumi:.4f}  -  {lumi:.4f}  -\n")

    print(f"  Wrote: {os.path.basename(out_path)}  "
          f"[xsec_factor={xsec_factor:.2f}  "
          f"HH_SRHH={r_HH_SRHH:.3f} ZH_SRHH={r_ZH_SRHH:.3f} "
          f"HH_SRZH={r_HH_SRZH:.3f} ZH_SRZH={r_ZH_SRZH:.3f}]")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Write Combine datacards")
    parser.add_argument("--era",   choices=["run2", "run3", "run2run3", "both"],
                        default="both")
    parser.add_argument("--bf",    type=float, default=1.0,
                        help="B(chi->H): 1.0=pure HH (default), 0.5=equal mixing")
    parser.add_argument("--xsec",  choices=["2d", "1d"], default="2d",
                        help="'2d': yields as-is from CSV (default); "
                             "'1d': scale by sigma_1d/sigma_2d for 1D limit plots")
    parser.add_argument("--mchi",  type=int, default=None)
    parser.add_argument("--mlsp",  type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not (0.0 <= args.bf <= 1.0):
        parser.error("--bf must be between 0 and 1")

    eras = ["run2", "run3"] if args.era == "both" else [args.era]

    print(f"Settings: bf={args.bf:.3f} -> HH x{args.bf**2:.3f}, "
          f"ZH x{4*args.bf*(1-args.bf):.3f}  |  xsec={args.xsec}", flush=True)
    print("Loading signal_master.csv...", flush=True)
    df = load_signal_csv()
    print(f"  {len(df)} rows at final cut", flush=True)

    if args.bf == 1.0:
        mass_points = sorted(config.SIGNAL_GRID['HH'])
    else:
        mass_points = sorted(set(config.SIGNAL_GRID['HH']) & set(config.SIGNAL_GRID['ZH']))

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
                ok = write_datacard(mchi, mlsp, era, df,
                                    bf=args.bf, xsec_mode=args.xsec,
                                    force=args.force)
                if ok:  n_ok   += 1
                else:   n_skip += 1
            except Exception as e:
                print(f"  ERROR mchi={mchi} mlsp={mlsp}: {e}")
                import traceback; traceback.print_exc()
                n_fail += 1

        print(f"\n  Done: {n_ok} written  {n_skip} skipped  {n_fail} failed")
        if n_ok > 0:
            card_dir = os.path.join(config.CARDS_DIR, era)
            print(f"\n  Validate:")
            print(f"    cd {card_dir}")
            example = config.datacard_path(300, 0, era, args.bf, args.xsec)
            print(f"    ValidateDatacards.py {os.path.basename(example)}")
            print(f"    combine -M AsymptoticLimits {os.path.basename(example)} "
                  f"-t -1 --expectSignal 0")


if __name__ == "__main__":
    main()