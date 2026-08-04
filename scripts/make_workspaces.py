#!/usr/bin/env python3
"""
make_workspaces.py
Prepares the background RooFit workspace for Combine.

What this script does:
  For each era (run2 / run3):
    - Loads background PDFs from fit_background.py outputs (SRHH + SRZH)
    - Adds {pdf_name}_norm RooRealVars initialized to bkg_estimate_full
      (the sideband-fit-extrapolated background estimate from
      fit_background.py -- NOT the real observed full-range data count.
      Combine multiplies rate=1 by _norm, so the background floats freely)
    - Builds data_obs RooDataSets from data slim files (mbb cut + explicit
      mgg-sideband cut as defense-in-depth; slims/data/ is already
      hard-blinded at creation by make_slims.py, so this is normally a
      no-op). Run Combine with -t -1 while blinded.
    - Saves everything to one background workspace ROOT file

Signal workspaces already exist from fit_signal.py and are referenced
directly by make_datacards.py without modification.

Output:
  combine/workspaces/combined/bkg_workspace_{era}.root
    workspace name: ws_bkg_{era}
    contents:
      bkg_total_SRHH_{era}         <- background PDF (floating params)
      bkg_total_SRZH_{era}         <- background PDF (floating params)
      bkg_total_SRHH_{era}_norm    <- RooRealVar, initialized to n_data_SRHH
      bkg_total_SRZH_{era}_norm    <- RooRealVar, initialized to n_data_SRZH
      data_obs_SRHH                <- RooDataSet (mgg, met) with SRHH mbb cut
      data_obs_SRZH                <- RooDataSet (mgg, met) with SRZH mbb cut

Usage:
    python3 scripts/make_workspaces.py --era run2
    python3 scripts/make_workspaces.py --era run3
    python3 scripts/make_workspaces.py           # both eras
    python3 scripts/make_workspaces.py --force   # overwrite existing
"""

import ROOT
import json
import os
import sys
import argparse

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))
import config

REGIONS = list(config.SIGNAL_REGIONS.keys())   # ['SRHH', 'SRZH']


# ── Data loading ──────────────────────────────────────────────────────────────

def build_data_obs(region, years, mgg_var, met_var):
    """
    Build data_obs RooDataSet for one SR from slim data files.

    Applies mbb cut plus an explicit mgg-sideband cut. The sideband cut is
    defense-in-depth: slims/data/ is already hard-blinded at creation
    (make_slims.py drops SR-window events from the file entirely), so this
    should normally be a no-op. It's kept explicit so data_obs itself is
    blind-safe even if ever built from an unblinded or externally-provided
    slim. Run Combine with -t -1 to use Asimov while the SR is blinded.
    """
    files = config.slim_data_files(years)
    if not files:
        raise RuntimeError(f"No data slim files found for years {years}")

    chain = ROOT.TChain("tree")
    for f in files:
        chain.Add(f)

    mbb_lo = config.SIGNAL_REGIONS[region]['mbb_lo']
    mbb_hi = config.SIGNAL_REGIONS[region]['mbb_hi']

    mbb_var = ROOT.RooRealVar("mbb", "mbb", 0.0, 500.0)
    obs_all = ROOT.RooArgSet(mgg_var, met_var, mbb_var)
    obs_2d  = ROOT.RooArgSet(mgg_var, met_var)

    cut = (f"mbb > {mbb_lo} && mbb < {mbb_hi} && "
           f"{config.sideband_cut_expr('mgg')}")
    ds_full = ROOT.RooDataSet("ds_tmp", "ds_tmp", chain, obs_all, cut)

    # Drop mbb — Combine only sees the fit observables mgg and met
    ds_obs = ds_full.reduce(obs_2d)
    ds_obs.SetName(f"data_obs_{region}")
    ds_obs.SetTitle(f"data_obs_{region}")

    return ds_obs


# ── Background workspace builder ──────────────────────────────────────────────

def make_bkg_workspace(era_tag, force=False):
    out_path = os.path.join(config.WS_DIR, 'combined',
                            f'bkg_workspace_{era_tag}.root')

    if not force and os.path.exists(out_path):
        print(f"  EXISTS (skip): {os.path.basename(out_path)}")
        return out_path

    years = config.era_years(era_tag)

    # Shared observables — same range in both SRs so Combine allows sharing
    mgg_var = ROOT.RooRealVar("mgg", "mgg", config.MGG_LO, config.MGG_HI)
    met_var = ROOT.RooRealVar("met", "met", config.MET_LO, config.MET_HI)

    ws = ROOT.RooWorkspace(f"ws_bkg_{era_tag}", f"ws_bkg_{era_tag}")

    # Keep Python objects alive until ws is written
    live = [mgg_var, met_var]

    for region in REGIONS:
        print(f"  Processing {region}...", flush=True)

        # ── Load background fit JSON (for bkg_estimate_full) ──
        json_path = config.bkg_fit_json(region, era_tag)
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Background fit JSON not found: {json_path}\n"
                f"Run fit_background.py --region {region} --era {era_tag} first.")

        with open(json_path) as f:
            bkg_info = json.load(f)
        # bkg_estimate_full is extrapolated from the sideband fit (see
        # fit_background.py) -- it is NOT the real observed data count in
        # the full mgg range, which would include the blinded SR window.
        n_data = bkg_info['bkg_estimate_full']
        print(f"    bkg_estimate_full = {n_data:.1f}  "
              f"(sideband n={bkg_info['n_data_sideband']}, "
              f"sideband_frac={bkg_info['sideband_frac']:.4f})", flush=True)

        # ── Load background fit workspace (for PDF) ──
        ws_path = config.bkg_fit_ws(region, era_tag)
        if not os.path.exists(ws_path):
            raise FileNotFoundError(
                f"Background fit workspace not found: {ws_path}\n"
                f"Run fit_background.py --region {region} --era {era_tag} first.")

        bkg_file = ROOT.TFile(ws_path, 'READ')
        bkg_ws   = bkg_file.Get(f"ws_bkg_{region}_{era_tag}")
        if not bkg_ws:
            bkg_file.Close()
            raise RuntimeError(f"Workspace ws_bkg_{region}_{era_tag} not found in {ws_path}")

        pdf_name = f"bkg_total_{region}_{era_tag}"
        bkg_pdf  = bkg_ws.pdf(pdf_name)
        if not bkg_pdf:
            bkg_file.Close()
            raise RuntimeError(f"PDF {pdf_name} not found in {ws_path}")

        # Import PDF — RecycleConflictNodes merges shared params (mgg, met)
        # rather than duplicating them
        ws.Import(bkg_pdf, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
        bkg_file.Close()

        # ── Add _norm parameter ──
        # Combine multiplies rate=1 by {pdf}_norm, so this is the floating
        # background normalisation. Range: [0, 5×n_data] gives plenty of room.
        norm_name = f"{pdf_name}_norm"
        norm_var  = ROOT.RooRealVar(norm_name, norm_name,
                                    n_data, 0.0, 5.0 * n_data)
        ws.Import(norm_var)
        live.append(norm_var)
        print(f"    Added {norm_name} = {n_data:.1f}", flush=True)

        # ── Build data_obs ──
        print(f"    Building data_obs (slow for large data files)...", flush=True)
        ds_obs = build_data_obs(region, years, mgg_var, met_var)
        ws.Import(ds_obs)
        live.append(ds_obs)
        print(f"    data_obs_{region}: {ds_obs.numEntries()} events", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ws.writeToFile(out_path)
    print(f"  Saved: {out_path}", flush=True)
    return out_path


# ── Signal workspace check ────────────────────────────────────────────────────

def check_signal_workspaces(era_tag):
    """
    Verify signal workspaces exist for all mass points.
    These are produced by fit_signal.py — we don't modify them,
    just check they're there so make_datacards.py won't fail.
    """
    missing = []
    for sigtype in ['HH', 'ZH']:
        for mchi, mlsp in config.SIGNAL_GRID[sigtype]:
            ws_path = config.sig_fit_ws(sigtype, mchi, mlsp, era_tag)
            if not os.path.exists(ws_path):
                missing.append(ws_path)
    if missing:
        print(f"  WARNING: {len(missing)} signal workspaces missing for era={era_tag}.")
        print(f"  First few: {missing[:3]}")
        print(f"  Run fit_signal.py --era {era_tag} --force to generate them.")
    else:
        n_HH = len(config.SIGNAL_GRID['HH'])
        n_ZH = len(config.SIGNAL_GRID['ZH'])
        print(f"  Signal workspaces: {n_HH} HH + {n_ZH} ZH — all present.")
    return len(missing) == 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build background workspace for Combine")
    parser.add_argument("--era",   choices=["run2", "run3", "both"], default="both")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing workspaces")
    args = parser.parse_args()

    eras = ["run2", "run3"] if args.era == "both" else [args.era]

    for era in eras:
        print(f"\n=== Background workspace: era={era} ===")
        try:
            out = make_bkg_workspace(era, force=args.force)
            print(f"  OK: {out}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

        print(f"\n=== Checking signal workspaces: era={era} ===")
        check_signal_workspaces(era)

    print("\nDone. Next step: make_datacards.py")


if __name__ == "__main__":
    main()