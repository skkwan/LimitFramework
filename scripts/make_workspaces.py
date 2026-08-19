#!/usr/bin/env python3
"""
make_workspaces.py
Prepares the background RooFit workspace for Combine.

What this script does, per era (run2 / run3):
  - Loads the background PDF from fit_background.py's output (fit to the
    combined peaking+non-peaking background-MC sample -- see
    fit_background.py/fit_zpeak.py; there's no real data ntuple in this
    framework yet). All of the PDF's shape/mixture parameters already
    arrive frozen (set constant in fit_background.py).
  - Adds a bkg_total_{era}_norm RooRealVar fixed at config.BKG_NORM[era]
    (not the MC-weighted n_bkg_mc) and sets it constant, so the entire
    background (shape and normalization) is fixed -- only the signal
    strength r floats in the fit.
  - Builds data_obs as a background-only Asimov PSEUDO-dataset, generated
    from the fitted bkg_total pdf with n_bkg_mc events -- a placeholder
    until real (unblinded) data ntuples exist. Run Combine with -t -1 for
    an Asimov-based expected limit (which regenerates its own Asimov
    dataset from the model anyway; data_obs here just needs to exist with
    the right observables).
  - Saves everything to one background workspace ROOT file

Signal workspaces already exist from fit_signal.py and are referenced
directly by make_datacards.py without modification.

Output:
  combine/workspaces/combined/bkg_workspace_{era}.root
    workspace name: ws_bkg_{era}
    contents:
      bkg_total_{era}         <- background PDF (all params fixed)
      bkg_total_{era}_norm    <- RooRealVar, fixed at config.BKG_NORM[era]
      data_obs                <- RooDataSet (mll, met), Asimov pseudo-data

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


# ── Pseudo-data ───────────────────────────────────────────────────────────────

def build_data_obs(bkg_pdf, mll_var, met_var, n_events_target):
    """
    No real (unblinded) data ntuples exist yet in this framework -- generate
    a background-only Asimov pseudo-dataset from the fitted bkg_total pdf as
    a stand-in for data_obs, with n_events_target (config.BKG_NORM[era] --
    the same fixed normalization Combine uses for the background rate)
    events. Replace with a real data_obs once actual data ntuples exist.
    """
    n_events = max(1, round(n_events_target))
    ds_obs = bkg_pdf.generate(ROOT.RooArgSet(mll_var, met_var), n_events)
    ds_obs.SetName("data_obs")
    ds_obs.SetTitle("data_obs")
    return ds_obs


# ── Background workspace builder ──────────────────────────────────────────────

def make_bkg_workspace(era_tag, force=False):
    out_path = config.combined_bkg_ws(era_tag)

    if not force and os.path.exists(out_path):
        print(f"  EXISTS (skip): {os.path.basename(out_path)}")
        return out_path

    mll_var = ROOT.RooRealVar("mll", "mll", config.MLL_LO, config.MLL_HI)
    met_var = ROOT.RooRealVar("met", "met", config.MET_LO, config.MET_HI)

    ws = ROOT.RooWorkspace(f"ws_bkg_{era_tag}", f"ws_bkg_{era_tag}")
    live = [mll_var, met_var]

    # ── Load background fit JSON (for n_bkg_mc) ──
    json_path = config.bkg_fit_json(era_tag)
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Background fit JSON not found: {json_path}\n"
            f"Run fit_background.py --era {era_tag} first.")
    with open(json_path) as f:
        bkg_info = json.load(f)
    n_bkg_mc = bkg_info['n_bkg_mc']
    print(f"    n_bkg_mc = {n_bkg_mc:.4f}", flush=True)

    # ── Load background fit workspace (for the PDF) ──
    ws_path = config.bkg_fit_ws(era_tag)
    if not os.path.exists(ws_path):
        raise FileNotFoundError(
            f"Background fit workspace not found: {ws_path}\n"
            f"Run fit_background.py --era {era_tag} first.")

    bkg_file = ROOT.TFile(ws_path, 'READ')
    bkg_ws   = bkg_file.Get(f"ws_bkg_{era_tag}")
    if not bkg_ws:
        bkg_file.Close()
        raise RuntimeError(f"Workspace ws_bkg_{era_tag} not found in {ws_path}")

    pdf_name = f"bkg_total_{era_tag}"
    bkg_pdf  = bkg_ws.pdf(pdf_name)
    if not bkg_pdf:
        bkg_file.Close()
        raise RuntimeError(f"PDF {pdf_name} not found in {ws_path}")

    # Import PDF — RecycleConflictNodes merges shared params (mll, met)
    # rather than duplicating them
    ws.Import(bkg_pdf, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    bkg_file.Close()

    # ── Add _norm parameter, fixed (not derived from n_bkg_mc) ──
    norm_name  = f"{pdf_name}_norm"
    bkg_norm   = config.BKG_NORM[era_tag]
    norm_var   = ROOT.RooRealVar(norm_name, norm_name, bkg_norm)
    ws.Import(norm_var)
    live.append(norm_var)
    print(f"    Added {norm_name} = {bkg_norm:.4f}", flush=True)

    # ── Build data_obs (Asimov pseudo-data -- see build_data_obs docstring) ──
    # Use the copy already imported into our own `ws` (safe to use after
    # bkg_file.Close(), unlike the original file-backed `bkg_pdf`).
    print(f"    Building data_obs (Asimov pseudo-data from bkg_total)...", flush=True)
    ds_obs = build_data_obs(ws.pdf(pdf_name), mll_var, met_var, bkg_norm)
    ws.Import(ds_obs)
    live.append(ds_obs)
    print(f"    data_obs: {ds_obs.numEntries()} events", flush=True)

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
    for mchi, mlsp in config.SIGNAL_GRID:
        ws_path = config.sig_fit_ws(mchi, mlsp, era_tag)
        if not os.path.exists(ws_path):
            missing.append(ws_path)
    if missing:
        print(f"  WARNING: {len(missing)} signal workspaces missing for era={era_tag}.")
        print(f"  First few: {missing[:3]}")
        print(f"  Run fit_signal.py --era {era_tag} --force to generate them.")
    else:
        print(f"  Signal workspaces: {len(config.SIGNAL_GRID)} mass points — all present.")
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
