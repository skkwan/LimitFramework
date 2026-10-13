#!/usr/bin/env python3
"""
make_workspaces.py
Prepares the signal+background RooFit workspaces for Combine.

What this script does, per era (run2 / run3):
  - Loads the background component PDFs bkg_peak_{era} and bkg_nonpeak_{era}
    from fit_background.py's output (shape parameters already frozen there).
  - Makes two RooRealVars:
      ratio_peaking_{era}        the ratio r of peaking to total background,
                                 initial value ratio_peaking_mc from
                                 fit_bkg_{era}.json, NOT constant: the
                                 datacard's param line gives it a Gaussian
                                 constraint (named ratio_peaking, not r,
                                 since r is Combine's POI)
      sig_pdf_..._{era}_norm     the signal yield, fixed at the per-mass-point
                                 yield from signal_config.json (Combine scales
                                 it by the signal strength r)
    and builds
      bkg_total_{era} = ratio_peaking * bkg_peak + (1-ratio_peaking) * bkg_nonpeak
    The total background normalization (sum of config.BKG_YIELD) is not in
    the workspace: make_datacards.py writes it as the bkg rate in the datacard.
  - Fixes (setConstant) every m(ll) and MET shape parameter of the signal
    and background pdfs; make_datacards.py also states them in the datacard.
  - Builds data_obs as a background-only pseudo-dataset generated from
    bkg_total -- a placeholder until real (unblinded) data ntuples exist.
    Run Combine with -t -1 for an Asimov-based expected limit (which
    regenerates its own Asimov dataset from the model anyway; data_obs here
    just needs to exist with the right observables).
  - Saves one workspace per (mass point, era): Combine only picks up a
    <pdf>_norm from the same workspace as the pdf, so the signal pdf, its
    yield and the background model live together.

Output:
  combine/workspaces/combined/workspace_mchi{mchi}_mlsp{mlsp}_{era}.root
    workspace name: ws_{era}
    contents:
      bkg_total_{era}                    <- background PDF (shape params fixed)
      ratio_peaking_{era}                <- RooRealVar, starts at ratio_peaking_mc, floating
      sig_pdf_mchi{}_mlsp{}_{era}        <- signal PDF (from fit_signal.py, shape params fixed)
      sig_pdf_mchi{}_mlsp{}_{era}_norm   <- RooRealVar, fixed at the signal yield
      data_obs                           <- RooDataSet (mll, met), pseudo-data

Usage:
    python3 scripts/make_workspaces.py --era run2
    python3 scripts/make_workspaces.py --era run3
    python3 scripts/make_workspaces.py           # both eras
    python3 scripts/make_workspaces.py --force   # overwrite existing
    python3 scripts/make_workspaces.py --era run2 --mchi 650 --mlsp 1
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
from make_datacards import load_signal_config, get_yield


# ── Pseudo-data ───────────────────────────────────────────────────────────────

def build_data_obs(bkg_pdf, mll_var, met_var, n_events_target):
    """
    No real (unblinded) data ntuples exist yet in this framework -- generate
    a background-only pseudo-dataset from the bkg_total pdf as a stand-in
    for data_obs, with n_events_target (sum of config.BKG_YIELD -- the same
    total Combine uses for the background rate) events.
    Replace with a real data_obs once actual data ntuples exist.
    """
    n_events = max(1, round(n_events_target))
    ds_obs = bkg_pdf.generate(ROOT.RooArgSet(mll_var, met_var), n_events)
    ds_obs.SetName("data_obs")
    ds_obs.SetTitle("data_obs")
    return ds_obs


# ── Background model (once per era) ───────────────────────────────────────────

def build_bkg_model(era_tag):
    """
    Loads bkg_peak_{era} and bkg_nonpeak_{era} from fit_background.py's
    workspace and builds, in an in-memory workspace:
      ratio_peaking_{era}, bkg_total_{era}, data_obs
    Returns that workspace; make_mass_point_workspace copies from it.
    """
    # ── Load background fit JSON (for ratio_peaking_mc) ──
    json_path = config.bkg_fit_json(era_tag)
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Background fit JSON not found: {json_path}\n"
            f"Run fit_background.py --era {era_tag} first.")
    with open(json_path) as f:
        bkg_info = json.load(f)
    ratio_mc = bkg_info['ratio_peaking_mc']
    print(f"    n_bkg_mc = {bkg_info['n_bkg_mc']:.4f}  ratio_peaking_mc = {ratio_mc:.4f}", flush=True)

    # ── Load background component PDFs ──
    ws_path = config.bkg_fit_ws(era_tag)
    if not os.path.exists(ws_path):
        raise FileNotFoundError(
            f"Background fit workspace not found: {ws_path}\n"
            f"Run fit_background.py --era {era_tag} first.")

    bkg_file = ROOT.TFile(ws_path, 'READ')
    fit_ws   = bkg_file.Get(f"ws_bkg_{era_tag}")
    if not fit_ws:
        bkg_file.Close()
        raise RuntimeError(f"Workspace ws_bkg_{era_tag} not found in {ws_path}")

    bkg_ws = ROOT.RooWorkspace(f"ws_bkg_model_{era_tag}")
    for name in (f"bkg_peak_{era_tag}", f"bkg_nonpeak_{era_tag}"):
        pdf = fit_ws.pdf(name)
        if not pdf:
            bkg_file.Close()
            raise RuntimeError(f"PDF {name} not found in {ws_path}")
        # RecycleConflictNodes merges shared vars (mll, met) rather than duplicating them
        bkg_ws.Import(pdf, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    bkg_file.Close()

    # ── Ratio r of peaking to total background ──
    ratio_peaking = ROOT.RooRealVar(f"ratio_peaking_{era_tag}", f"ratio_peaking_{era_tag}",
                                    ratio_mc, 0.0, 1.0)
    # Left floating: the datacard constrains it (ratio_peaking_{era} param ...)

    # ── Total background shape ──
    pdf_name  = f"bkg_total_{era_tag}"
    bkg_total = ROOT.RooAddPdf(pdf_name, pdf_name,
                               ROOT.RooArgList(bkg_ws.pdf(f"bkg_peak_{era_tag}"),
                                               bkg_ws.pdf(f"bkg_nonpeak_{era_tag}")),
                               ROOT.RooArgList(ratio_peaking))
    bkg_ws.Import(bkg_total, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    print(f"    Added {ratio_peaking.GetName()} = {ratio_mc:.4f}", flush=True)

    # ── Build data_obs (pseudo-data -- see build_data_obs docstring). The
    # background normalization itself is the bkg rate in the datacard
    # (make_datacards.py), not a _norm var in the workspace. ──
    bkg_norm = sum(config.BKG_YIELD.values())
    print(f"    Building data_obs (pseudo-data from bkg_total)...", flush=True)
    ds_obs = build_data_obs(bkg_ws.pdf(pdf_name), bkg_ws.var("mll"), bkg_ws.var("met"), bkg_norm)
    bkg_ws.Import(ds_obs)
    print(f"    data_obs: {ds_obs.numEntries()} events", flush=True)
    return bkg_ws


# ── Per-mass-point workspace ──────────────────────────────────────────────────

def make_mass_point_workspace(mchi, mlsp, era_tag, bkg_ws, sig_yield, force=False):
    """
    Writes ws_{era} for one mass point: the background model from bkg_ws,
    the signal PDF from fit_signal.py's workspace, and the signal yield
    RooRealVar. Returns True if written (or already present), False if skipped.
    """
    out_path = config.combined_ws(mchi, mlsp, era_tag)
    if not force and os.path.exists(out_path):
        print(f"  EXISTS (skip): {os.path.basename(out_path)}")
        return True

    if sig_yield <= 0:
        print(f"  SKIP mchi={mchi} mlsp={mlsp}: signal yield zero or missing in signal_config.json")
        return False

    sig_ws_path = config.sig_fit_ws(mchi, mlsp, era_tag)
    if not os.path.exists(sig_ws_path):
        print(f"  SKIP mchi={mchi} mlsp={mlsp}: signal workspace missing: {sig_ws_path}\n"
              f"    Run fit_signal.py --era {era_tag} --force to generate it.")
        return False

    sig_file = ROOT.TFile(sig_ws_path, 'READ')
    sig_ws   = sig_file.Get(f"ws_sig_mchi{mchi}_mlsp{mlsp}")
    sig_pdf_name = f"sig_pdf_mchi{mchi}_mlsp{mlsp}_{era_tag}"
    sig_pdf  = sig_ws.pdf(sig_pdf_name) if sig_ws else None
    if not sig_pdf:
        sig_file.Close()
        raise RuntimeError(f"PDF {sig_pdf_name} not found in {sig_ws_path}")

    ws = ROOT.RooWorkspace(f"ws_{era_tag}", f"ws_{era_tag}")

    # Background: bkg_total (brings ratio_peaking with it)
    ws.Import(bkg_ws.pdf(f"bkg_total_{era_tag}"), ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())

    # Signal: pdf and its yield
    ws.Import(sig_pdf, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    sig_file.Close()
    sig_norm = ROOT.RooRealVar(f"{sig_pdf_name}_norm", f"{sig_pdf_name}_norm",
                               sig_yield, 0.0, 10.0 * sig_yield)
    sig_norm.setConstant(True)
    ws.Import(sig_norm, ROOT.RooFit.Silence())

    # Fix every m(ll)/MET shape parameter of the signal and background pdfs
    # (fit_signal.py leaves the signal m(ll) params floating). Only
    # ratio_peaking stays floating, constrained by the datacard.
    observables = ROOT.RooArgSet(ws.var("mll"), ws.var("met"))
    n_frozen = 0
    for pdf_name in (sig_pdf_name, f"bkg_total_{era_tag}"):
        for p in ws.pdf(pdf_name).getParameters(observables):
            if p.GetName() == f"ratio_peaking_{era_tag}":
                continue
            if not p.isConstant():
                n_frozen += 1
            p.setConstant(True)

    ws.Import(bkg_ws.data("data_obs"))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ws.writeToFile(out_path)
    print(f"  Wrote: {os.path.basename(out_path)}  [{sig_norm.GetName()} = {sig_yield:.4f}, "
          f"froze {n_frozen} floating shape params]", flush=True)
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build signal+background workspaces for Combine")
    parser.add_argument("--era",   choices=["run2", "run3", "both"], default="both")
    parser.add_argument("--mchi",  type=int, default=None)
    parser.add_argument("--mlsp",  type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing workspaces")
    args = parser.parse_args()

    eras = ["run2", "run3"] if args.era == "both" else [args.era]

    mass_points = sorted(config.SIGNAL_GRID)
    if args.mchi is not None:
        mass_points = [(mc, ml) for mc, ml in mass_points if mc == args.mchi]
    if args.mlsp is not None:
        mass_points = [(mc, ml) for mc, ml in mass_points if ml == args.mlsp]
    print(f"Mass points: {len(mass_points)}", flush=True)

    df = load_signal_config()

    for era in eras:
        print(f"\n=== Background model: era={era} ===")
        try:
            bkg_ws = build_bkg_model(era)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        print(f"\n=== Workspaces: era={era} ===")
        n_ok = n_skip = n_fail = 0
        for mchi, mlsp in mass_points:
            try:
                ok = make_mass_point_workspace(mchi, mlsp, era, bkg_ws,
                                               get_yield(df, mchi, mlsp), force=args.force)
                if ok:  n_ok   += 1
                else:   n_skip += 1
            except Exception as e:
                print(f"  ERROR mchi={mchi} mlsp={mlsp}: {e}")
                import traceback; traceback.print_exc()
                n_fail += 1
        print(f"\n  Done: {n_ok} written  {n_skip} skipped  {n_fail} failed")

    print("\nDone. Next step: make_datacards.py")


if __name__ == "__main__":
    main()
