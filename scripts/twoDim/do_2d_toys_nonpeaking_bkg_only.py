#!/usr/bin/env python3
"""
Not used anymore
do_2d_toys_nonpeaking_bkg_only.py
2D (m(ll), MET) analogue of oneDim_simplification/do_1d_met_toys.py: generates and fits
toy datasets from the full 2D signal product pdf and the non-peaking-in-m(ll)-only
background product pdf (no peaking component, no r), reusing toy_noPkgBkg.py's
get_signal_model/get_background_model -- unlike do_1d_met_toys.py, the pdfs here are
NOT reduced to their MET marginal, so RooMCStudy generates+fits over both observables
at once (RooArgSet(mll, met)), the same pattern already used and tested in
toy_noPkgBkg.py's own __main__ block.

Model: sig_pdf = sigtot_mll_met_2dpdf  (m(ll) DCB x MET spline, full 2D product)
       bkg_pdf = bkgnonpeak_mll_met_2dpdf  (m(ll) exponential x MET Gumbel, non-peaking only)
       total_pdf = n_sig * sig_pdf + n_bkg * bkg_pdf   (both yields floating)
"""

import ROOT
import os
import sys

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))

ONE_DIM_DIR = os.path.join(os.path.dirname(__file__), "..", "oneDim_simplification")
sys.path.insert(0, ONE_DIM_DIR)
from do_1d_met_toys import collect_results, _first_converged_toys  # noqa: F401 -- re-exported, dimension-agnostic

HBB_ZLL_TOY_DIR = ("/afs/cern.ch/work/s/skkwan/public/zhmet/CMSSW_14_0_21/src/"
                    "2DFit_higgsinos/hbb_zll/toy_experiments")
sys.path.insert(0, HBB_ZLL_TOY_DIR)
import toy_noPkgBkg as toy_ref


# ── Model construction ──────────────────────────────────────────────────────────────

def build_model_2d(m1, m2):
    """
    Build (mll, met, sig_pdf, bkg_pdf, components) for one mass point: the full 2D
    signal product pdf and the non-peaking-only 2D background product pdf, reusing
    toy_noPkgBkg.py's fitted shapes (same chdir/mll-global pattern as
    do_1d_met_toys.py's build_model_1d -- get_signal_model/get_background_model read
    paths relative to hbb_zll/toy_experiments/ and a bare module-level `mll` global).
    Caller must keep the returned vars/pdfs/components alive for as long as the model
    is in use.
    """
    prev_cwd = os.getcwd()
    try:
        os.chdir(HBB_ZLL_TOY_DIR)
        mll = ROOT.RooRealVar("m_ll", "m_ll", 60, 120)
        toy_ref.mll = mll
        sig_pdf, met, sig_components = toy_ref.get_signal_model(m1, m2)
        bkg_pdf, bkg_components = toy_ref.get_background_model(met)
    finally:
        os.chdir(prev_cwd)

    return mll, met, sig_pdf, bkg_pdf, sig_components + bkg_components


# ── Toy generation + fitting ─────────────────────────────────────────────────────────

def run_mcstudy_2d(mll, met, sig_pdf, bkg_pdf, n_sig_in, n_bkg_in, n_sig_range, n_bkg_range,
                   n_experiments, seed):
    n_sig = ROOT.RooRealVar("n_sig", "n_sig", n_sig_in, *n_sig_range)
    n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", n_bkg_in, *n_bkg_range)
    total_pdf = ROOT.RooAddPdf("total_pdf_2d_mll_met", "total_pdf_2d_mll_met",
                               ROOT.RooArgList(sig_pdf, bkg_pdf),
                               ROOT.RooArgList(n_sig, n_bkg))

    ROOT.RooRandom.randomGenerator().SetSeed(seed)
    mcs = ROOT.RooMCStudy(
        total_pdf,
        ROOT.RooArgSet(mll, met),
        ROOT.RooFit.Extended(),
        ROOT.RooFit.Silence(),
        ROOT.RooFit.FitOptions(ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1)),
    )
    print(f"Running {n_experiments} toy experiments "
         f"(n_sig_in={n_sig_in}, n_bkg_in={n_bkg_in}, seed={seed})...", flush=True)
    mcs.generateAndFit(n_experiments, 0, True)
    print("Done.", flush=True)
    return mcs, n_sig, n_bkg, total_pdf


# ── Example toy plots ────────────────────────────────────────────────────────────────
# total_pdf here is a 2D RooProdPdf combination (not a 1D marginal like
# do_1d_met_toys.py's), so a single obs.frame() can't show the actual fitted pdf the
# way do_1d_met_toys.make_example_toy_plot does. Instead, follow do_toys.py's proven
# pattern: build one RooPlot per observable (mll, met) -- plotOn a multi-dim pdf onto a
# single-observable frame analytically projects out the other observable, so each frame
# is still a faithful marginal of the actual 2D fit, not an approximation.

_OBS_CONFIGS = [
    ("mll", 40, 60.0, 120.0, "m(ll) [GeV]"),
    ("met", 60, 200.0, 1200.0, "MET [GeV]"),
]


def make_example_toy_plot(mcs, mll, met, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
                          plots_dir, tag, toy_index):
    """Plot one generated toy dataset with the total/signal/background curves
    overlaid at that toy's own best-fit yields (not the injected truth), once per
    observable (m(ll) and MET marginals of the same 2D fit)."""
    os.makedirs(plots_dir, exist_ok=True)

    data = mcs.genData(toy_index)
    fr = mcs.fitResult(toy_index)
    fitted_pars = fr.floatParsFinal()
    sig_par = fitted_pars.find("n_sig")
    bkg_par = fitted_pars.find("n_bkg")
    n_sig.setVal(sig_par.getVal())
    n_bkg.setVal(bkg_par.getVal())

    obs_by_name = {"mll": mll, "met": met}
    for obs_name, nbins, xmin, xmax, xlabel in _OBS_CONFIGS:
        obs = obs_by_name[obs_name]
        frame = obs.frame(ROOT.RooFit.Bins(nbins), ROOT.RooFit.Range(xmin, xmax),
                          ROOT.RooFit.Title(f"Toy #{toy_index}: data + best fit"))
        data.plotOn(frame, ROOT.RooFit.Name("data"))
        total_pdf.plotOn(frame, ROOT.RooFit.Name("total"), ROOT.RooFit.LineColor(ROOT.kBlue + 1))
        total_pdf.plotOn(frame, ROOT.RooFit.Name("bkg"),
                         ROOT.RooFit.Components(bkg_pdf.GetName()),
                         ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed + 1))
        total_pdf.plotOn(frame, ROOT.RooFit.Name("sig"),
                         ROOT.RooFit.Components(sig_pdf.GetName()),
                         ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen + 2))

        canv = ROOT.TCanvas(f"canv_example_toy_{obs_name}_{tag}", "", 700, 600)
        frame.SetXTitle(xlabel)
        frame.Draw()

        leg = ROOT.TLegend(0.60, 0.56, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.032)
        leg.AddEntry(frame.findObject("data"), "Toy data", "pe")
        leg.AddEntry(frame.findObject("total"), "S+B fit", "l")
        leg.AddEntry(frame.findObject("bkg"), "Background", "l")
        leg.AddEntry(frame.findObject("sig"), "Signal", "l")
        leg.AddEntry(ROOT.nullptr, f"n_{{sig}} = {sig_par.getVal():.2f} #pm {sig_par.getError():.2f}", "")
        leg.AddEntry(ROOT.nullptr, f"n_{{bkg}} = {bkg_par.getVal():.2f} #pm {bkg_par.getError():.2f}", "")
        leg.Draw()

        outpath = os.path.join(plots_dir, f"example_toy{toy_index}_{obs_name}_{tag}.pdf")
        canv.SaveAs(outpath)
        canv.SaveAs(outpath.replace(".pdf", ".png"))
        print(f"  Created {outpath} (+.png) "
             f"[n_sig_fit={n_sig.getVal():.3f}, n_bkg_fit={n_bkg.getVal():.3f}]")
