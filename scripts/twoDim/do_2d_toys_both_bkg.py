#!/usr/bin/env python3
"""
Not used anymore
do_2d_toys_both_bkg.py
2D (m(ll), MET) analogue of do_2d_toys_nonpeaking_bkg_only.py, but using BOTH
background components (peaking-in-m(ll) + non-peaking-in-m(ll)) instead of the
non-peaking-only shape, reusing toy.py's get_signal_model/get_background_model
(unlike do_2d_toys_nonpeaking_bkg_only.py, which uses toy_noPkgBkg.py):

    bkg_pdf = r * bkgpeaking_mll_met_2dpdf + (1 - r) * bkgnonpeak_mll_met_2dpdf

r (ratio_peaking) can be fixed or left floating in the fit -- build_model_2d_both_bkg's
floatR argument controls this (toy.py's get_background_model(fix_r=...) already does
the right thing internally). Toy generation always uses whatever value r is set to at
RooMCStudy construction time, regardless of its floating/constant flag.

Model: sig_pdf = sigtot_mll_met_2dpdf  (m(ll) DCB x MET spline, full 2D product)
       bkg_pdf = bkgtot_mll_met_2dpdf  (r-weighted mix of peaking + non-peaking 2D pdfs)
       total_pdf = n_sig * sig_pdf + n_bkg * bkg_pdf
"""

import ROOT
import os
import sys

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))
from do_2d_toys_nonpeaking_bkg_only import run_mcstudy_2d  # dimension-specific, r-agnostic (reused as-is)

ONE_DIM_DIR = os.path.join(os.path.dirname(__file__), "..", "oneDim_simplification")
sys.path.insert(0, ONE_DIM_DIR)
from do_1d_met_toys import _first_converged_toys  # noqa: F401 -- re-exported, dimension-agnostic

HBB_ZLL_TOY_DIR = ("/afs/cern.ch/work/s/skkwan/public/zhmet/CMSSW_14_0_21/src/2DFit_higgsinos/hbb_zll/toy_experiments")
sys.path.insert(0, HBB_ZLL_TOY_DIR)
import toy as toy_ref  # full model (peaking + non-peaking bkg), unlike toy_noPkgBkg


# ── Model construction ──────────────────────────────────────────────────────────────

def build_model_2d_both_bkg(m1, m2, r, floatR, r_range=(0.0, 1.0)):
    """
    Build (mll, met, sig_pdf, bkg_pdf, ratio_peaking, components) for one mass point:
    the full 2D signal product pdf and the r-weighted mix of the peaking and
    non-peaking 2D background product pdfs, reusing toy.py's fitted shapes (same
    chdir/mll-global pattern as do_2d_toys_nonpeaking_bkg_only.py's build_model_2d).
    ratio_peaking is generated at r and left floating in the fit iff floatR (fit range
    r_range); otherwise held constant at r. Caller must keep the returned
    vars/pdfs/components alive for as long as the model is in use.
    """
    prev_cwd = os.getcwd()
    try:
        os.chdir(HBB_ZLL_TOY_DIR)
        mll = ROOT.RooRealVar("m_ll", "m_ll", 60, 120)
        toy_ref.mll = mll
        sig_pdf, met, sig_components = toy_ref.get_signal_model(m1, m2)
        bkg_pdf, bkg_components, ratio_peaking = toy_ref.get_background_model(met, fix_r=not floatR)
    finally:
        os.chdir(prev_cwd)

    ratio_peaking.setRange(*r_range)
    ratio_peaking.setVal(r)
    ratio_peaking.setConstant(not floatR)

    return mll, met, sig_pdf, bkg_pdf, ratio_peaking, sig_components + bkg_components


# ── Fit-result collection (n_sig, n_bkg, plus r iff floatR) ─────────────────────────

def collect_results_2d(mcs, n_experiments, floatR):
    rows = []
    for i in range(n_experiments):
        fr = mcs.fitResult(i)
        pars = fr.floatParsFinal()
        sig_par = pars.find("n_sig")
        bkg_par = pars.find("n_bkg")
        row = {
            'status':     fr.status(),
            'cov_qual':   fr.covQual(),
            'n_sig_val':  sig_par.getVal(),
            'n_sig_err':  sig_par.getError(),
            'n_bkg_val':  bkg_par.getVal(),
            'n_bkg_err':  bkg_par.getError(),
        }
        if floatR:
            r_par = pars.find("ratio_peaking")
            row['r_val'] = r_par.getVal()
            row['r_err'] = r_par.getError()
        rows.append(row)
    return rows


def add_pulls_2d(rows, n_sig_in, n_bkg_in, r_in, floatR):
    for row in rows:
        row["n_sig_pull"] = (row["n_sig_val"] - n_sig_in) / row["n_sig_err"] if row["n_sig_err"] > 0 else None
        row["n_bkg_pull"] = (row["n_bkg_val"] - n_bkg_in) / row["n_bkg_err"] if row["n_bkg_err"] > 0 else None
        if floatR:
            row["r_pull"] = (row["r_val"] - r_in) / row["r_err"] if row["r_err"] > 0 else None
    return rows


# ── Example toy plots ────────────────────────────────────────────────────────────────
# total_pdf here is a 2D RooProdPdf combination (not a 1D marginal), so a single
# obs.frame() can't show the actual fitted pdf directly -- follow
# do_2d_toys_nonpeaking_bkg_only.make_example_toy_plot's pattern: plotOn a multi-dim
# pdf onto a single-observable frame analytically projects out the other observable,
# so each frame is still a faithful marginal of the actual 2D fit.

_OBS_CONFIGS = [
    ("mll", 40, 60.0, 120.0, "m(ll) [GeV]"),
    ("met", 60, 200.0, 1200.0, "MET [GeV]"),
]


def make_example_toy_plot(mcs, mll, met, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
                          ratio_peaking, floatR, plots_dir, tag, toy_index):
    """Plot one generated toy dataset with the total/signal/background curves
    overlaid at that toy's own best-fit yields (not the injected truth), once per
    observable (m(ll) and MET marginals of the same 2D fit). The legend also reports
    r -- its best fit if floatR, otherwise the fixed injected value."""
    os.makedirs(plots_dir, exist_ok=True)

    data = mcs.genData(toy_index)
    fr = mcs.fitResult(toy_index)
    fitted_pars = fr.floatParsFinal()
    sig_par = fitted_pars.find("n_sig")
    bkg_par = fitted_pars.find("n_bkg")
    n_sig.setVal(sig_par.getVal())
    n_bkg.setVal(bkg_par.getVal())

    if floatR:
        r_par = fitted_pars.find("ratio_peaking")
        ratio_peaking.setVal(r_par.getVal())
        r_val, r_err = r_par.getVal(), r_par.getError()
    else:
        r_val, r_err = ratio_peaking.getVal(), None

    # bkg_pdf is the r-weighted RooAddPdf of the peaking-in-m(ll) and non-peaking-in-
    # m(ll) 2D component pdfs (built by toy.py's get_background_model) -- pick their
    # names out of its pdfList() so they can be plotted as separate curves.
    peaking_name, nonpeaking_name = None, None
    for comp in bkg_pdf.pdfList():
        name = comp.GetName()
        if "nonpeak" in name:
            nonpeaking_name = name
        elif "peaking" in name:
            peaking_name = name

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
        total_pdf.plotOn(frame, ROOT.RooFit.Name("bkg_peaking"),
                         ROOT.RooFit.Components(peaking_name),
                         ROOT.RooFit.LineStyle(ROOT.kDotted), ROOT.RooFit.LineColor(ROOT.kRed + 1))
        total_pdf.plotOn(frame, ROOT.RooFit.Name("bkg_nonpeaking"),
                         ROOT.RooFit.Components(nonpeaking_name),
                         ROOT.RooFit.LineStyle(ROOT.kDashDotted), ROOT.RooFit.LineColor(ROOT.kOrange + 1))
        total_pdf.plotOn(frame, ROOT.RooFit.Name("sig"),
                         ROOT.RooFit.Components(sig_pdf.GetName()),
                         ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen + 2))

        canv = ROOT.TCanvas(f"canv_example_toy_{obs_name}_{tag}", "", 700, 600)
        frame.SetXTitle(xlabel)
        frame.Draw()

        leg = ROOT.TLegend(0.60, 0.46, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.032)
        leg.AddEntry(frame.findObject("data"), "Toy data", "pe")
        leg.AddEntry(frame.findObject("total"), "S+B fit", "l")
        leg.AddEntry(frame.findObject("bkg"), "Total background", "l")
        leg.AddEntry(frame.findObject("bkg_peaking"), "Peaking background", "l")
        leg.AddEntry(frame.findObject("bkg_nonpeaking"), "Non-peaking background", "l")
        leg.AddEntry(frame.findObject("sig"), "Signal", "l")
        leg.AddEntry(ROOT.nullptr, f"n_{{sig}} = {sig_par.getVal():.2f} #pm {sig_par.getError():.2f}", "")
        leg.AddEntry(ROOT.nullptr, f"n_{{bkg}} = {bkg_par.getVal():.2f} #pm {bkg_par.getError():.2f}", "")
        if r_err is not None:
            leg.AddEntry(ROOT.nullptr, f"r = {r_val:.3f} #pm {r_err:.3f}", "")
        else:
            leg.AddEntry(ROOT.nullptr, f"r = {r_val:.3f} (fixed)", "")
        leg.Draw()

        outpath = os.path.join(plots_dir, f"example_toy{toy_index}_{obs_name}_{tag}.pdf")
        canv.SaveAs(outpath)
        canv.SaveAs(outpath.replace(".pdf", ".png"))
        print(f"  Created {outpath} (+.png) "
             f"[n_sig_fit={n_sig.getVal():.3f}, n_bkg_fit={n_bkg.getVal():.3f}, r={r_val:.3f}]")
