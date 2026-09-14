#!/usr/bin/env python3
"""
helper.py
Single source of truth for everything shared across the twoDim/ scripts: the 2D
(m(ll), MET) signal/background model getters, toy generation/result collection, and
CMS-styled plotting utilities. No sys.path manipulation -- the model-fetching
internals below are copied in from toy.py/toy_noPkgBkg.py (a different CMSSW area,
zhmet/.../hbb_zll/toy_experiments/) rather than imported, since those relative-path
ROOT.TFile.Open() calls only work with the cwd temporarily chdir'd into that area
anyway (see build_model_2d/build_model_2d_both_bkg).
"""

import ROOT
import os
import cmsstyle as CMS

HBB_ZLL_TOY_DIR = ("/afs/cern.ch/work/s/skkwan/public/zhmet/CMSSW_14_0_21/src/"
                    "2DFit_higgsinos/hbb_zll/toy_experiments")

# Set by build_model_2d/build_model_2d_both_bkg before calling the _get_*_model_2d
# functions below, which reference it as a bare module-level name -- mirrors the
# original toy_ref.mll = mll pattern without needing a separate imported module.
mll = None


# ── Model-fetching internals (copied from toy.py / toy_noPkgBkg.py) ─────────────────

def _get_signal_model_2d(m1=650, m2=1):
    """
    Get the signal model for a given mass point.
    Returns (pdf, ws_met, components) where ws_met is the workspace's MET observable --
    the caller must use ws_met everywhere so the spline PDF is properly connected.
    """
    signalresultsfile = ROOT.TFile.Open(f"../individual_pdf_fits/individual_fit_results/fitresult_signal_{m1}_{m2}.root", "READ")

    workspace  = signalresultsfile.Get(f"workspace_{m1}_{m2}")
    sig_result = signalresultsfile.Get("sig_result")

    # Use the workspace's own met variable; creating a new RooRealVar("met") would be a
    # different object, leaving the spline disconnected and returning a constant in fits.
    ws_met = workspace.var("met")
    ws_met.setRange(200, 1200)

    # sigtot_mll_met_2dpdf = sig_dcb_mll (mll) x pdf_of_spline (MET)
    mean_mll   = sig_result.floatParsFinal().find(f"mean_mll_{m1}_{m2}")
    sigmal_mll = sig_result.floatParsFinal().find(f"sigmal_mll_{m1}_{m2}")
    sigmar_mll = sig_result.floatParsFinal().find(f"sigmar_mll_{m1}_{m2}")
    alphal_mll = sig_result.floatParsFinal().find(f"alphal_mll_{m1}_{m2}")
    alphar_mll = sig_result.floatParsFinal().find(f"alphar_mll_{m1}_{m2}")
    nl_mll     = sig_result.floatParsFinal().find(f"nl_mll_{m1}_{m2}")
    nr_mll     = sig_result.floatParsFinal().find(f"nr_mll_{m1}_{m2}")
    for v in [mean_mll, sigmal_mll, sigmar_mll, alphal_mll, alphar_mll, nl_mll, nr_mll]:
        v.setConstant(True)

    sig_dcb_mll = ROOT.RooCrystalBall("sig_dcb_mll", "sig_dcb_mll",
                                       mll, mean_mll, sigmal_mll, sigmar_mll,
                                       alphal_mll, nl_mll, alphar_mll, nr_mll)

    pdf_of_spline = workspace.pdf(f"pdf_of_spline_{m1}_{m2}")

    sigtot_mll_met_2dpdf = ROOT.RooProdPdf("sigtot_mll_met_2dpdf", "sigtot_mll_met_2dpdf",
                                            ROOT.RooArgList(sig_dcb_mll, pdf_of_spline))

    components = [
        signalresultsfile,
        workspace,
        sig_result, mean_mll, sigmal_mll, sigmar_mll, alphal_mll, alphar_mll, nl_mll, nr_mll,
        sig_dcb_mll, pdf_of_spline,
    ]

    return sigtot_mll_met_2dpdf, ws_met, components


def _get_background_model_nonpeaking_2d(met):
    """Get the non-peaking background model only (no peaking component, no ratio)."""
    bkgresultsfile = ROOT.TFile.Open("../individual_pdf_fits/individual_fit_results/fitresult_background_all_except_ZPeak.root", "READ")

    bkg_nonpeak_result_met = bkgresultsfile.Get("bkg_nonpeak_result_met")
    bkg_nonpeak_result_mll = bkgresultsfile.Get("bkg_nonpeak_result_mll")

    mu_nonpeak_met = bkg_nonpeak_result_met.floatParsFinal().find("mu_nonpeak_met")
    b_nonpeak_met  = bkg_nonpeak_result_met.floatParsFinal().find("b_nonpeak_met")
    a_nonpeak_mll  = bkg_nonpeak_result_mll.floatParsFinal().find("a_nonpeak_mll")
    for v in [mu_nonpeak_met, b_nonpeak_met, a_nonpeak_mll]:
        v.setConstant(True)

    bkgnonpeak_met = ROOT.RooGenericPdf(
        "bkgnonpeak_met", "bkgnonpeak_met",
        "1/b_nonpeak_met * exp(-(@0 - mu_nonpeak_met)/b_nonpeak_met"
        " - exp(-(@0 - mu_nonpeak_met)/b_nonpeak_met))",
        ROOT.RooArgList(met, mu_nonpeak_met, b_nonpeak_met))

    bkgnonpeak_mll = ROOT.RooExponential("bkgnonpeak_mll", "bkgnonpeak_mll",
                                          mll, a_nonpeak_mll)

    bkgnonpeak_mll_met_2dpdf = ROOT.RooProdPdf("bkgnonpeak_mll_met_2dpdf",
                                                "bkgnonpeak_mll_met_2dpdf",
                                                ROOT.RooArgList(bkgnonpeak_mll, bkgnonpeak_met))

    components = [
        bkgresultsfile,
        bkg_nonpeak_result_met, bkg_nonpeak_result_mll,
        mu_nonpeak_met, b_nonpeak_met, a_nonpeak_mll,
        bkgnonpeak_met, bkgnonpeak_mll, bkgnonpeak_mll_met_2dpdf,
    ]
    return bkgnonpeak_mll_met_2dpdf, components


def _get_background_model_both_2d(met, fix_r=False):
    """Get the total background model (peaking + non-peaking in m(ll)), leaving the
    ratio r (peaking/total) floating unless fix_r."""
    bkgresultsfile   = ROOT.TFile.Open("../individual_pdf_fits/individual_fit_results/fitresult_background_all_except_ZPeak.root", "READ")
    zpeakresultsfile = ROOT.TFile.Open("../zpeak_fit/initial_zPeak_fit_result.root", "READ")

    bkg_nonpeak_result_met = bkgresultsfile.Get("bkg_nonpeak_result_met")
    bkg_nonpeak_result_mll = bkgresultsfile.Get("bkg_nonpeak_result_mll")
    bkg_peaking_result_met = bkgresultsfile.Get("bkg_peaking_result_met")
    zpeak_result           = zpeakresultsfile.Get("zPeak_CRZ_fit_result")

    # Non-peaking in m(ll)
    mu_nonpeak_met = bkg_nonpeak_result_met.floatParsFinal().find("mu_nonpeak_met")
    b_nonpeak_met  = bkg_nonpeak_result_met.floatParsFinal().find("b_nonpeak_met")
    a_nonpeak_mll  = bkg_nonpeak_result_mll.floatParsFinal().find("a_nonpeak_mll")
    for v in [mu_nonpeak_met, b_nonpeak_met, a_nonpeak_mll]:
        v.setConstant(True)

    bkgnonpeak_met = ROOT.RooGenericPdf(
        "bkgnonpeak_met", "bkgnonpeak_met",
        "1/b_nonpeak_met * exp(-(@0 - mu_nonpeak_met)/b_nonpeak_met"
        " - exp(-(@0 - mu_nonpeak_met)/b_nonpeak_met))",
        ROOT.RooArgList(met, mu_nonpeak_met, b_nonpeak_met))

    bkgnonpeak_mll = ROOT.RooExponential("bkgnonpeak_mll", "bkgnonpeak_mll",
                                          mll, a_nonpeak_mll)

    bkgnonpeak_mll_met_2dpdf = ROOT.RooProdPdf("bkgnonpeak_mll_met_2dpdf",
                                                "bkgnonpeak_mll_met_2dpdf",
                                                ROOT.RooArgList(bkgnonpeak_mll, bkgnonpeak_met))

    # Leave the ratio_peaking floating unless fix_r is true
    ratio_peaking = ROOT.RooRealVar("ratio_peaking", "ratio_peaking", 0.088, 0, 1)
    if fix_r:
        ratio_peaking.setConstant(True)

    # Peaking in m(ll)
    mu_peaking_met = bkg_peaking_result_met.floatParsFinal().find("mu_peaking_met")
    b_peaking_met  = bkg_peaking_result_met.floatParsFinal().find("b_peaking_met")
    for v in [mu_peaking_met, b_peaking_met]:
        v.setConstant(True)

    bkgpeaking_met = ROOT.RooGenericPdf(
        "bkgpeaking_met", "bkgpeaking_met",
        "1/b_peaking_met * exp(-(@0 - mu_peaking_met)/b_peaking_met"
        " - exp(-(@0 - mu_peaking_met)/b_peaking_met))",
        ROOT.RooArgList(met, mu_peaking_met, b_peaking_met))

    zpeak_mean_mll   = zpeak_result.floatParsFinal().find("peak_mean_mll")
    zpeak_sigmal_mll = zpeak_result.floatParsFinal().find("peak_sigmal_mll")
    zpeak_sigmar_mll = zpeak_result.floatParsFinal().find("peak_sigmar_mll")
    zpeak_alphal_mll = zpeak_result.floatParsFinal().find("peak_alphal_mll")
    zpeak_nl_mll     = zpeak_result.floatParsFinal().find("peak_nl_mll")
    zpeak_alphar_mll = zpeak_result.floatParsFinal().find("peak_alphar_mll")
    zpeak_nr_mll     = zpeak_result.floatParsFinal().find("peak_nr_mll")
    for v in [zpeak_mean_mll, zpeak_sigmal_mll, zpeak_sigmar_mll,
              zpeak_alphal_mll, zpeak_nl_mll, zpeak_alphar_mll, zpeak_nr_mll]:
        v.setConstant(True)

    bkgpeaking_mll = ROOT.RooCrystalBall("bkgpeaking_mll", "bkgpeaking_mll",
                                          mll, zpeak_mean_mll, zpeak_sigmal_mll, zpeak_sigmar_mll,
                                          zpeak_alphal_mll, zpeak_nl_mll, zpeak_alphar_mll, zpeak_nr_mll)

    bkgpeaking_mll_met_2dpdf = ROOT.RooProdPdf("bkgpeaking_mll_met_2dpdf",
                                                "bkgpeaking_mll_met_2dpdf",
                                                ROOT.RooArgList(bkgpeaking_mll, bkgpeaking_met))

    bkgtot_mll_met_2dpdf = ROOT.RooAddPdf("bkgtot_mll_met_2dpdf", "bkgtot_mll_met_2dpdf",
                                           ROOT.RooArgList(bkgpeaking_mll_met_2dpdf,
                                                           bkgnonpeak_mll_met_2dpdf),
                                           ratio_peaking)

    components = [
        bkgresultsfile, zpeakresultsfile,
        bkg_nonpeak_result_met, bkg_nonpeak_result_mll,
        bkg_peaking_result_met, zpeak_result,
        bkgnonpeak_met, bkgnonpeak_mll, bkgnonpeak_mll_met_2dpdf,
        bkgpeaking_met, bkgpeaking_mll, bkgpeaking_mll_met_2dpdf,
        ratio_peaking,
    ]
    return bkgtot_mll_met_2dpdf, components, ratio_peaking


# ── 2D model getters ─────────────────────────────────────────────────────────────────

def build_model_2d(m1, m2):
    """
    Build (mll, met, sig_pdf, bkg_pdf, components) for one mass point: the full 2D
    signal product pdf and the non-peaking-only 2D background product pdf.
    Caller must keep the returned vars/pdfs/components alive for as long as the model
    is in use.
    """
    global mll
    prev_cwd = os.getcwd()
    try:
        os.chdir(HBB_ZLL_TOY_DIR)
        mll = ROOT.RooRealVar("m_ll", "m_ll", 60, 120)
        sig_pdf, met, sig_components = _get_signal_model_2d(m1, m2)
        bkg_pdf, bkg_components = _get_background_model_nonpeaking_2d(met)
    finally:
        os.chdir(prev_cwd)

    return mll, met, sig_pdf, bkg_pdf, sig_components + bkg_components


def build_model_2d_both_bkg(m1, m2, r, floatR, r_range=(0.0, 1.0)):
    """
    Build (mll, met, sig_pdf, bkg_pdf, ratio_peaking, components) for one mass point:
    the full 2D signal product pdf and the r-weighted mix of the peaking and
    non-peaking 2D background product pdfs. ratio_peaking is generated at r and left
    floating in the fit iff floatR (fit range r_range); otherwise held constant at r.
    Caller must keep the returned vars/pdfs/components alive for as long as the model
    is in use.
    """
    global mll
    prev_cwd = os.getcwd()
    try:
        os.chdir(HBB_ZLL_TOY_DIR)
        mll = ROOT.RooRealVar("m_ll", "m_ll", 60, 120)
        sig_pdf, met, sig_components = _get_signal_model_2d(m1, m2)
        bkg_pdf, bkg_components, ratio_peaking = _get_background_model_both_2d(met, fix_r=not floatR)
    finally:
        os.chdir(prev_cwd)

    ratio_peaking.setRange(*r_range)
    ratio_peaking.setVal(r)
    ratio_peaking.setConstant(not floatR)

    return mll, met, sig_pdf, bkg_pdf, ratio_peaking, sig_components + bkg_components


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


# ── Fit-result collection ────────────────────────────────────────────────────────────

def collect_results(mcs, n_experiments, n_sig_in, n_bkg_in):
    """Pull (value, error, status) for n_sig/n_bkg out of each toy's fit result."""
    rows = []
    for i in range(n_experiments):
        fr = mcs.fitResult(i)
        sig_par = fr.floatParsFinal().find("n_sig")
        bkg_par = fr.floatParsFinal().find("n_bkg")
        rows.append({
            'status':     fr.status(),
            'cov_qual':   fr.covQual(),
            'n_sig_val':  sig_par.getVal(),
            'n_sig_err':  sig_par.getError(),
            'n_bkg_val':  bkg_par.getVal(),
            'n_bkg_err':  bkg_par.getError(),
        })
    return rows


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


def _first_converged_toys(rows, n):
    return [i for i, r in enumerate(rows) if r['status'] == 0][:n]


def add_pulls(rows, n_sig_in, n_bkg_in):
    for r in rows:
        r["n_sig_pull"] = (r["n_sig_val"] - n_sig_in) / r["n_sig_err"] if r["n_sig_err"] > 0 else None
        r["n_bkg_pull"] = (r["n_bkg_val"] - n_bkg_in) / r["n_bkg_err"] if r["n_bkg_err"] > 0 else None
    return rows


def add_pulls_2d(rows, n_sig_in, n_bkg_in, r_in, floatR):
    for row in rows:
        row["n_sig_pull"] = (row["n_sig_val"] - n_sig_in) / row["n_sig_err"] if row["n_sig_err"] > 0 else None
        row["n_bkg_pull"] = (row["n_bkg_val"] - n_bkg_in) / row["n_bkg_err"] if row["n_bkg_err"] > 0 else None
        if floatR:
            row["r_pull"] = (row["r_val"] - r_in) / row["r_err"] if row["r_err"] > 0 else None
    return rows


def aggregate_point(point, val_key):
    import statistics as stats

    toys = [t for t in point["toys"] if t["status"] == 0]

    vals = [t[val_key] for t in toys]
    mean_val = stats.mean(vals)
    spread = stats.pstdev(vals) if len(vals) > 1 else 0.0
    print(f">>> {mean_val}, {spread}")
    return mean_val, spread


# ── Example toy plots ────────────────────────────────────────────────────────────────
# total_pdf here is a 2D RooProdPdf combination (not a 1D marginal), so a single
# obs.frame() can't show the actual fitted pdf directly -- plotOn a multi-dim pdf onto
# a single-observable frame analytically projects out the other observable, so each
# frame is still a faithful marginal of the actual 2D fit.

_OBS_CONFIGS = [
    ("mll", 40, 60.0, 120.0, "m(ll) [GeV]"),
    ("met", 60, 200.0, 1200.0, "MET [GeV]"),
]


def make_example_toy_plot_nonpeaking_bkg(mcs, mll, met, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
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


def make_example_toy_plot_both_bkg(mcs, mll, met, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
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
    # m(ll) 2D component pdfs -- pick their names out of its pdfList() so they can be
    # plotted as separate curves.
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


# ── Plotting (CMS-styled, cmsstyle "Private work") ───────────────────────────────────

def _finalize_canvas(canv, plotname, plot_dir):
    """Shared eps->pdf conversion (keeps Greek symbols intact, unlike a direct .pdf
    SaveAs) + .png + move-to-plot_dir tail, used by every plot in this module."""
    canv.SaveAs(f"{plotname}.eps")
    os.system(f"gs -q -dBATCH -dNOPAUSE -dSAFER -dEPSCrop -dPDFSETTINGS=/prepress -sDEVICE=pdfwrite "
              f"-dEmbedAllFonts=true -dSubsetFonts=true -sOutputFile={plotname}.pdf {plotname}.eps && rm {plotname}.eps")

    canv.SaveAs(f"{plotname}.png")
    print(f"Created {plotname}.pdf / .png")
    os.makedirs(plot_dir, exist_ok=True)
    os.system(f"mv {plotname}.pdf {plotname}.png {plot_dir}/")


def make_scatter_plot(x_vals, y_vals, y_errs, xlabel, ylabel, refline, plotname,
                       plot_dir, mass_point, n_bkg_in, n_experiments, refline_label=None,
                       label_points=False):
    m1, m2 = mass_point
    n = len(x_vals)
    graph = ROOT.TGraphErrors(n)
    for i, (x, y, ey) in enumerate(zip(x_vals, y_vals, y_errs)):
        graph.SetPoint(i, x, y)
        graph.SetPointError(i, 0.0, ey)
    graph.SetMarkerStyle(ROOT.kFullCircle)
    graph.SetMarkerColor(ROOT.kRed + 1)
    graph.SetLineColor(ROOT.kRed + 1)
    graph.SetMarkerSize(1.1)

    CMS.SetExtraText("Private work")
    CMS.SetCmsText("CMS", font=62, size=0.76)
    CMS.SetLumi(250, unit="fb", run="2018")

    x_min = min(0.0, min(x_vals)) - 0.1 * (max(x_vals) - min(x_vals) + 1)
    x_max = max(x_vals) * 1.15 + 1
    y_min = min(0.0, min(y - ey for y, ey in zip(y_vals, y_errs))) - 2
    y_max = max(y + ey for y, ey in zip(y_vals, y_errs)) * 1.25 + 2

    if "nsig" in plotname:
        y_max *= 1.1

    canv = CMS.cmsCanvas(f"canv_{plotname}",
                          x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                          nameXaxis=xlabel, nameYaxis=ylabel,
                          square=False, extraSpace=0.02, iPos=0.)
    canv.SetRightMargin(0.05)
    CMS.UpdatePad(canv)

    line = ROOT.TLine(x_min, refline(x_min), x_max, refline(x_max))
    line.SetLineStyle(ROOT.kDashed)
    line.SetLineColor(ROOT.kBlack)
    line.Draw("SAME")

    CMS.cmsObjectDraw(graph, "P")

    leg = ROOT.TLegend(0.15, 0.78, 0.55, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.032)
    leg.AddEntry(ROOT.nullptr, f"Signal ({m1}, {m2}) GeV", "")
    leg.AddEntry(ROOT.nullptr, f"n_{{bkg}}^{{in}} = {n_bkg_in:g}, N_{{toys}} = {n_experiments}", "")
    if refline_label is not None:
        leg.AddEntry(line, refline_label, "l")
    CMS.cmsObjectDraw(leg)

    if label_points:
        # Keep the TLatex objects alive past this function's return (PyROOT drops
        # unreferenced objects), by hanging them off the canvas.
        canv.point_labels = []
        y_offset = 0.03 * (y_max - y_min)
        for x, y, ey in zip(x_vals, y_vals, y_errs):
            latex = ROOT.TLatex(x, y + ey + y_offset, f"{y:.2f} #pm {ey:.2f}")
            latex.SetTextFont(42)
            latex.SetTextSize(0.032)
            latex.SetTextAlign(21)
            latex.Draw("SAME")
            canv.point_labels.append(latex)

    CMS.UpdatePad(canv)

    _finalize_canvas(canv, plotname, plot_dir)


GAUSSIAN_FIT_COLOR = "#5790fc"


def make_distribution_plot(vals, xlabel, plotname, plot_dir, mass_point, n_sig_in,
                            n_bkg_in, n_experiments, x_range=None, fit_gaussian=False,
                            nbins=40, refline_val=None, fit_color=GAUSSIAN_FIT_COLOR):
    """CMS-styled plot (black points + error bars) of per-toy values (best-fit yield
    or pull) for one injected point. x_range fixes the axis to that (lo, hi) instead
    of the data-driven padded range; fit_gaussian overlays a Gaussian fit and reports
    its mean/sigma/chi2/ndf. refline_val, if given, draws a thin black dashed vertical
    line at that x position (e.g. the injected yield). Values outside [lo, hi] (e.g. a
    fit range wider than the plotted range) are clamped into the first/last bin, so
    that bin doubles as a visible underflow/overflow bin instead of silently vanishing
    into TH1's internal (unplotted) under/overflow bins.

    Returns a dict of the Gaussian fit results (mu, mu_err, sigma, sigma_err, chi2,
    ndf) if fit_gaussian is True, else None -- callers use the sigma as the toy-spread
    error bar on the corresponding scatter-plot point."""
    m1, m2 = mass_point
    if x_range is not None:
        lo, hi = x_range
    else:
        lo, hi = min(vals), max(vals)
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - pad, hi + pad

    hist = ROOT.TH1D(f"h_{plotname}", "", nbins, lo, hi)
    bin_width = (hi - lo) / nbins
    eps = bin_width * 1e-6
    for v in vals:
        hist.Fill(min(max(v, lo + eps), hi - eps))
    hist.SetStats(False)
    hist.SetMarkerStyle(ROOT.kFullCircle)
    hist.SetMarkerColor(ROOT.kBlack)
    hist.SetLineColor(ROOT.kBlack)

    fit = None
    fit_result = None
    if fit_gaussian:
        fit = ROOT.TF1(f"fit_{plotname}", "gaus", lo, hi)
        hist.Fit(fit, "QR")
        fit_result = {
            "mu": fit.GetParameter(1), "mu_err": fit.GetParError(1),
            "sigma": fit.GetParameter(2), "sigma_err": fit.GetParError(2),
            "chi2": fit.GetChisquare(), "ndf": fit.GetNDF(),
        }

    CMS.SetExtraText("Private work")
    CMS.SetCmsText("CMS", font=62, size=0.76)
    CMS.SetLumi(250, unit="fb", run="2018")

    peak = max(hist.GetMaximum(), fit.GetMaximum(lo, hi)) if fit is not None else hist.GetMaximum()
    y_max = peak * (2.0 if fit_gaussian else 1.35)

    canv = CMS.cmsCanvas(f"canv_{plotname}",
                          x_min=lo, x_max=hi, y_min=0.0, y_max=y_max,
                          nameXaxis=xlabel, nameYaxis="Toys",
                          square=False, extraSpace=0.02, iPos=0.)
    canv.SetRightMargin(0.05)
    CMS.UpdatePad(canv)

    CMS.cmsObjectDraw(hist, "PE")
    if fit is not None:
        fit.SetLineColor(ROOT.TColor.GetColor(fit_color))
        fit.SetLineWidth(2)
        fit.Draw("SAME")

    refline = None
    if refline_val is not None:
        refline = ROOT.TLine(refline_val, 0.0, refline_val, y_max)
        refline.SetLineStyle(ROOT.kDashed)
        refline.SetLineColor(ROOT.kBlack)
        refline.SetLineWidth(1)
        refline.Draw("SAME")

    leg = ROOT.TLegend(0.58, 0.76, 0.93, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.032)
    leg.AddEntry(ROOT.nullptr, f"Signal ({m1}, {m2}) GeV", "")
    leg.AddEntry(ROOT.nullptr, f"n_{{sig}}^{{in}} = {n_sig_in:g}, n_{{bkg}}^{{in}} = {n_bkg_in:g}", "")
    leg.AddEntry(ROOT.nullptr, f"N_{{toys}} = {n_experiments}", "")
    CMS.cmsObjectDraw(leg)

    if fit_result is not None:
        mu, mu_err = fit_result["mu"], fit_result["mu_err"]
        sigma, sigma_err = fit_result["sigma"], fit_result["sigma_err"]
        chi2, ndf = fit_result["chi2"], fit_result["ndf"]
        chi2_ndf = chi2 / ndf if ndf > 0 else float("nan")
        box = ROOT.TPaveText(0.58, 0.51, 0.93, 0.74, "NDC")
        box.SetBorderSize(0)
        box.SetFillStyle(0)
        box.SetTextSize(0.032)
        box.SetTextAlign(12)
        box.AddText("Fit parameters:")
        box.AddText(f"#mu: {mu:.3g} #pm {mu_err:.2g}")
        box.AddText(f"#sigma: {sigma:.3g} #pm {sigma_err:.2g}")
        box.AddText(f"#chi^{{2}} / ndf = {chi2:.2f} / {ndf} = {chi2_ndf:.2f}")
        box.Draw()

    CMS.UpdatePad(canv)

    _finalize_canvas(canv, plotname, plot_dir)

    return fit_result
