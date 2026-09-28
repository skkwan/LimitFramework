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
import json
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


def _get_background_model_both_2d(met):
    """Get the peaking-in-m(ll) and non-peaking-in-m(ll) 2D background product pdfs
    individually -- each is separately yield-parameterized (n_bkg_peak, n_bkg_nonpeak)
    by the caller's fit model rather than combined here via a shape-mixture ratio."""
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

    components = [
        bkgresultsfile, zpeakresultsfile,
        bkg_nonpeak_result_met, bkg_nonpeak_result_mll,
        bkg_peaking_result_met, zpeak_result,
        bkgnonpeak_met, bkgnonpeak_mll, bkgnonpeak_mll_met_2dpdf,
        bkgpeaking_met, bkgpeaking_mll, bkgpeaking_mll_met_2dpdf,
    ]
    return bkgpeaking_mll_met_2dpdf, bkgnonpeak_mll_met_2dpdf, components


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


def build_model_2d_both_bkg(m1, m2):
    """
    Build (mll, met, sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf, components) for one
    mass point: the full 2D signal product pdf and the peaking-in-m(ll) and
    non-peaking-in-m(ll) 2D background product pdfs, each separately yield-
    parameterized by the caller's fit model (n_sig, n_bkg_peak, n_bkg_nonpeak) rather
    than combined here via a shape-mixture ratio.
    Caller must keep the returned vars/pdfs/components alive for as long as the model
    is in use.
    """
    global mll
    prev_cwd = os.getcwd()
    try:
        os.chdir(HBB_ZLL_TOY_DIR)
        mll = ROOT.RooRealVar("m_ll", "m_ll", 60, 120)
        sig_pdf, met, sig_components = _get_signal_model_2d(m1, m2)
        bkg_peaking_pdf, bkg_nonpeaking_pdf, bkg_components = _get_background_model_both_2d(met)
    finally:
        os.chdir(prev_cwd)

    return mll, met, sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf, sig_components + bkg_components


class ToyStudy:
    """Minimal RooMCStudy-like container: per-toy generated datasets and fit results,
    plus the exact generated n_sig/n_bkg (fixed every toy -- see run_mcstudy_2d) that a
    merged-pdf RooMCStudy generation could not recover, since it tags no event with its
    sig/bkg origin."""
    def __init__(self, n_sig_gen, n_bkg_gen, n_bkg_peaking_gen=None, n_bkg_nonpeak_gen=None):
        self.n_sig_gen = n_sig_gen
        self.n_bkg_gen = n_bkg_gen
        self.n_bkg_peaking_gen = n_bkg_peaking_gen
        self.n_bkg_nonpeak_gen = n_bkg_nonpeak_gen
        self._gen_data = []
        self._fit_results = []

    def genData(self, i):
        return self._gen_data[i]

    def fitResult(self, i):
        return self._fit_results[i]


# ── Toy generation + fitting ─────────────────────────────────────────────────────────

def run_mcstudy_2d(mll, met, sig_pdf, bkg_pdf, n_sig_in, n_bkg_in, n_sig_range, n_bkg_range,
                   n_experiments, seed):
    n_sig = ROOT.RooRealVar("n_sig", "n_sig", n_sig_in, *n_sig_range)
    n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", n_bkg_in, *n_bkg_range)
    total_pdf = ROOT.RooAddPdf("total_pdf_2d_mll_met", "total_pdf_2d_mll_met",
                               ROOT.RooArgList(sig_pdf, bkg_pdf),
                               ROOT.RooArgList(n_sig, n_bkg))

    observables = ROOT.RooArgSet(mll, met)
    ROOT.RooRandom.randomGenerator().SetSeed(seed)

    # Fixed (not Poisson-fluctuated) per-toy generated counts -- generate() below draws
    # exactly this many events from each component every toy, so the true sig/bkg split
    # is known exactly rather than only recoverable as an Extended-fit expectation.
    n_gen_sig = round(n_sig_in)
    n_gen_bkg = round(n_bkg_in)
    study = ToyStudy(n_gen_sig, n_gen_bkg)

    print(f"Running {n_experiments} toy experiments "
         f"(n_sig_in={n_sig_in}, n_bkg_in={n_bkg_in}, seed={seed})...", flush=True)
    for _ in range(n_experiments):
        sig_data = sig_pdf.generate(observables, n_gen_sig)
        bkg_data = bkg_pdf.generate(observables, n_gen_bkg)
        sig_data.append(bkg_data)

        fr = total_pdf.fitTo(sig_data, ROOT.RooFit.Extended(), ROOT.RooFit.Save(True),
                             ROOT.RooFit.PrintLevel(-1))

        study._gen_data.append(sig_data)
        study._fit_results.append(fr)

        # Reset the yields so the next toy's fit starts from the generated values, not
        # from wherever this fit ended -- otherwise one fit that runs off into an invalid
        # region (negative total yield) leaves every later fit stuck there.
        for var, val in ((n_sig, n_sig_gen), (n_bkg_peak, n_bkg_peaking_gen),
                         (n_bkg_nonpeak, n_bkg_nonpeak_gen)):
            var.setVal(val)
            var.setError(0.0)
    print("Done.", flush=True)
    return study, n_sig, n_bkg, total_pdf


def run_mcstudy_2d_both_bkg(mll, met, sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf,
                            n_sig_gen, n_bkg_peaking_gen, n_bkg_nonpeak_gen,
                            n_sig_range, n_bkg_peak_range, n_bkg_nonpeak_range,
                            n_experiments, seed):
    """Like run_mcstudy_2d, but generates AND fits the peaking-in-m(ll) and
    non-peaking-in-m(ll) background components as two independent yields
    (n_bkg_peak, n_bkg_nonpeak) instead of one total n_bkg times a shape-mixture ratio.
    Every toy generates exactly n_sig_gen/n_bkg_peaking_gen/n_bkg_nonpeak_gen events
    from sig_pdf/bkg_peaking_pdf/bkg_nonpeaking_pdf respectively (fixed counts, no
    Poisson, no probabilistic mixture split), so the true composition of every toy is
    known exactly; total_pdf = RooAddPdf(sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf;
    n_sig, n_bkg_peak, n_bkg_nonpeak) then fits all three yields independently."""
    n_sig = ROOT.RooRealVar("n_sig", "n_sig", n_sig_gen, *n_sig_range)
    n_bkg_peak = ROOT.RooRealVar("n_bkg_peak", "n_bkg_peak", n_bkg_peaking_gen, *n_bkg_peak_range)
    n_bkg_nonpeak = ROOT.RooRealVar("n_bkg_nonpeak", "n_bkg_nonpeak", n_bkg_nonpeak_gen, *n_bkg_nonpeak_range)
    total_pdf = ROOT.RooAddPdf("total_pdf_2d_mll_met", "total_pdf_2d_mll_met",
                               ROOT.RooArgList(sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf),
                               ROOT.RooArgList(n_sig, n_bkg_peak, n_bkg_nonpeak))

    observables = ROOT.RooArgSet(mll, met)
    ROOT.RooRandom.randomGenerator().SetSeed(seed)

    n_gen_sig = round(n_sig_gen)
    n_gen_bkg_peaking = round(n_bkg_peaking_gen)
    n_gen_bkg_nonpeak = round(n_bkg_nonpeak_gen)
    study = ToyStudy(n_gen_sig, n_gen_bkg_peaking + n_gen_bkg_nonpeak,
                     n_gen_bkg_peaking, n_gen_bkg_nonpeak)

    print(f"Running {n_experiments} toy experiments "
         f"(n_sig_gen={n_sig_gen}, n_bkg_peaking_gen={n_bkg_peaking_gen}, "
         f"n_bkg_nonpeak_gen={n_bkg_nonpeak_gen}, seed={seed})...", flush=True)
    for _ in range(n_experiments):
        sig_data = sig_pdf.generate(observables, n_gen_sig)
        peaking_data = bkg_peaking_pdf.generate(observables, n_gen_bkg_peaking)
        nonpeak_data = bkg_nonpeaking_pdf.generate(observables, n_gen_bkg_nonpeak)
        sig_data.append(peaking_data)
        sig_data.append(nonpeak_data)

        fr = total_pdf.fitTo(sig_data, ROOT.RooFit.Extended(), ROOT.RooFit.Save(True),
                             ROOT.RooFit.PrintLevel(-1))

        study._gen_data.append(sig_data)
        study._fit_results.append(fr)
    print("Done.", flush=True)
    return study, n_sig, n_bkg_peak, n_bkg_nonpeak, total_pdf


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
            'n_sig_gen':  mcs.n_sig_gen,
            'n_bkg_gen':  mcs.n_bkg_gen,
            'n_sig_val':  sig_par.getVal(),
            'n_sig_err':  sig_par.getError(),
            'n_bkg_val':  bkg_par.getVal(),
            'n_bkg_err':  bkg_par.getError(),
        })
    return rows


def collect_results_2d(mcs, n_experiments):
    """Pull (value, error, status) for n_sig/n_bkg_peak/n_bkg_nonpeak -- each an
    independent fitted yield -- out of every toy's fit result."""
    rows = []
    for i in range(n_experiments):
        fr = mcs.fitResult(i)
        pars = fr.floatParsFinal()
        sig_par = pars.find("n_sig")
        bkg_peak_par = pars.find("n_bkg_peak")
        bkg_nonpeak_par = pars.find("n_bkg_nonpeak")
        rows.append({
            'status':            fr.status(),
            'cov_qual':          fr.covQual(),
            'n_sig_gen':         mcs.n_sig_gen,
            'n_bkg_peaking_gen': mcs.n_bkg_peaking_gen,
            'n_bkg_nonpeak_gen': mcs.n_bkg_nonpeak_gen,
            'n_sig_val':         sig_par.getVal(),
            'n_sig_err':         sig_par.getError(),
            'n_bkg_peak_val':    bkg_peak_par.getVal(),
            'n_bkg_peak_err':    bkg_peak_par.getError(),
            'n_bkg_nonpeak_val': bkg_nonpeak_par.getVal(),
            'n_bkg_nonpeak_err': bkg_nonpeak_par.getError(),
        })
    return rows


def _first_converged_toys(rows, n):
    return [i for i, r in enumerate(rows) if r['status'] == 0][:n]


# ── Single-toy persistence ───────────────────────────────────────────────────────────

TOY_DATA_NAME = "toy_data"
TOY_META_NAME = "meta"


def save_toy_dataset(data, meta, toy_dir, basename):
    """Write one generated toy dataset to <toy_dir>/<basename>.root, together with a
    TNamed("meta") holding the JSON-serialized meta dict -- everything analyze_single_toy.py
    needs to rebuild the identical model (mass point, yield ranges, generated counts) and
    cross-check against the fit this toy originally got. Returns the written path."""
    os.makedirs(toy_dir, exist_ok=True)
    path = os.path.join(toy_dir, f"{basename}.root")

    f = ROOT.TFile.Open(path, "RECREATE")
    data.Write(TOY_DATA_NAME)
    ROOT.TNamed(TOY_META_NAME, json.dumps(meta)).Write()
    f.Close()
    return path


def load_toy_dataset(path, mll, met):
    """Read back a file written by save_toy_dataset. Returns (data, meta), where data is a
    fresh RooDataSet built on the *caller's* mll/met RooRealVars: the stored dataset carries
    its own copies of those variables, and the signal MET spline is only connected to the
    workspace's own met object, so the events are re-filled rather than used in place."""
    f = ROOT.TFile.Open(path, "READ")
    if not f or f.IsZombie():
        raise IOError(f"Could not open toy file {path}")

    stored = f.Get(TOY_DATA_NAME)
    if not stored:
        raise IOError(f"No '{TOY_DATA_NAME}' RooDataSet in {path}")
    meta_obj = f.Get(TOY_META_NAME)
    meta = json.loads(meta_obj.GetTitle()) if meta_obj else {}

    obs = ROOT.RooArgSet(mll, met)
    data = ROOT.RooDataSet(TOY_DATA_NAME, TOY_DATA_NAME, obs)
    for i in range(stored.numEntries()):
        row = stored.get(i)
        mll.setVal(row.getRealValue(mll.GetName()))
        met.setVal(row.getRealValue(met.GetName()))
        data.add(obs)
    f.Close()

    return data, meta


def make_nll(total_pdf, data):
    """One extended NLL object for a (pdf, data) pair. Absolute NLL values are only
    comparable when they come from the same object, so callers create it once and then
    re-evaluate it at each set of yield values via getVal()."""
    return total_pdf.createNLL(data, ROOT.RooFit.Extended(True))


def add_pulls(rows, n_sig_in, n_bkg_in):
    for r in rows:
        r["n_sig_pull"] = (r["n_sig_val"] - n_sig_in) / r["n_sig_err"] if r["n_sig_err"] > 0 else None
        r["n_bkg_pull"] = (r["n_bkg_val"] - n_bkg_in) / r["n_bkg_err"] if r["n_bkg_err"] > 0 else None
    return rows


def add_pulls_2d(rows, n_sig_in, n_bkg_peak_in, n_bkg_nonpeak_in):
    for row in rows:
        row["n_sig_pull"] = (row["n_sig_val"] - n_sig_in) / row["n_sig_err"] if row["n_sig_err"] > 0 else None
        row["n_bkg_peak_pull"] = ((row["n_bkg_peak_val"] - n_bkg_peak_in) / row["n_bkg_peak_err"]
                                  if row["n_bkg_peak_err"] > 0 else None)
        row["n_bkg_nonpeak_pull"] = ((row["n_bkg_nonpeak_val"] - n_bkg_nonpeak_in) / row["n_bkg_nonpeak_err"]
                                     if row["n_bkg_nonpeak_err"] > 0 else None)
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

def _marginal_chi2(frame, hist_name, curve_name, n_float):
    """chi2 of a plotted curve against the plotted (binned) data on one RooPlot frame.

    Computed by hand rather than via RooPlot::chiSquare so the raw chi2 and the ndf come
    back separately, not just their ratio. Per bin the curve is averaged over the bin
    width (RooCurve::average), as RooFit does, and compared to the bin content using the
    asymmetric Poisson error on the side the curve falls. Empty bins are skipped -- the
    RooFit convention, and unavoidable here since a Poisson chi2 has no meaningful error
    for a zero-content bin (at these yields most MET bins are empty).

    Returns (chi2, ndf, chi2/ndf), with ndf = (filled bins) - n_float."""
    hist = frame.getHist(hist_name)
    curve = frame.getCurve(curve_name)
    if hist is None or curve is None:
        return float("nan"), 0, float("nan")

    chi2 = 0.0
    n_bins = 0
    for i in range(hist.GetN()):
        x, y = hist.GetPointX(i), hist.GetPointY(i)
        if y == 0.0:
            continue
        avg = curve.average(x - hist.GetErrorXlow(i), x + hist.GetErrorXhigh(i))
        err = hist.GetErrorYlow(i) if y > avg else hist.GetErrorYhigh(i)
        if err <= 0.0:
            continue
        chi2 += ((y - avg) / err) ** 2
        n_bins += 1

    ndf = max(n_bins - n_float, 0)
    return chi2, ndf, (chi2 / ndf if ndf > 0 else float("nan"))


_OBS_CONFIGS = [
    ("mll", 40, 60.0, 120.0, "m(ll) [GeV]"),
    ("met", 60, 200.0, 1200.0, "MET [GeV]"),
]


def make_example_toy_plot_nonpeaking_bkg(mcs, mll, met, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
                                         plot_dir, tag, toy_index):
    """Plot one generated toy dataset with the total/signal/background curves
    overlaid at that toy's own best-fit yields (not the injected truth), once per
    observable (m(ll) and MET marginals of the same 2D fit)."""
    os.makedirs(plot_dir, exist_ok=True)

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
        total_pdf.plotOn(frame, ROOT.RooFit.Name("total"), ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#717581"))) # grey
        total_pdf.plotOn(frame, ROOT.RooFit.Name("bkg"),
                         ROOT.RooFit.Components(bkg_pdf.GetName()),
                         ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#3f90da"))) # total bkg light blue
        total_pdf.plotOn(frame, ROOT.RooFit.Name("sig"),
                         ROOT.RooFit.Components(sig_pdf.GetName()),
                         ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#bd1f01"))) # signal: dark red

        canv = ROOT.TCanvas(f"canv_example_toy_{obs_name}_{tag}", "", 700, 600)
        frame.SetXTitle(xlabel)
        frame.Draw()

        leg = ROOT.TLegend(0.60, 0.51, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.032)
        leg.AddEntry(frame.findObject("data"), "Toy data", "pe")
        leg.AddEntry(frame.findObject("total"), "S+B fit", "l")
        leg.AddEntry(frame.findObject("bkg"), "Background", "l")
        leg.AddEntry(frame.findObject("sig"), "Signal", "l")
        leg.AddEntry(ROOT.nullptr, f"n_{{sig}} = {sig_par.getVal():.2f} #pm {sig_par.getError():.2f}", "")
        leg.AddEntry(ROOT.nullptr, f"n_{{bkg}} = {bkg_par.getVal():.2f} #pm {bkg_par.getError():.2f}", "")
        leg.AddEntry(ROOT.nullptr, f"n_{{sig}}^{{gen}} = {mcs.n_sig_gen}, n_{{bkg}}^{{gen}} = {mcs.n_bkg_gen}", "")
        leg.Draw()

        plotname = f"example_toy{toy_index}_{obs_name}_{tag}"
        _finalize_canvas(canv, plotname, plot_dir)

        print(f"  Created {outpath} (+.png) "
             f"[n_sig_fit={n_sig.getVal():.3f}, n_bkg_fit={n_bkg.getVal():.3f}]")


def make_example_toy_plot_both_bkg(mcs, mll, met, total_pdf, sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf,
                                   n_sig, n_bkg_peak, n_bkg_nonpeak, plot_dir, tag, toy_index):
    """Plot one generated toy dataset with the total/signal/background curves
    overlaid at that toy's own best-fit yields (not the injected truth), once per
    observable (m(ll) and MET marginals of the same 2D fit). n_bkg_peak and
    n_bkg_nonpeak are independent fitted yields (no shape-mixture ratio parameter).
    Thin wrapper around plot_toy_both_bkg: pulls the toy's data and best-fit yields off
    the study, then delegates. Returns the per-observable chi2/ndf dict."""
    data = mcs.genData(toy_index)
    fr = mcs.fitResult(toy_index)
    fitted_pars = fr.floatParsFinal()
    sig_par = fitted_pars.find("n_sig")
    bkg_peak_par = fitted_pars.find("n_bkg_peak")
    bkg_nonpeak_par = fitted_pars.find("n_bkg_nonpeak")
    n_sig.setVal(sig_par.getVal())
    n_bkg_peak.setVal(bkg_peak_par.getVal())
    n_bkg_nonpeak.setVal(bkg_nonpeak_par.getVal())

    return plot_toy_both_bkg(
        data, mll, met, total_pdf, sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf,
        n_sig, n_bkg_peak, n_bkg_nonpeak,
        yield_errs={"n_sig": sig_par.getError(),
                    "n_bkg_peak": bkg_peak_par.getError(),
                    "n_bkg_nonpeak": bkg_nonpeak_par.getError()},
        gen_counts=(mcs.n_sig_gen, mcs.n_bkg_peaking_gen, mcs.n_bkg_nonpeak_gen),
        plot_dir=plot_dir, tag=tag, label=f"example_toy{toy_index}",
        title=f"Toy #{toy_index}: data + best fit")


def plot_toy_both_bkg(data, mll, met, total_pdf, sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf,
                      n_sig, n_bkg_peak, n_bkg_nonpeak, yield_errs, gen_counts,
                      plot_dir, tag, label, title=None):
    """Core toy plotter: one plot per observable (m(ll) and MET marginals of the same 2D
    fit) of `data` overlaid with total_pdf at whatever values the caller has already set on
    n_sig/n_bkg_peak/n_bkg_nonpeak. yield_errs is a dict keyed by those three names (None,
    or a missing key, drops the +/- from the legend -- e.g. for hand-picked yields that have
    no uncertainty). gen_counts is the (n_sig, n_peaking, n_nonpeak) triplet the toy was
    generated with, or None if unknown. Plots are written as <label>_<obs>_<tag>.

    Returns {"mll": {"chi2_ndf": ..., "ndf": ...}, "met": {...}} -- chi2/ndf of the total
    curve against the binned data in each marginal, with ndf reduced by the number of
    yields currently floating."""
    os.makedirs(plot_dir, exist_ok=True)

    yield_errs = yield_errs or {}
    n_float = sum(1 for v in (n_sig, n_bkg_peak, n_bkg_nonpeak) if not v.isConstant())

    peaking_name = bkg_peaking_pdf.GetName()
    nonpeaking_name = bkg_nonpeaking_pdf.GetName()

    n_total = n_sig.getVal() + n_bkg_peak.getVal() + n_bkg_nonpeak.getVal()
    # Normalize every curve to the sum of the three yields rather than to the data count
    # (RooFit's default for a pdf plotted on a frame that already holds data). For a
    # converged extended fit the two agree to within rounding, but for hand-picked yields
    # they need not, and the whole point of plotting those is to see the mis-normalization.
    norm = ROOT.RooFit.Normalization(n_total, ROOT.RooAbsReal.NumEvent)

    chi2_by_obs = {}
    obs_by_name = {"mll": mll, "met": met}
    for obs_name, nbins, xmin, xmax, xlabel in _OBS_CONFIGS:
        obs = obs_by_name[obs_name]
        frame = obs.frame(ROOT.RooFit.Bins(nbins), ROOT.RooFit.Range(xmin, xmax),
                          ROOT.RooFit.Title(title or f"{label}: data + model"))
        data.plotOn(frame, ROOT.RooFit.Name("data"))
        total_pdf.plotOn(frame, ROOT.RooFit.Name("total"), norm,
                         ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#717581"))) # grey

        total_pdf.plotOn(frame, ROOT.RooFit.Name("bkg"), norm,
                         ROOT.RooFit.Components(f"{peaking_name},{nonpeaking_name}"),
                         ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#3f90da"))) # total bkg light blue
        total_pdf.plotOn(frame, ROOT.RooFit.Name("bkg_peaking"), norm,
                         ROOT.RooFit.Components(peaking_name),
                         ROOT.RooFit.LineStyle(ROOT.kDotted), ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#b9ac70"))) # peaking tan
        total_pdf.plotOn(frame, ROOT.RooFit.Name("bkg_nonpeaking"), norm,
                         ROOT.RooFit.Components(nonpeaking_name),
                         ROOT.RooFit.LineStyle(ROOT.kDashDotted), ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#92dadd"))) # nonpeak light turquoise
        total_pdf.plotOn(frame, ROOT.RooFit.Name("sig"), norm,
                         ROOT.RooFit.Components(sig_pdf.GetName()),
                         ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#bd1f01"))) # signal: dark red

        chi2, ndf, chi2_ndf = _marginal_chi2(frame, "data", "total", n_float)
        chi2_by_obs[obs_name] = {"chi2": chi2, "ndf": ndf, "chi2_ndf": chi2_ndf}

        if obs_name == "mll":
            # Extra headroom so the legend (which spans y=0.32-0.88 of the pad) doesn't
            # overlap the m(ll) peak.
            frame.SetMaximum(frame.GetMaximum() * 5.0)

        canv = ROOT.TCanvas(f"canv_{label}_{obs_name}_{tag}", "", 700, 600)
        frame.SetXTitle(xlabel)
        frame.Draw()

        def _yield_text(sym, var, err_key):
            err = yield_errs.get(err_key)
            if err is None:
                return f"n_{{{sym}}} = {var.getVal():.1f}"
            return f"n_{{{sym}}} = {var.getVal():.1f} #pm {err:.1f}"

        leg = ROOT.TLegend(0.40, 0.32, 0.88, 0.88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.028)
        leg.AddEntry(frame.findObject("data"), "Toy data", "pe")
        leg.AddEntry(frame.findObject("total"), "S+B fit", "l")
        leg.AddEntry(frame.findObject("bkg"), "Total background", "l")
        leg.AddEntry(frame.findObject("bkg_peaking"), "Peaking background", "l")
        leg.AddEntry(frame.findObject("bkg_nonpeaking"), "Non-peaking background", "l")
        leg.AddEntry(frame.findObject("sig"), "Signal", "l")
        leg.AddEntry(ROOT.nullptr, _yield_text("sig", n_sig, "n_sig"), "")
        leg.AddEntry(ROOT.nullptr, _yield_text("peaking", n_bkg_peak, "n_bkg_peak"), "")
        leg.AddEntry(ROOT.nullptr, _yield_text("nonpeak", n_bkg_nonpeak, "n_bkg_nonpeak"), "")
        if gen_counts is not None:
            n_sig_gen, n_peak_gen, n_nonpeak_gen = gen_counts
            leg.AddEntry(ROOT.nullptr,
                        f"n_{{sig}}^{{gen}} = {n_sig_gen}, "
                        f"n_{{peaking}}^{{gen}} = {n_peak_gen}, "
                        f"n_{{nonpeak}}^{{gen}} = {n_nonpeak_gen}", "")
            n_bkg_gen_total = n_peak_gen + n_nonpeak_gen
            if n_bkg_gen_total > 0:
                r_gen = float(n_peak_gen) / float(n_bkg_gen_total)
                leg.AddEntry(ROOT.nullptr,
                            f"generated ratio r = {r_gen:.2f}", "")
        leg.AddEntry(ROOT.nullptr,
                    f"#chi^{{2}} / ndf = {chi2:.1f} / {ndf} = {chi2_ndf:.2f}", "")
        leg.Draw()

        plotname = f"{label}_{obs_name}_{tag}"
        _finalize_canvas(canv, plotname, plot_dir)

        print(f"  [n_sig={n_sig.getVal():.3f}, n_bkg_peak={n_bkg_peak.getVal():.3f}, "
             f"n_bkg_nonpeak={n_bkg_nonpeak.getVal():.3f}] "
             f"chi2/ndf({obs_name}) = {chi2_ndf:.3f}")

    return chi2_by_obs


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
    y_max = max(y + ey for y, ey in zip(y_vals, y_errs)) * 1.8 + 2

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
