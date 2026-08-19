#!/usr/bin/env python3
"""
fit_background.py
Fits the 2D (mll, met) background model to the COMBINED background-MC
sample (peaking + non-peaking categories from make_inputs.py's bkg mode),
per era. Adapted from 2DFit_higgsinos/hbb_zll's obtain_background_components.py
(individual shape fits) and fit2D.py (the ratio-r AddPdf combination,
run there in "bkgOnly" mode against backgrounds_total.root -- the same
peaking+non-peaking hadd this script builds on the fly).

Model:
  bkg_total = ratio_peaking * [peak_dcb_mll(FIXED) x Gumbel_peak(met)]
            + (1-ratio_peaking) * [Exp_nonpeak(mll) x Gumbel_nonpeak(met)]

  peak_dcb_mll is a double-sided Crystal Ball with parameters imported
  CONSTANT from fit_zpeak.py's peaking-background-MC fit -- everything else
  (including ratio_peaking, i.e. "r") floats freely DURING this fit. r is
  seeded from the true MC-weighted peaking/non-peaking split so the fit
  starts close to the expected composition, then lets the joint shape fit
  adjust it, exactly like hbb_zll's fit2D.py. Once the fit converges, all
  remaining shape/mixture parameters (mu_peak, b_peak, mu_nonpeak,
  b_nonpeak, a_nonpeak_mll, ratio_peaking) are frozen at their best-fit
  values before being saved to the workspace -- the background model is
  fully fixed by this MC fit, not refit by Combine.

No blind window is implemented yet (pre-unblinding / early validation
stage): the fit runs over the full declared (mll, met) signal-region range
[config.MLL_LO,MLL_HI] x [config.ME111T_LO,MET_HI] with nothing excluded.

Produces one fit per era: {run2, run3}.

Saves per fit:
  - JSON with fit parameters, errors, fit quality, n_bkg_mc (the weighted
    MC yield, i.e. ds.sumEntries())
  - 9-page PDF: a summary page (n-tuple entries and summed weights used in
    the fit, per category) followed by one linear + one log-scale page per
    category x variable (m(ll) peaking, m(ll) non-peaking, MET peaking,
    MET non-peaking), each a data/fit projection with a residual sub-pad,
    legend parameters
  - Workspace with FIXED-parameter PDF: all shape/mixture parameters
    (mu_peak, b_peak, mu_nonpeak, b_nonpeak, a_nonpeak_mll, ratio_peaking)
    are set constant at their best-fit values before import, since they're
    fully determined by this MC fit rather than refit by Combine. The
    overall normalization (bkg_total_{era}_norm) is set separately and
    fixed in make_workspaces.py (see config.BKG_NORM).

Usage:
    python3 scripts/fit_background.py --era run2
    python3 scripts/fit_background.py --force                  # both eras
"""

import ROOT
import json
import os
import sys
import argparse
import logging
import cmsstyle as CMS

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))
import config


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
    return logger


def result_to_dict(fit_result):
    out = {}
    for var in fit_result.floatParsFinal():
        out[var.GetName()] = (var.getVal(), var.getError())
    return out


def high_correlations(fit_result, threshold=0.6):
    pairs = []
    n = fit_result.floatParsFinal().getSize()
    for i in range(n):
        for j in range(i):
            p1   = fit_result.floatParsFinal().at(i).GetName()
            p2   = fit_result.floatParsFinal().at(j).GetName()
            corr = fit_result.correlation(p1, p2)
            if abs(corr) > threshold:
                pairs.append((p1, p2, corr))
    return pairs


# ── Model builder ─────────────────────────────────────────────────────────────

def load_zpeak_shape(era_tag):
    """
    Load the peak_dcb_mll pdf (and its mll RooRealVar) from
    fit_zpeak.py's output workspace. Returns (mll_var, peak_dcb_mll, ws_file)
    -- caller must keep ws_file alive as long as the pdf is in use.
    """
    ws_path = config.zpeak_fit_ws(era_tag)
    if not os.path.exists(ws_path):
        raise FileNotFoundError(
            f"Z-peak fit workspace not found: {ws_path}\n"
            f"Run fit_zpeak.py --era {era_tag} first.")
    f = ROOT.TFile(ws_path, "READ")
    ws = f.Get(f"ws_zpeak_{era_tag}")
    if not ws:
        f.Close()
        raise RuntimeError(f"Workspace ws_zpeak_{era_tag} not found in {ws_path}")
    pdf = ws.pdf(f"peak_dcb_mll_{era_tag}")
    mll_var = ws.var("mll")
    if not pdf or not mll_var:
        f.Close()
        raise RuntimeError(f"peak_dcb_mll_{era_tag} / mll not found in {ws_path}")
    return mll_var, pdf, f


def build_background_model(mll_var, met_var, era, peak_dcb_mll, init_ratio=0.1):
    """
    Builds the 2D background model (mll_var and met_var are the two dimensions, era_tag is used to name the variables,
    peak_dcb_mll is the double-sided crystal ball m(ll) shape from the control region, init_ratio is the ratio r)

        Peaking-in-m(ll) background: (Z+jets and TTZ):  peak_dcb_mll(mll) x Gumbel_peak(met)
        Non-peaking-in-m(ll) background: (ttbar and WZ and ZZ): Exp(mll) x Gumbel_nonpeak(met)
        Total 2D background = Peaking*(1-r) + r*Nonpeaking

    Returns the total 2D background PDF, a list of the parameters, and a list of the PDFs that need to stay in scope
    """
    # peaking-in-m(ll) background
    # declare parameters
    mu_peak = ROOT.RooRealVar(f"mu_peak_{era}", f"mu_peak_{era}", 220.0, 200.0, 400.0)
    b_peak  = ROOT.RooRealVar(f"b_peak_{era}",  f"b_peak_{era}",   40.0,  10.0, 100.0)
    # MET distribution of peaking-in-m(ll) background
    gumbel_peak = ROOT.RooGenericPdf(
        f"gumbel_peak_{era}", f"gumbel_peak_{era}",
        f"(1.0/b_peak_{era}) * "
        f"exp(-(@0 - mu_peak_{era})/b_peak_{era} "
        f"     - exp(-(@0 - mu_peak_{era})/b_peak_{era}))",
        ROOT.RooArgList(met_var, mu_peak, b_peak))
    # 2D (MET x m(ll)) PDF = MET PDF x m(ll) PDF 
    bkg_peak = ROOT.RooProdPdf(f"bkg_peak_{era}", f"bkg_peak_{era}",
                                ROOT.RooArgList(peak_dcb_mll, gumbel_peak))

    # non-peaking-in-m(ll) background
    # declare parameters
    mu_nonpeak = ROOT.RooRealVar(f"mu_nonpeak_{era}", f"mu_nonpeak_{era}", 225.0, 200.0, 300.0)
    b_nonpeak  = ROOT.RooRealVar(f"b_nonpeak_{era}",  f"b_nonpeak_{era}",   60.0,  20.0, 100.0)
    gumbel_nonpeak = ROOT.RooGenericPdf(
        f"gumbel_nonpeak_{era}", f"gumbel_nonpeak_{era}",
        f"(1.0/b_nonpeak_{era}) * "
        f"exp(-(@0 - mu_nonpeak_{era})/b_nonpeak_{era} "
        f"     - exp(-(@0 - mu_nonpeak_{era})/b_nonpeak_{era}))",
        ROOT.RooArgList(met_var, mu_nonpeak, b_nonpeak))
    # non-peaking m(ll) distribution is a falling exponential
    a_nonpeak_mll = ROOT.RooRealVar(f"a_nonpeak_mll_{era}", f"a_nonpeak_mll_{era}",
                                     -0.03, -1.0, 1.0)
    exp_nonpeak_mll = ROOT.RooExponential(f"exp_nonpeak_mll_{era}", f"exp_nonpeak_mll_{era}",
                                           mll_var, a_nonpeak_mll)
    bkg_nonpeak = ROOT.RooProdPdf(f"bkg_nonpeak_{era}", f"bkg_nonpeak_{era}",
                                   ROOT.RooArgList(exp_nonpeak_mll, gumbel_nonpeak))
    
    # total background = peaking + r * non-peaking
    ratio_peaking = ROOT.RooRealVar(f"ratio_peaking_{era}", f"ratio_peaking_{era}",
                                     init_ratio, 0.0, 1.0)
    bkg_total = ROOT.RooAddPdf(f"bkg_total_{era}", f"bkg_total_{era}",
                                ROOT.RooArgList(bkg_peak, bkg_nonpeak),
                                ROOT.RooArgList(ratio_peaking))

    params = [mu_peak, b_peak, mu_nonpeak, b_nonpeak, a_nonpeak_mll, ratio_peaking]
    components = {
        "gumbel_peak": gumbel_peak, "bkg_peak": bkg_peak,
        "exp_nonpeak_mll": exp_nonpeak_mll, "gumbel_nonpeak": gumbel_nonpeak,
        "bkg_nonpeak": bkg_nonpeak,
    }
    return bkg_total, params, components


# ── Data loading ──────────────────────────────────────────────────────────────

def sum_weight(files, cut=None):
    """Total weighted yield (sum of weight_nominal) across a list of files,
    optionally restricted by a filter expression."""
    if not files:
        return 0.0
    df = ROOT.RDataFrame("tree", files)
    if cut:
        df = df.Filter(cut)
    return df.Sum("weight_nominal").GetValue()


def load_bkg_mc(years, mll_var, met_var):
    """
    Weighted RooDataSets (mll, met, weight_nominal) built from the
    background-MC categories (make_inputs.py's bkg mode): the combined
    "backgrounds_total" analogue from hbb_zll's reformat.py/fit2D.py used
    for the joint fit, plus peaking-only and non-peaking-only datasets used
    for per-category plotting.

    Returns (ds, n_entries, peak_sumw, nonpeak_sumw, peak_ds, nonpeak_ds).
    """
    peak_files    = config.bkg_category_files('peaking', years)
    nonpeak_files = config.bkg_category_files('nonpeak', years)
    files = peak_files + nonpeak_files
    if not files:
        print(f"  ERROR: no background-MC ntuples found for years {years} "
              f"(run make_inputs.py --mode bkg first)")
        return None, 0, 0.0, 0.0, None, None

    cut = f"met > {config.MET_LO} && met < {config.MET_HI}"
    peak_sumw    = sum_weight(peak_files, cut)
    nonpeak_sumw = sum_weight(nonpeak_files, cut)

    def _build(files_list, name):
        chain = ROOT.TChain("tree")
        for f in files_list:
            chain.Add(f)
        w = ROOT.RooRealVar("weight_nominal", "weight_nominal", -1e6, 1e6)
        ds = ROOT.RooDataSet(name, name, ROOT.RooArgSet(mll_var, met_var, w),
                              ROOT.RooFit.Import(chain), ROOT.RooFit.Cut(cut),
                              ROOT.RooFit.WeightVar(w))
        return ds if ds.numEntries() > 0 else None

    ds         = _build(files, "ds_bkg_mc")
    peak_ds    = _build(peak_files, "ds_bkg_peak_mc")
    nonpeak_ds = _build(nonpeak_files, "ds_bkg_nonpeak_mc")
    n = ds.numEntries() if ds is not None else 0
    return ds, n, peak_sumw, nonpeak_sumw, peak_ds, nonpeak_ds


# ── Plotting ──────────────────────────────────────────────────────────────────


def draw_projection(var, ds, pdf, xlabel, tag, data_label, pdf_shape_label,
                     data_color="#9c9ca1", fitted_color="#964a8b",
                     x_max=None, n_plot_bins=100, log_scale=False):
    """
    Single dataset vs. single PDF projection + residual sub-pad, ported
    from fit_signal.py's draw_projection. If log_scale is True, the top
    pad's y-axis is drawn logarithmic (with a wider headroom multiplier
    and a positive floor so RooFit's zero-content bins don't break the
    log axis).
    """
    lo, hi = var.getMin(), x_max or var.getMax()
    frame     = var.frame(ROOT.RooFit.Range(lo, hi), ROOT.RooFit.Title(" "), ROOT.RooFit.Bins(n_plot_bins))
    res_frame = var.frame(ROOT.RooFit.Range(lo, hi), ROOT.RooFit.Title(" "), ROOT.RooFit.Bins(n_plot_bins))

    frame.SetLineWidth(0)
    res_frame.SetLineWidth(0)

    ds.plotOn(frame, ROOT.RooFit.Name("data"),
              ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2),
              ROOT.RooFit.MarkerColor(ROOT.TColor.GetColor(data_color)),
              ROOT.RooFit.LineColor(ROOT.TColor.GetColor(data_color)),
              ROOT.RooFit.LineWidth(2), ROOT.RooFit.MarkerSize(1))
    pdf.plotOn(frame, ROOT.RooFit.Name("pdf"),
               ROOT.RooFit.LineColor(ROOT.TColor.GetColor(fitted_color)),
               ROOT.RooFit.LineWidth(2), ROOT.RooFit.LineStyle(1))

    raw_max = frame.GetMaximum()
    if log_scale:
        frame.SetMaximum(raw_max * 20)
        frame.SetMinimum(max(raw_max * 1e-4, 1e-3))
    else:
        frame.SetMaximum(raw_max * 1.5)

    # If you want to have "CMS (Private work)" and not the "CMS (Preliminary)" default text,
    # you need both of the following lines
    CMS.SetExtraText("Private work")
    CMS.SetCmsText("CMS", font=62, size=0.76)
    CMS.SetLumi(250, unit="fb", run="")

    cname  = f"c_{tag}"
    canvas = ROOT.TCanvas(cname, cname, 650, 700)

    pad1 = ROOT.TPad("p1_" + cname, "", 0, 0.28, 1, 1.0)
    pad1.SetBottomMargin(0.02); pad1.SetLeftMargin(0.20); pad1.SetRightMargin(0.04)
    if log_scale:
        pad1.SetLogy(True)
    pad1.Draw(); pad1.cd()
    frame.GetXaxis().SetLabelSize(0)
    frame.GetXaxis().SetTitleSize(0)
    frame.GetYaxis().SetTitleSize(0.055)
    frame.GetYaxis().SetTitleOffset(1.4)
    frame.GetYaxis().SetLabelSize(0.048)
    frame.Draw()
    CMS.CMS_lumi(pad1, iPosX=0)
    CMS.UpdatePad(pad1)

    leg = ROOT.TLegend(0.20, 0.66, 0.95, 0.88)
    leg.AddEntry(frame.findObject("data"), data_label, "PE")
    leg.AddEntry(frame.findObject("pdf"), pdf_shape_label, "L")
    leg.SetBorderSize(0); leg.SetTextSize(0.045)
    leg.Draw()

    canvas.cd()
    pad2 = ROOT.TPad("p2_" + cname, "", 0, 0.0, 1, 0.28)
    pad2.SetTopMargin(0.02); pad2.SetBottomMargin(0.40); pad2.SetLeftMargin(0.20); pad2.SetRightMargin(0.04)
    pad2.Draw(); pad2.cd()

    residual = frame.residHist("data", "pdf")
    res_frame.addPlotable(residual, "P")
    res_frame.GetYaxis().SetTitle("Residual")
    res_frame.GetYaxis().SetNdivisions(505)
    res_frame.GetYaxis().SetTitleSize(0.14)
    res_frame.GetYaxis().SetTitleOffset(0.55)
    res_frame.GetYaxis().SetLabelSize(0.12)
    res_frame.GetXaxis().SetTitle(xlabel)
    res_frame.GetXaxis().SetTitleSize(0.14)
    res_frame.GetXaxis().SetLabelSize(0.12)
    res_frame.Draw()

    line0 = ROOT.TLine(lo, 0, hi, 0)
    line0.SetLineColor(ROOT.TColor.GetColor(fitted_color)); line0.SetLineWidth(2)
    line0.Draw()

    canvas._refs = [pad1, pad2, frame, res_frame, leg, residual, line0]
    return canvas


def make_bkg_plots(result, pdf_components, peak_dcb_mll, mll_var, met_var,
                    era_tag, n_bkg, peak_sumw, nonpeak_sumw,
                    peak_ds, nonpeak_ds, zpeak_info, pdf_file, log_file):
    logger = setup_logger(log_file)
    pars = result_to_dict(result)
    hc   = high_correlations(result)

    logger.info(f"=== Background fit (combined peaking+non-peaking MC): era={era_tag} ===")
    logger.info(f"  n_bkg_mc (weighted) = {n_bkg:.4f}  "
                f"[MC truth: peak_sumw={peak_sumw:.4f}  nonpeak_sumw={nonpeak_sumw:.4f}]")
    logger.info(f"  fit range: mll [{config.MLL_LO:.0f},{config.MLL_HI:.0f}], "
                f"met [{config.MET_LO:.0f},{config.MET_HI:.0f}]")
    for name, (val, err) in pars.items():
        logger.info(f"  {name:<28}: {val:.6f} +/- {err:.6f}")
    logger.info(f"  covQual={result.covQual()}  status={result.status()}  "
                f"EDM={result.edm():.2e}  minNLL={result.minNll():.4f}")
    for p1, p2, c in hc:
        logger.info(f"  HIGH CORR: {p1} <-> {p2} = {c:.4f}")

    ratio_val = pars[f"ratio_peaking_{era_tag}"][0]
    mu_peak_v, mu_peak_e = pars[f"mu_peak_{era_tag}"]
    b_peak_v,  b_peak_e  = pars[f"b_peak_{era_tag}"]
    mu_np_v,   mu_np_e   = pars[f"mu_nonpeak_{era_tag}"]
    b_np_v,    b_np_e    = pars[f"b_nonpeak_{era_tag}"]
    a_np_v,    a_np_e    = pars[f"a_nonpeak_mll_{era_tag}"]
    logger.info(f"  n_peak ~ {ratio_val*n_bkg:.1f}  n_nonpeak ~ {(1-ratio_val)*n_bkg:.1f}")

    zp_pars = zpeak_info["params"]
    zp_mean = zp_pars[f"peak_mean_mll_{era_tag}"]
    zp_mean_v, zp_mean_e = zp_mean["val"], zp_mean["err"]

    peak_mll_ds    = peak_ds.reduce(ROOT.RooArgSet(mll_var))
    nonpeak_mll_ds = nonpeak_ds.reduce(ROOT.RooArgSet(mll_var))
    peak_met_ds    = peak_ds.reduce(ROOT.RooArgSet(met_var))
    nonpeak_met_ds = nonpeak_ds.reduce(ROOT.RooArgSet(met_var))

    # Peaking-in-m(ll) m(ll)
    mll_peak_args = dict(
        var=mll_var, ds=peak_mll_ds, pdf=peak_dcb_mll, xlabel="m(ll) [GeV]",
        data_label=f"Peaking-in-m(ll) MC background",
        pdf_shape_label=f"Crystal Ball: mean = {zp_mean_v:.2f}  #pm {zp_mean_e:.2f}",
        n_plot_bins=config.N_MLL_BINS)
    canvas_mll_peak     = draw_projection(tag=f"mll_peak_{era_tag}", **mll_peak_args)
    canvas_mll_peak_log = draw_projection(tag=f"mll_peak_{era_tag}_log", log_scale=True, **mll_peak_args)

    # Non-peaking-in-m(ll) m(ll)
    mll_nonpeak_args = dict(
        var=mll_var, ds=nonpeak_mll_ds, pdf=pdf_components["exp_nonpeak_mll"], xlabel="m(ll) [GeV]",
        data_label=f"Non-peaking-in-m(ll) background MC",
        pdf_shape_label=f"Exponential (a = {a_np_v:.3f}  #pm {a_np_e:.3f})",
        n_plot_bins=config.N_MLL_BINS)
    canvas_mll_nonpeak     = draw_projection(tag=f"mll_nonpeak_{era_tag}", **mll_nonpeak_args)
    canvas_mll_nonpeak_log = draw_projection(tag=f"mll_nonpeak_{era_tag}_log", log_scale=True, **mll_nonpeak_args)

    # Peaking-in-m(ll) MET: Gumbel: mu and b
    met_peak_args = dict(
        var=met_var, ds=peak_met_ds, pdf=pdf_components["gumbel_peak"], xlabel="p_{T}^{miss} [GeV]",
        data_label=f"Peaking-in-m(ll) MC background",
        pdf_shape_label=f"Gumbel (#mu = {mu_peak_v:.1f}  #pm {mu_peak_e:.1f}, b = {b_peak_v:.1f}  #pm {b_peak_e:.1f})",
        x_max=config.MET_HI, n_plot_bins=config.N_MET_BINS)
    canvas_met_peak     = draw_projection(tag=f"met_peak_{era_tag}", **met_peak_args)
    canvas_met_peak_log = draw_projection(tag=f"met_peak_{era_tag}_log", log_scale=True, **met_peak_args)

    # Non-peaking-in-m(ll) MET Gumbel
    met_nonpeak_args = dict(
        var=met_var, ds=nonpeak_met_ds, pdf=pdf_components["gumbel_nonpeak"], xlabel="p_{T}^{miss} [GeV]",
        data_label=f"Non-peaking-in-m(ll) MC background",
        pdf_shape_label=f"Gumbel (#mu = {mu_np_v:.1f}  #pm {mu_np_e:.1f}, b = {b_np_v:.1f}  #pm {b_np_e:.1f})",
        x_max=config.MET_HI, n_plot_bins=config.N_MET_BINS)
    canvas_met_nonpeak     = draw_projection(tag=f"met_nonpeak_{era_tag}", **met_nonpeak_args)
    canvas_met_nonpeak_log = draw_projection(tag=f"met_nonpeak_{era_tag}_log", log_scale=True, **met_nonpeak_args)

    os.makedirs(os.path.dirname(pdf_file), exist_ok=True)

    n_peak_entries    = peak_ds.numEntries()
    n_nonpeak_entries = nonpeak_ds.numEntries()

    summary = ROOT.TCanvas("c_summary", "c_summary", 850, 1100)
    text = ROOT.TPaveText(0.05, 0.75, 0.95, 0.95, "NDC")
    text.SetTextAlign(13)
    text.SetTextFont(42)
    text.SetTextSize(0.022)
    text.SetBorderSize(0)
    text.SetFillStyle(0)
    text.AddText(f"n_entries (peak+nonpeak) = {n_peak_entries + n_nonpeak_entries}  "
                 f"sum_weights = {n_bkg:.4f}")
    text.AddText(f"  peaking-in-m(ll):     n_entries = {n_peak_entries}  "
                 f"sum_weights = {peak_sumw:.4f}")
    text.AddText(f"  non-peaking-in-m(ll): n_entries = {n_nonpeak_entries}  "
                 f"sum_weights = {nonpeak_sumw:.4f}")
    text.AddText(f"fit range: mll [{config.MLL_LO:.0f},{config.MLL_HI:.0f}], "
                 f"met [{config.MET_LO:.0f},{config.MET_HI:.0f}]")
    text.AddText(f"fit_zpeak.py unbinned mll fit: n_entries = {zpeak_info['n_mc']}  "
                 f"sum_weights = {zpeak_info['sum_weights']:.4f}")
    text.Draw()

    # Build the 9-page vector PDF (a summary page, then linear + log scale
    # per plot) via EPS + ghostscript, same method as fit_signal.py's
    # make_plots.
    pdf_stem = os.path.splitext(pdf_file)[0]
    eps_pages = []
    for i, canvas in enumerate([summary,
                                 canvas_mll_peak, canvas_mll_peak_log,
                                 canvas_mll_nonpeak, canvas_mll_nonpeak_log,
                                 canvas_met_peak, canvas_met_peak_log,
                                 canvas_met_nonpeak, canvas_met_nonpeak_log]):
        eps_path = f"{pdf_stem}_page{i}.eps"
        canvas.SaveAs(eps_path)
        eps_pages.append(eps_path)

    cmd = (f"gs -q -dBATCH -dNOPAUSE -dSAFER -dEPSCrop -dPDFSETTINGS=/prepress "
           f"-sDEVICE=pdfwrite -dEmbedAllFonts=true -dSubsetFonts=true "
           f"-sOutputFile={pdf_file} " + " ".join(eps_pages))
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"gs merge failed (exit {ret}): {cmd}")
    for eps_path in eps_pages:
        os.remove(eps_path)

    logger.info(f"PDF saved: {pdf_file}")

    os.system(f"cp {pdf_file} {config.OUTPUT_WWW_DIR}")
    logger.info(f"PDF copied to: {config.OUTPUT_WWW_DIR}")


# ── Main fit ──────────────────────────────────────────────────────────────────

def fit_one(era_tag, force=False):
    out_json = config.bkg_fit_json(era_tag)
    out_pdf  = config.bkg_fit_pdf(era_tag)
    out_log  = config.bkg_fit_log(era_tag)
    out_ws   = config.bkg_fit_ws(era_tag)

    if not force and os.path.exists(out_json):
        print(f"  EXISTS (skip): {os.path.basename(out_json)}")
        return True

    mll_var, peak_dcb_mll, zpeak_file = load_zpeak_shape(era_tag)
    mll_var.setRange(config.MLL_LO, config.MLL_HI)
    met_var = ROOT.RooRealVar("met", "met", config.MET_LO, config.MET_HI)

    zpeak_json_path = config.zpeak_fit_json(era_tag)
    if not os.path.exists(zpeak_json_path):
        raise FileNotFoundError(
            f"Z-peak fit JSON not found: {zpeak_json_path}\n"
            f"Run fit_zpeak.py --era {era_tag} first.")
    with open(zpeak_json_path) as f:
        zpeak_info = json.load(f)

    years = config.era_years(era_tag)
    ds, n_bkg, peak_sumw, nonpeak_sumw, peak_ds, nonpeak_ds = load_bkg_mc(years, mll_var, met_var)
    if ds is None:
        print(f"  SKIP: no background MC loaded")
        zpeak_file.Close()
        return False
    print(f"  Loaded {n_bkg} background MC entries "
          f"(mll [{config.MLL_LO:.0f},{config.MLL_HI:.0f}], "
          f"met [{config.MET_LO:.0f},{config.MET_HI:.0f}])  "
          f"peak_sumw={peak_sumw:.4f}  nonpeak_sumw={nonpeak_sumw:.4f}")

    total_sumw = peak_sumw + nonpeak_sumw
    init_ratio = peak_sumw / total_sumw if total_sumw > 0 else 0.1
    print(f"   Using {init_ratio} for the ratio!")
    init_ratio = min(max(init_ratio, 1e-3), 1 - 1e-3)  # keep strictly inside (0,1)

    bkg_total, params, pdf_components = build_background_model(
        mll_var, met_var, era_tag, peak_dcb_mll, init_ratio=init_ratio)

    # Weighted MC -> SumW2Error for correctly-scaled parameter uncertainties.
    result = bkg_total.fitTo(
        ds,
        ROOT.RooFit.Save(True),
        ROOT.RooFit.SumW2Error(True),
        ROOT.RooFit.PrintLevel(-1),
        ROOT.RooFit.Warnings(False),
    )

    print(f"  Fit: status={result.status()}  covQual={result.covQual()}  "
          f"EDM={result.edm():.2e}  minNLL={result.minNll():.4f}")
    for p in params:
        print(f"    {p.GetName():<28} = {p.getVal():.4f} +/- {p.getError():.4f}")
    if result.status() != 0:
        print(f"  WARNING: fit did not converge cleanly")

    # Freeze the shape/mixture params at their best-fit values -- the
    # background model is fully determined by this MC fit, not refit by
    # Combine (only the overall norm, fixed separately in
    # make_workspaces.py, and the signal strength r float downstream).
    for p in params:
        p.setConstant(True)

    ws = ROOT.RooWorkspace(f"ws_bkg_{era_tag}")
    ws.Import(bkg_total, ROOT.RooFit.RecycleConflictNodes())
    os.makedirs(os.path.dirname(out_ws), exist_ok=True)
    ws.writeToFile(out_ws)
    zpeak_file.Close()

    n_bkg_mc = ds.sumEntries()  # weighted yield -- the physically meaningful count
    pars = result_to_dict(result)
    out_data = {
        "era":          era_tag,
        "n_entries":    n_bkg,
        "n_bkg_mc":     n_bkg_mc,
        "peak_sumw":    peak_sumw,
        "nonpeak_sumw": nonpeak_sumw,
        "init_ratio":   init_ratio,
        "mll_range":    [config.MLL_LO, config.MLL_HI],
        "met_range":    [config.MET_LO, config.MET_HI],
        "zpeak_json":   config.zpeak_fit_json(era_tag),
        "fit_status":   result.status(),
        "cov_qual":     result.covQual(),
        "edm":          result.edm(),
        "min_nll":      result.minNll(),
        "params":       {n: {"val": v, "err": e} for n, (v, e) in pars.items()},
        "high_corr":    [[p1, p2, c] for p1, p2, c in high_correlations(result)],
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"  Saved {out_json}")

    make_bkg_plots(result, pdf_components, peak_dcb_mll, mll_var, met_var, era_tag, n_bkg_mc,
                   peak_sumw, nonpeak_sumw, peak_ds, nonpeak_ds, zpeak_info, out_pdf, out_log)
    print(f"  Saved {out_pdf}")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit 2-component (mll, met) background model to data")
    parser.add_argument("--era",   choices=["run2", "run3", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    eras = ["run2", "run3"] if args.era == "both" else [args.era]
    print(f"Fitting {len(eras)} background fit(s)  [full mll x met range, no blind window]",
          flush=True)

    n_ok, n_skip, n_fail = 0, 0, 0
    for era in eras:
        print(f"\n--- era={era} ---", flush=True)
        try:
            ok = fit_one(era, force=args.force)
            if ok:   n_ok   += 1
            else:    n_skip += 1
        except Exception as ex:
            print(f"  ERROR: {ex}")
            import traceback
            traceback.print_exc()
            n_fail += 1

    print(f"\nDone: {n_ok} fitted  {n_skip} skipped  {n_fail} failed")


if __name__ == "__main__":
    main()
