#!/usr/bin/env python3
"""
fit_background.py
Fits the background shapes to background MC (peaking and non-peaking
categories from make_inputs.py's bkg mode), per era, as separate 1D fits.
Adapted from 2DFit_higgsinos/hbb_zll's obtain_background_components.py.

Separate fits:
  met_peak:    Gumbel_peak(met)    <- peaking-in-m(ll) MC
  met_nonpeak: Gumbel_nonpeak(met) <- non-peaking-in-m(ll) MC
  mll_nonpeak: Exp_nonpeak(mll)    <- non-peaking-in-m(ll) MC

  The peaking m(ll) shape, peak_dcb_mll (double-sided Crystal Ball), is
  imported CONSTANT from fit_zpeak.py and not refit here. After the fits,
  all shape parameters (mu_peak, b_peak, mu_nonpeak, b_nonpeak,
  a_nonpeak_mll) are frozen at their best-fit values and the two 2D
  components are saved:
    bkg_peak    = peak_dcb_mll(mll) x Gumbel_peak(met)
    bkg_nonpeak = Exp_nonpeak(mll)  x Gumbel_nonpeak(met)
  make_workspaces.py combines them with the ratio r (ratio_peaking,
  seeded from ratio_peaking_mc saved here) and the normalizations.

No blind window is implemented yet (pre-unblinding / early validation
stage): the fit runs over the full declared (mll, met) signal-region range
[config.MLL_LO,MLL_HI] x [config.ME111T_LO,MET_HI] with nothing excluded.

Produces one fit per era: {run2, run3}.

Saves per era:
  - fit_bkg_components_{era}.json: parameters, errors and fit quality of
    each separate fit (met_peak, met_nonpeak, mll_nonpeak)
  - fit_bkg_{era}.json: summary -- weighted MC yield per category, n_bkg_mc
    (their sum) and ratio_peaking_mc (peaking fraction of n_bkg_mc)
  - 9-page PDF: a summary page (n-tuple entries and summed weights per
    category, status of each fit) followed by one linear + one log-scale
    page per category x variable (m(ll) peaking, m(ll) non-peaking, MET
    peaking, MET non-peaking), each a data/fit projection with a residual
    sub-pad, legend parameters
  - each of those 8 plots also on its own, as fit_bkg_{era}_{plot}.pdf and
    .png ({plot} = mll_peak, mll_peak_log, mll_nonpeak, ..., met_nonpeak_log);
    the 9-page PDF and these are copied to config.OUTPUT_WWW_DIR/backgrounds
  - Workspace ws_bkg_{era} with bkg_peak_{era} and bkg_nonpeak_{era}, all
    shape parameters constant.

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


def build_background_components(mll_var, met_var, era, peak_dcb_mll):
    """
    Builds the 2D background components (mll_var and met_var are the two dimensions, era_tag is used to name the variables,
    peak_dcb_mll is the double-sided crystal ball m(ll) shape from the control region)

        Peaking-in-m(ll) background: (Z+jets and TTZ):  peak_dcb_mll(mll) x Gumbel_peak(met)
        Non-peaking-in-m(ll) background: (ttbar and WZ and ZZ): Exp(mll) x Gumbel_nonpeak(met)

    The mixture (total = r*Peaking + (1-r)*Nonpeaking) is built in make_workspaces.py.

    Returns a list of the shape parameters and a dict of the PDFs (which also need to stay in scope)
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

    params = [mu_peak, b_peak, mu_nonpeak, b_nonpeak, a_nonpeak_mll]
    components = {
        "gumbel_peak": gumbel_peak, "bkg_peak": bkg_peak,
        "exp_nonpeak_mll": exp_nonpeak_mll, "gumbel_nonpeak": gumbel_nonpeak,
        "bkg_nonpeak": bkg_nonpeak,
    }
    return params, components


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bkg_mc(years, mll_var, met_var):
    """
    Weighted RooDataSets (mll, met, weight_nominal) built separately from
    the peaking and non-peaking background-MC n-tuples (make_inputs.py's
    bkg mode), restricted to the (mll, met) fit range.

    Returns (peak_ds, nonpeak_ds); either is None if its n-tuples are
    missing or empty.
    """
    cut = f"met > {config.MET_LO} && met < {config.MET_HI}"

    def _build(files_list, name):
        if not files_list:
            return None
        chain = ROOT.TChain("tree")
        for f in files_list:
            chain.Add(f)
        w = ROOT.RooRealVar("weight_nominal", "weight_nominal", -1e6, 1e6)
        ds = ROOT.RooDataSet(name, name, ROOT.RooArgSet(mll_var, met_var, w),
                              ROOT.RooFit.Import(chain), ROOT.RooFit.Cut(cut),
                              ROOT.RooFit.WeightVar(w))
        return ds if ds.numEntries() > 0 else None

    peak_ds    = _build(config.bkg_category_files('peaking', years), "ds_bkg_peak_mc")
    nonpeak_ds = _build(config.bkg_category_files('nonpeak', years), "ds_bkg_nonpeak_mc")
    return peak_ds, nonpeak_ds


def fit_component(pdf, ds, obs_var):
    """Weighted unbinned 1D fit of pdf to ds projected onto obs_var."""
    ds_1d = ds.reduce(ROOT.RooArgSet(obs_var))
    return pdf.fitTo(
        ds_1d,
        ROOT.RooFit.Save(True),
        ROOT.RooFit.SumW2Error(True),  # weighted MC -> correctly-scaled uncertainties
        ROOT.RooFit.PrintLevel(-1),
        ROOT.RooFit.Warnings(False),
    )


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

    # Plot at config.PROJECTED_LUMI: the MC is normalized to 2018 lumi. RooFit
    # does not carry the data's Rescale over to the curve, so scale both.
    lumi_scale = config.PROJECTED_LUMI / config.LUMI['2018']
    ds.plotOn(frame, ROOT.RooFit.Name("data"), ROOT.RooFit.Rescale(lumi_scale),
              ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2),
              ROOT.RooFit.MarkerColor(ROOT.TColor.GetColor(data_color)),
              ROOT.RooFit.LineColor(ROOT.TColor.GetColor(data_color)),
              ROOT.RooFit.LineWidth(2), ROOT.RooFit.MarkerSize(1))
    pdf.plotOn(frame, ROOT.RooFit.Name("pdf"),
               ROOT.RooFit.Normalization(lumi_scale, ROOT.RooAbsReal.Relative),
               ROOT.RooFit.LineColor(ROOT.TColor.GetColor(fitted_color)),
               ROOT.RooFit.LineWidth(2), ROOT.RooFit.LineStyle(1))
    frame.GetYaxis().SetTitle(f"{frame.GetYaxis().GetTitle()} "
                              f"(scaled to {config.PROJECTED_LUMI:g} fb^{{#minus1}})")

    raw_max = frame.GetMaximum()
    if log_scale:
        frame.SetMaximum(raw_max * 200)
        frame.SetMinimum(max(raw_max * 1e-4, 1e-3))
    else:
        frame.SetMaximum(raw_max * 1.5)

    # If you want to have "CMS (Private work)" and not the "CMS (Preliminary)" default text,
    # you need both of the following lines
    CMS.SetExtraText("Private work")
    CMS.SetCmsText("CMS", font=62, size=0.76)
    # Label with the projected luminosity; the plotted MC and the legend's yield are
    # scaled up to it from 2018 MC
    CMS.SetLumi(config.PROJECTED_LUMI, unit="fb", run="", round_lumi=0)

    cname  = f"c_{tag}"
    canvas = ROOT.TCanvas(cname, cname, 650, 700)

    pad1 = ROOT.TPad("p1_" + cname, "", 0, 0.28, 1, 1.0)
    pad1.SetBottomMargin(0.02); pad1.SetLeftMargin(0.20); pad1.SetRightMargin(0.04)
    if log_scale:
        pad1.SetLogy(True)
    pad1.Draw(); pad1.cd()
    frame.GetXaxis().SetLabelSize(0)
    frame.GetXaxis().SetTitleSize(0)
    frame.GetYaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleOffset(1.4)
    frame.GetYaxis().SetLabelSize(0.048)
    frame.Draw()
    CMS.CMS_lumi(pad1, iPosX=0)
    CMS.UpdatePad(pad1)

    leg = ROOT.TLegend(0.20, 0.64, 0.96, 0.88)
    leg.SetMargin(0.18)
    leg.AddEntry(frame.findObject("data"), data_label, "PE")
    n_projected = ds.sumEntries() * lumi_scale
    leg.AddEntry(ROOT.nullptr, f"{ds.numEntries()} entries in 2018 MC, {n_projected:.3g} events in "
                               f"{config.PROJECTED_LUMI:g} fb^{{#minus1}}", "")
    leg.AddEntry(frame.findObject("pdf"), pdf_shape_label, "L")
    leg.SetBorderSize(0); leg.SetTextSize(0.038)
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


def make_bkg_plots(results, pdf_components, peak_dcb_mll, mll_var, met_var,
                    era_tag, peak_ds, nonpeak_ds, zpeak_info, pdf_file, log_file):
    """results: {fit key: RooFitResult} for the separate fits (met_peak, met_nonpeak, mll_nonpeak)."""
    logger = setup_logger(log_file)
    peak_sumw    = peak_ds.sumEntries()
    nonpeak_sumw = nonpeak_ds.sumEntries()
    n_bkg        = peak_sumw + nonpeak_sumw

    logger.info(f"=== Background component fits (separate peaking / non-peaking MC): era={era_tag} ===")
    logger.info(f"  n_bkg_mc (weighted) = {n_bkg:.4f}  "
                f"[peak_sumw={peak_sumw:.4f}  nonpeak_sumw={nonpeak_sumw:.4f}  "
                f"ratio_peaking_mc={peak_sumw / n_bkg:.4f}]")
    logger.info(f"  fit range: mll [{config.MLL_LO:.0f},{config.MLL_HI:.0f}], "
                f"met [{config.MET_LO:.0f},{config.MET_HI:.0f}]")
    pars = {}
    for key, result in results.items():
        logger.info(f"  --- {key} ---")
        for name, (val, err) in result_to_dict(result).items():
            logger.info(f"  {name:<28}: {val:.6f} +/- {err:.6f}")
            pars[name] = (val, err)
        logger.info(f"  covQual={result.covQual()}  status={result.status()}  "
                    f"EDM={result.edm():.2e}  minNLL={result.minNll():.4f}")
        for p1, p2, c in high_correlations(result):
            logger.info(f"  HIGH CORR: {p1} <-> {p2} = {c:.4f}")

    mu_peak_v, mu_peak_e = pars[f"mu_peak_{era_tag}"]
    b_peak_v,  b_peak_e  = pars[f"b_peak_{era_tag}"]
    mu_np_v,   mu_np_e   = pars[f"mu_nonpeak_{era_tag}"]
    b_np_v,    b_np_e    = pars[f"b_nonpeak_{era_tag}"]
    a_np_v,    a_np_e    = pars[f"a_nonpeak_mll_{era_tag}"]

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
    text = ROOT.TPaveText(0.05, 0.65, 0.95, 0.95, "NDC")
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
    for key, result in results.items():
        text.AddText(f"{key} fit: status = {result.status()}  covQual = {result.covQual()}  "
                     f"EDM = {result.edm():.2e}")
    text.Draw()

    # Build the 9-page vector PDF (a summary page, then linear + log scale
    # per plot) via EPS + ghostscript, same method as fit_signal.py's
    # make_plots. Each plot is also saved on its own, as a single-page PDF
    # (gs from the same EPS) and a PNG.
    plots = [("mll_peak",    canvas_mll_peak),    ("mll_peak_log",    canvas_mll_peak_log),
             ("mll_nonpeak", canvas_mll_nonpeak), ("mll_nonpeak_log", canvas_mll_nonpeak_log),
             ("met_peak",    canvas_met_peak),    ("met_peak_log",    canvas_met_peak_log),
             ("met_nonpeak", canvas_met_nonpeak), ("met_nonpeak_log", canvas_met_nonpeak_log)]
    pdf_stem = os.path.splitext(pdf_file)[0]
    eps_pages = []
    for i, canvas in enumerate([summary] + [canvas for _, canvas in plots]):
        eps_path = f"{pdf_stem}_page{i}.eps"
        canvas.SaveAs(eps_path)
        eps_pages.append(eps_path)

    gs = ("gs -q -dBATCH -dNOPAUSE -dSAFER -dEPSCrop -dPDFSETTINGS=/prepress "
          "-sDEVICE=pdfwrite -dEmbedAllFonts=true -dSubsetFonts=true")
    gs_jobs = [(pdf_file, eps_pages)]
    plot_files = [pdf_file]
    for (name, canvas), eps_path in zip(plots, eps_pages[1:]):
        gs_jobs.append((f"{pdf_stem}_{name}.pdf", [eps_path]))
        canvas.SaveAs(f"{pdf_stem}_{name}.png")
        plot_files += [f"{pdf_stem}_{name}.pdf", f"{pdf_stem}_{name}.png"]

    for out_path, inputs in gs_jobs:
        cmd = f"{gs} -sOutputFile={out_path} " + " ".join(inputs)
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(f"gs failed (exit {ret}): {cmd}")
    for eps_path in eps_pages:
        os.remove(eps_path)

    logger.info(f"PDF saved: {pdf_file}")
    logger.info(f"Individual plots saved: {' '.join(plot_files[1:])}")

    os.system(f"cp {' '.join(plot_files)} {config.OUTPUT_WWW_DIR}/backgrounds")
    logger.info(f"PDF and individual plots copied to: {config.OUTPUT_WWW_DIR}/backgrounds")


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
    peak_ds, nonpeak_ds = load_bkg_mc(years, mll_var, met_var)
    if peak_ds is None or nonpeak_ds is None:
        missing = [c for c, d in [("peaking", peak_ds), ("nonpeak", nonpeak_ds)] if d is None]
        print(f"  SKIP: no background MC loaded for {missing} "
              f"(run make_inputs.py --mode bkg first)")
        zpeak_file.Close()
        return False

    # Weighted yields within the (mll, met) fit range
    peak_sumw    = peak_ds.sumEntries()
    nonpeak_sumw = nonpeak_ds.sumEntries()
    n_bkg_mc     = peak_sumw + nonpeak_sumw
    ratio_peaking_mc = peak_sumw / n_bkg_mc
    n_entries    = peak_ds.numEntries() + nonpeak_ds.numEntries()
    print(f"  Loaded {n_entries} background MC entries "
          f"(mll [{config.MLL_LO:.0f},{config.MLL_HI:.0f}], "
          f"met [{config.MET_LO:.0f},{config.MET_HI:.0f}])  "
          f"peak_sumw={peak_sumw:.4f}  nonpeak_sumw={nonpeak_sumw:.4f}  "
          f"ratio_peaking_mc={ratio_peaking_mc:.4f}")

    params, pdf_components = build_background_components(
        mll_var, met_var, era_tag, peak_dcb_mll)

    # Separate 1D fits: (key, pdf, dataset, observable)
    fits = [
        ("met_peak",    "gumbel_peak",     peak_ds,    met_var),
        ("met_nonpeak", "gumbel_nonpeak",  nonpeak_ds, met_var),
        ("mll_nonpeak", "exp_nonpeak_mll", nonpeak_ds, mll_var),
    ]
    results = {}
    for key, pdf_key, ds, obs_var in fits:
        result = fit_component(pdf_components[pdf_key], ds, obs_var)
        results[key] = result
        print(f"  Fit {key}: status={result.status()}  covQual={result.covQual()}  "
              f"EDM={result.edm():.2e}  minNLL={result.minNll():.4f}")
        for name, (val, err) in result_to_dict(result).items():
            print(f"    {name:<28} = {val:.4f} +/- {err:.4f}")
        if result.status() != 0:
            print(f"  WARNING: {key} fit did not converge cleanly")

    # Freeze the shape params at their best-fit values -- the background
    # shapes are fully determined by these MC fits, not refit by Combine.
    for p in params:
        p.setConstant(True)

    ws = ROOT.RooWorkspace(f"ws_bkg_{era_tag}")
    ws.Import(pdf_components["bkg_peak"], ROOT.RooFit.RecycleConflictNodes())
    ws.Import(pdf_components["bkg_nonpeak"], ROOT.RooFit.RecycleConflictNodes())
    os.makedirs(os.path.dirname(out_ws), exist_ok=True)
    ws.writeToFile(out_ws)
    zpeak_file.Close()

    # Separate JSON with the parameters of each component fit
    out_components_json = config.bkg_components_json(era_tag)
    components_data = {"era": era_tag}
    for key, pdf_key, ds, obs_var in fits:
        result = results[key]
        components_data[key] = {
            "pdf":         pdf_components[pdf_key].GetName(),
            "observable":  obs_var.GetName(),
            "dataset":     ds.GetName(),
            "n_entries":   ds.numEntries(),
            "sum_weights": ds.sumEntries(),
            "fit_status":  result.status(),
            "cov_qual":    result.covQual(),
            "edm":         result.edm(),
            "min_nll":     result.minNll(),
            "params":      {n: {"val": v, "err": e} for n, (v, e) in result_to_dict(result).items()},
            "high_corr":   [[p1, p2, c] for p1, p2, c in high_correlations(result)],
        }
    os.makedirs(os.path.dirname(out_components_json), exist_ok=True)
    with open(out_components_json, 'w') as f:
        json.dump(components_data, f, indent=2)
    print(f"  Saved {out_components_json}")

    out_data = {
        "era":              era_tag,
        "n_entries":        n_entries,
        "n_bkg_mc":         n_bkg_mc,
        "peak_sumw":        peak_sumw,
        "nonpeak_sumw":     nonpeak_sumw,
        "ratio_peaking_mc": ratio_peaking_mc,
        "mll_range":        [config.MLL_LO, config.MLL_HI],
        "met_range":        [config.MET_LO, config.MET_HI],
        "zpeak_json":       zpeak_json_path,
        "components_json":  out_components_json,
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"  Saved {out_json}")

    make_bkg_plots(results, pdf_components, peak_dcb_mll, mll_var, met_var, era_tag,
                   peak_ds, nonpeak_ds, zpeak_info, out_pdf, out_log)
    print(f"  Saved {out_pdf}")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit background component shapes (MET peaking/non-peaking, m(ll) non-peaking) to background MC")
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
