#!/usr/bin/env python3
"""
fit_signal.py
Fits a 2D signal model per mass point: RooCrystalBall (double-sided Crystal
Ball) for m(ll), times a spline-based KDE-like PDF for MET, built directly
from the signal MC MET histogram. Adapted from
2DFit_higgsinos/hbb_zll/individual_pdf_fits/obtain_signal_components.py.

NOTE: RooSpline requires ROOT >= 6.38 -- if you're running inside a CMSSW
environment with an older ROOT, source a standalone ROOT 6.38+ instead
(matching the hbb_zll reference scripts' convention).

Usage:
    python3 scripts/fit_signal.py --mchi 300 --mlsp 0 --era run2
    python3 scripts/fit_signal.py --era run2 --force
"""

import ROOT
from ROOT import RooFit as RF
import json
import logging
import os
import sys
import argparse
import numpy as np
import cmsstyle as CMS

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))
import config

# ── Helpers ───────────────────────────────────────────────────────────────────

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


def high_correlations(result, threshold=0.6):
    """Return list of (p1, p2, corr) for |corr| > threshold."""
    pairs = []
    n = result.floatParsFinal().getSize()
    for i in range(n):
        for j in range(i):
            p1   = result.floatParsFinal().at(i).GetName()
            p2   = result.floatParsFinal().at(j).GetName()
            corr = result.correlation(p1, p2)
            if abs(corr) > threshold:
                pairs.append((p1, p2, corr))
    return pairs


# ── MET spline construction (ported from hbb_zll obtain_signal_components.py) ──

def make_knot_x(x_min, x_max, n_knots, power=3.0, min_dx_bins=2, centers=None):
    """
    Construct knots spanning (x_min, x_max). `power` controls spacing
    (power>1 packs knots more densely at low x). If `centers` (bin centers)
    is given, snap knots to the nearest bin and enforce minimum spacing.
    """
    n_knots = int(max(2, n_knots))
    x_min, x_max = float(x_min), float(x_max)

    print(f"x_min: {x_min}, x_max: {x_max}")

    t = np.linspace(0.0, 1.0, n_knots)
    knot_x = x_min + (x_max - x_min) * (t ** power)

    print(knot_x)

    if centers is not None and len(centers) > 0:
        knot_x = np.array([centers[np.argmin(np.abs(centers - xx))] for xx in knot_x],
                          dtype=float)

        print(knot_x)

    if centers is not None and min_dx_bins is not None and min_dx_bins > 0:
        bw = float(np.median(np.diff(centers))) if len(centers) > 1 else 1.0
        min_dx = min_dx_bins * bw
        print("min_dx = ", min_dx)
        filtered = [knot_x[0]]
        for xx in knot_x[1:]:
            if xx - filtered[-1] >= min_dx:
                filtered.append(xx)
        if filtered[-1] != knot_x[-1]:
            filtered.append(knot_x[-1])
        knot_x = np.array(filtered, dtype=float)

    print(knot_x)
    return knot_x


def build_met_spline_pdf(met_hist, met_var, tag, n_knots=240, power=5, min_y=1e-8,
                         knot_avg_halfwidth_bins=2):
    """
    Build a RooSpline-based PDF over met_var from a (weighted) MET histogram,
    with knots densified at low MET via `power`.
    """
    nBins   = met_hist.GetNbinsX()
    centers = np.array([met_hist.GetBinCenter(i) for i in range(1, nBins + 1)])
    vals    = np.array([max(float(met_hist.GetBinContent(i)), 0.0)
                        for i in range(1, nBins + 1)])

    knot_x = make_knot_x(centers[0], centers[-1], n_knots, power=power, centers=centers)

    knot_y = []
    for xx in knot_x:
        ib = int(np.argmin(np.abs(centers - xx)))
        i0 = max(0, ib - knot_avg_halfwidth_bins)
        i1 = min(nBins - 1, ib + knot_avg_halfwidth_bins)
        local = vals[i0:i1 + 1]
        yk = float(np.mean(local)) if local.size else float(vals[ib])
        knot_y.append(max(yk, min_y))

    vx = ROOT.std.vector('double')(np.array(knot_x, dtype=np.double))
    vy = ROOT.std.vector('double')(np.array(knot_y, dtype=np.double))

    print(vx, vy)
    spline = ROOT.RooSpline(f"spline_{tag}", "spline", met_var, vx, vy, order=3)
    pdf_of_spline = ROOT.RooGenericPdf(
        f"pdf_of_spline_{tag}", "pdf_of_spline",
        f"(@1 < {met_var.getMin()}) ? 1e-12 : max(@0, 1e-12)",
        ROOT.RooArgList(spline, met_var))
    return pdf_of_spline, spline


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(mchi, mlsp, years):
    sig_years = list(dict.fromkeys(config.signal_year_for(y) for y in years))
    files = config.ntuple_sig_files(mchi, mlsp, sig_years)
    if not files:
        return None, None, None

    chain = ROOT.TChain("tree")
    for f in files:
        chain.Add(f)

    mll = ROOT.RooRealVar("mll", "m(ll) [GeV]", config.MLL_LO, config.MLL_HI)
    met = ROOT.RooRealVar("met", "p_{T}^{miss} [GeV]", config.MET_LO, config.MET_HI)
    w   = ROOT.RooRealVar("weight_nominal", "weight_nominal", -1e6, 1e6)
    ds  = ROOT.RooDataSet("ds", "ds", ROOT.RooArgSet(mll, met, w),
                          ROOT.RooFit.Import(chain), ROOT.RooFit.WeightVar(w))

    if ds.numEntries() < config.MIN_EVENTS_FOR_FIT:
        return None, None, None
    return ds, mll, met


# ── Plotting ──────────────────────────────────────────────────────────────────

def draw_projection(var, ds, pdf, xlabel, era_tag, mc_legend_label, xmax=None, tag="", nFloatParams=2, title=None, n_plot_bins=100):
    """
    Build the data/fit + residual canvas for one variable. Returns the
    TCanvas itself (kept alive via canvas._refs) so callers can Print()
    it as a page of a multi-page vector PDF, rather than rasterizing to
    PNG first.

    xlabel is the x-axis label
    mc_legend_label is the legend label for the MC curve
    """
    lo = var.getMin()
    hi = xmax if xmax else var.getMax()

    frame     = var.frame(ROOT.RooFit.Range(lo, hi), ROOT.RooFit.Title(" "), ROOT.RooFit.Bins(n_plot_bins))
    res_frame = var.frame(ROOT.RooFit.Range(lo, hi), ROOT.RooFit.Title(" "), ROOT.RooFit.Bins(n_plot_bins))
    # RooPlot's internal dummy histogram (all-zero bins) draws a hidden baseline at y=0
    # across the full frame width, which overlaps the bottom axis line and makes it look
    # doubled/thicker than the top/left/right edges. Hide that baseline explicitly.
    frame.SetLineWidth(0)
    res_frame.SetLineWidth(0)

    ds.plotOn(frame, ROOT.RooFit.Name("data"),
             ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#5790fc")),
             ROOT.RooFit.LineWidth(2),
             ROOT.RooFit.MarkerColor(ROOT.TColor.GetColor("#5790fc")),
             ROOT.RooFit.MarkerSize(1))
    pdf.plotOn(frame, ROOT.RooFit.Name("pdf"),
              ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#f89c20")),
              ROOT.RooFit.LineWidth(2), ROOT.RooFit.LineStyle(1))

    # data_hist = ds.createHistogram(f"data_hist_{tag}", var, ROOT.RooFit.Binning(50, lo, hi))
    # pdf_hist  = pdf.createHistogram(f"pdf_hist_{tag}", var, ROOT.RooFit.Binning(50, lo, hi))
    # y_max = 1.4 * max(data_hist.GetMaximum(), pdf_hist.GetMaximum())
    frame.SetMaximum(frame.GetMaximum() * 1.5)

    # If you want to have "CMS (Private work)" and not the "CMS (Preliminary)" default text,
    # you need both of the following lines
    CMS.SetExtraText("Private work")
    CMS.SetCmsText("CMS", font=62, size=0.76)
    # TODO: fix this for better eras
    lumi = 250
    run_label = ""
    # lumi = config.LUMI_RUN2 if era_tag == "run2" else config.LUMI_RUN3
    # run_label = "Run 2" if era_tag == "run2" else "Run 3"
    CMS.SetLumi(lumi, unit="fb", run=run_label)

    cname  = f"c_{tag}"
    canvas = ROOT.TCanvas(cname, cname, 650, 700)

    pad1 = ROOT.TPad("p1_" + cname, "", 0, 0.28, 1, 1.0)
    pad1.SetBottomMargin(0.02); pad1.SetLeftMargin(0.16)
    pad1.Draw(); pad1.cd()
    frame.GetXaxis().SetLabelSize(0)
    frame.GetXaxis().SetTitleSize(0)
    frame.GetYaxis().SetTitleSize(0.055)
    frame.GetYaxis().SetLabelSize(0.048)
    frame.Draw()
    CMS.CMS_lumi(pad1, iPosX=0)
    CMS.UpdatePad(pad1)

    leg = ROOT.TLegend(0.25, 0.70, 0.88, 0.88)
    chi2_per_ndf = frame.chiSquare("pdf", "data", nFloatParams)
    leg.AddEntry(frame.findObject("data"), mc_legend_label, "PE")
    if "met" in tag: 
        leg.AddEntry(frame.findObject("pdf"),  f"Fit (spline)", "L")
    else:
        leg.AddEntry(frame.findObject("pdf"),  f"Fit ( #chi^{{2}}/ndf = {chi2_per_ndf:.2f})", "L")

    leg.SetBorderSize(0); leg.SetTextSize(0.040)
    leg.Draw()

    canvas.cd()
    pad2 = ROOT.TPad("p2_" + cname, "", 0, 0.0, 1, 0.28)
    pad2.SetTopMargin(0.02); pad2.SetBottomMargin(0.40); pad2.SetLeftMargin(0.16)
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
    line0.SetLineColor(ROOT.TColor.GetColor("#f89c20")); line0.SetLineWidth(2)
    line0.Draw()

    canvas.cd()
    header = None
    if title:
        header = ROOT.TLatex()
        header.SetNDC()
        header.SetTextAlign(21)
        header.SetTextFont(42)
        header.SetTextSize(0.028)
        header.DrawLatex(0.5, 0.965, title)

    # ROOT 6.38 TPDF clips TLatex drawn in sub-pad top margins (e.g. the
    # CMS label) unless clipping is disabled on the sub-pads first.
    for pad in (pad1, pad2):
        pad.ResetBit(ROOT.TPad.kClipFrame)

    canvas._refs = [pad1, pad2, frame, res_frame, leg, residual, line0, header]
    return canvas


def make_plots(pdf_mll, pdf_met, ds, mll, met, mchi, mlsp, era_tag, pdf_path, log_file):
    logger = setup_logger(log_file)
    logger.info(f"n_entries={ds.numEntries()}  sum_weights={ds.sumEntries():.4f}")

    mll_ds = ds.reduce(ROOT.RooArgSet(mll))
    met_ds = ds.reduce(ROOT.RooArgSet(met))

    base_MC_legend_label = f"Signal: ({mchi}, {mlsp}) GeV"
    canvas_mll = draw_projection(mll, mll_ds, pdf_mll, "m(ll) [GeV]", era_tag, base_MC_legend_label,
                                 tag="mll", nFloatParams=7, title="",
                                 n_plot_bins=config.N_MLL_BINS)
    canvas_met = draw_projection(met, met_ds, pdf_met, "p_{T}^{miss} [GeV]", era_tag, base_MC_legend_label,
                                 xmax=config.MET_HI,
                                 tag="met",
                                 title="", # p_{T}^{miss}: spline
                                 n_plot_bins=config.N_MET_BINS)
 

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    summary = ROOT.TCanvas("c_summary", "c_summary", 850, 1100)
    text = ROOT.TPaveText(0.05, 0.85, 0.95, 0.95, "NDC")
    text.SetTextAlign(13)
    text.SetTextFont(42)
    text.SetTextSize(0.022)
    text.SetBorderSize(0)
    text.SetFillStyle(0)
    text.AddText(f"n_entries={ds.numEntries()}  sum_weights={ds.sumEntries():.4f}")
    text.Draw()

    # Build one multi-page vector PDF via EPS + ghostscript: each page is
    # saved as EPS, then gs merges them (in page order) into a single
    # pdfwrite output. Same export method as obtain_signal_components.py.
    pdf_stem = os.path.splitext(pdf_path)[0]
    eps_pages = []
    for i, canvas in enumerate([summary, canvas_mll, canvas_met]):
        eps_path = f"{pdf_stem}_page{i}.eps"
        canvas.SaveAs(eps_path)
        eps_pages.append(eps_path)

    cmd = (f"gs -q -dBATCH -dNOPAUSE -dSAFER -dEPSCrop -dPDFSETTINGS=/prepress "
           f"-sDEVICE=pdfwrite -dEmbedAllFonts=true -dSubsetFonts=true "
           f"-sOutputFile={pdf_path} " + " ".join(eps_pages))
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"gs merge failed (exit {ret}): {cmd}")
    for eps_path in eps_pages:
        os.remove(eps_path)

    logger.info(f"PDF saved: {pdf_path}")

    os.system(f"cp {pdf_path} {config.OUTPUT_WWW_DIR}")
    logger.info(f"PDF copied to: {config.OUTPUT_WWW_DIR}")

# ── Main ──────────────────────────────────────────────────────────────────────

def fit_one(mchi, mlsp, years, era_tag, force=False):
    out_json = config.sig_fit_json(mchi, mlsp, era_tag)
    out_pdf  = config.sig_fit_pdf(mchi, mlsp, era_tag)
    out_log  = config.sig_fit_log(mchi, mlsp, era_tag)
    out_ws   = config.sig_fit_ws(mchi, mlsp, era_tag)

    if not force and os.path.exists(out_json):
        print(f"  EXISTS (skip): {os.path.basename(out_json)}")
        return True

    ds, mll, met = load_data(mchi, mlsp, years)
    if ds is None:
        print(f"  SKIP: too few events")
        return False

    print(f"  Loaded {ds.numEntries()} entries (sum_w={ds.sumEntries():.4f})")
    tag = f"mchi{mchi}_mlsp{mlsp}_{era_tag}"

    # ── m(ll): double-sided Crystal Ball, unbinned 1D fit ──
    mll_ds     = ds.reduce(ROOT.RooArgSet(mll))
    mean_mll   = ROOT.RooRealVar(f"mean_mll_{tag}",   "mean_mll",   90, 85, 95)
    sigmal_mll = ROOT.RooRealVar(f"sigmal_mll_{tag}", "sigmal_mll", 1.7, 0.1, 50)
    sigmar_mll = ROOT.RooRealVar(f"sigmar_mll_{tag}", "sigmar_mll", 0.5, 0.1, 50)
    alphal_mll = ROOT.RooRealVar(f"alphal_mll_{tag}", "alphal_mll", 2.4, 1, 50)
    nl_mll     = ROOT.RooRealVar(f"nl_mll_{tag}",     "nl_mll",     2.8, 0.5, 20)
    alphar_mll = ROOT.RooRealVar(f"alphar_mll_{tag}", "alphar_mll", 2.4, 1, 50)
    nr_mll     = ROOT.RooRealVar(f"nr_mll_{tag}",     "nr_mll",     2.25, 0.5, 20)
    pdf_mll = ROOT.RooCrystalBall(f"pdf_mll_{tag}", "pdf_mll", mll,
                                   mean_mll, sigmal_mll, sigmar_mll,
                                   alphal_mll, nl_mll, alphar_mll, nr_mll)

    result = pdf_mll.fitTo(mll_ds, RF.Save(), RF.SumW2Error(True), RF.PrintLevel(-1))
    print(f"  m(ll) fit: status={result.status()}  covQual={result.covQual()}  "
          f"EDM={result.edm():.2e}  minNLL={result.minNll():.4f}")
    if result.status() != 0:
        print(f"  WARNING: m(ll) fit did not converge cleanly")

    # ── MET: spline-based PDF from the (weighted) signal MET histogram ──
    met_ds   = ds.reduce(ROOT.RooArgSet(met))
    met_hist = met_ds.createHistogram(f"sig_met_hist_{tag}", met,
                                      RF.Binning(config.N_MET_BINS, config.MET_LO, config.MET_HI))
    pdf_met, spline = build_met_spline_pdf(met_hist, met, tag, n_knots=config.N_MET_BINS, power=5, knot_avg_halfwidth_bins=1)

    # ── Combine into the 2D product PDF. No further joint fit is needed --
    # pdf_met has no floating shape parameters, so the product's likelihood
    # is already maximized by the m(ll) fit above. ──
    sig_2d = ROOT.RooProdPdf(f"sig_pdf_{tag}", "sig_pdf", ROOT.RooArgList(pdf_mll, pdf_met))

    pars = result_to_dict(result)
    for name, (val, err) in pars.items():
        print(f"    {name:<28} = {val:.4f} +/- {err:.4f}")

    # Save workspace: the individual 1D pdf_mll/pdf_met pieces plus the
    # fitted 2D product pdf, with all mll shape params floating so Combine
    # can refit if desired, matching the un-fixed convention used elsewhere
    # in this framework for signal shapes.
    ws = ROOT.RooWorkspace(f"ws_sig_mchi{mchi}_mlsp{mlsp}")
    ws.Import(pdf_mll)
    ws.Import(pdf_met)
    ws.Import(sig_2d, RF.RecycleConflictNodes())
    os.makedirs(os.path.dirname(out_ws), exist_ok=True)
    ws.writeToFile(out_ws)

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({
            "mchi": mchi, "mlsp": mlsp, "era": era_tag,
            "n_entries": ds.numEntries(), "sum_weights": ds.sumEntries(),
            "fit_status": result.status(), "cov_qual": result.covQual(),
            "edm": result.edm(), "min_nll": result.minNll(),
            "params": {n: {"val": v, "err": e} for n, (v, e) in pars.items()},
            "high_corr": [[p1, p2, c] for p1, p2, c in high_correlations(result)],
        }, f, indent=2)
    print(f"  Saved {out_json}")

    make_plots(pdf_mll, pdf_met, ds, mll, met, mchi, mlsp, era_tag, out_pdf, out_log)
    print(f"  Saved {out_pdf}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mchi",    type=int, default=None)
    parser.add_argument("--mlsp",    type=int, default=None)
    parser.add_argument("--era",     choices=["run2", "run3", "run2run3"], default="run2")
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    years = config.era_years(args.era)

    mass_points = [(mc, ml) for mc, ml in config.SIGNAL_GRID
                   if (args.mchi is None or mc == args.mchi)
                   and (args.mlsp is None or ml == args.mlsp)]

    print(f"Fitting {len(mass_points)} mass points  [era={args.era}]")

    n_ok, n_skip, n_fail = 0, 0, 0
    for mchi, mlsp in mass_points:
        print(f"\n--- mchi={mchi} mlsp={mlsp} ---")
        try:
            ok = fit_one(mchi, mlsp, years, args.era, force=args.force)
            if ok: n_ok   += 1
            else:  n_skip += 1
        except Exception as ex:
            print(f"  ERROR: {ex}")
            import traceback; traceback.print_exc()
            n_fail += 1

    print(f"\nDone: {n_ok} fitted  {n_skip} skipped  {n_fail} failed")


if __name__ == "__main__":
    main()
