#!/usr/bin/env python3
"""
fit_zpeak.py
Fits a double-sided Crystal Ball to m(ll) using the dedicated Z-peak
control-region (CRZ) MC sample (DY + ttZ_peak, produced by make_inputs.py's
zpeak mode), per era. This produces a FIXED (non-floating) m(ll) shape for
the "peaking" background component used by fit_background.py.

Adapted from 2DFit_higgsinos/hbb_zll/zpeak_fit/initial_ZPeak_fit.py: the
peaking m(ll) shape is fit to the same CRZ-selected DY+ttZ_peak MC sample
that reformat_zPeak.py builds (backgrounds_CRZ_Zpeak_{year}.root), not the
signal-region-selected "peaking" background-MC category from make_inputs.py's
bkg mode.

Model:
  peak_dcb_mll = RooCrystalBall(mll; mean, sigmaL, sigmaR, alphaL, nL, alphaR, nR)

Saves per fit:
  - JSON with fit parameters, errors, fit quality
  - 3-page vector PDF (CMS style, via cmsstyle + EPS/ghostscript merge, same
    method as fit_background.py's make_bkg_plots): a summary page followed
    by a linear- and log-scale m(ll) data/fit projection with a residual
    sub-pad
  - Workspace with the DCB pdf, parameters left CONSTANT (fixed shape,
    ready to be imported as-is by fit_background.py)

Usage:
    python3 scripts/fit_zpeak.py --era run2
    python3 scripts/fit_zpeak.py --force                  # all eras
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
            p1 = fit_result.floatParsFinal().at(i).GetName()
            p2 = fit_result.floatParsFinal().at(j).GetName()
            corr = fit_result.correlation(p1, p2)
            if abs(corr) > threshold:
                pairs.append((p1, p2, corr))
    return pairs


# ── Data loading ──────────────────────────────────────────────────────────────

def load_peaking_mc(years, mll_var):
    """Weighted RooDataSet (mll, weight_nominal) built from the dedicated
    Z-peak control-region (CRZ) MC ntuples (make_inputs.py's zpeak mode)."""
    files = config.zpeak_crz_files(years)
    if not files:
        return None, 0

    chain = ROOT.TChain("tree")
    for f in files:
        chain.Add(f)

    w = ROOT.RooRealVar("weight_nominal", "weight_nominal", -1e6, 1e6)
    ds = ROOT.RooDataSet("ds_zpeak", "Z-peak CRZ background MC (DY/ttZ-like)",
                         ROOT.RooArgSet(mll_var, w),
                         ROOT.RooFit.Import(chain), ROOT.RooFit.WeightVar(w))
    n = ds.numEntries()
    if n == 0:
        return None, 0
    return ds, n


# ── Plotting ──────────────────────────────────────────────────────────────────

def draw_projection(var, ds, pdf, xlabel, tag, data_label, pdf_shape_label,
                     data_color="#9c9ca1", fitted_color="#964a8b",
                     x_max=None, n_plot_bins=100, log_scale=False):
    """
    Single dataset vs. single PDF projection + residual sub-pad, ported
    from fit_background.py's draw_projection. If log_scale is True, the
    top pad's y-axis is drawn logarithmic (with a wider headroom multiplier
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


def make_plots(ds, pdf, mll_var, n_mc, sum_w, era_tag, pdf_path, log_file,
                mean_v, mean_e):
    logger = setup_logger(log_file)
    logger.info(f"=== Z-peak fit (peaking background MC): era={era_tag} ===")
    logger.info(f"  n_mc={n_mc}  sum_weights={sum_w:.4f}")

    mll_args = dict(
        var=mll_var, ds=ds, pdf=pdf, xlabel="m(ll) [GeV]",
        data_label="Z-peak CRZ MC (DY/ttZ)",
        pdf_shape_label=f"Crystal Ball: mean = {mean_v:.2f}  #pm {mean_e:.2f}",
        n_plot_bins=config.N_MLL_BINS)
    canvas_mll     = draw_projection(tag=f"zpeak_mll_{era_tag}", **mll_args)
    canvas_mll_log = draw_projection(tag=f"zpeak_mll_{era_tag}_log", log_scale=True, **mll_args)

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    summary = ROOT.TCanvas("c_summary", "c_summary", 850, 1100)
    text = ROOT.TPaveText(0.05, 0.75, 0.95, 0.95, "NDC")
    text.SetTextAlign(13)
    text.SetTextFont(42)
    text.SetTextSize(0.022)
    text.SetBorderSize(0)
    text.SetFillStyle(0)
    text.AddText(f"Z-peak CRZ MC (DY+ttZ_peak): n_entries = {n_mc}  "
                 f"sum_weights = {sum_w:.4f}")
    text.AddText(f"fit range: mll [{config.MLL_LO:.0f},{config.MLL_HI:.0f}]")
    text.Draw()

    # Build the 3-page vector PDF (a summary page, then linear + log scale
    # m(ll) projection) via EPS + ghostscript, same method as
    # fit_background.py's make_bkg_plots.
    pdf_stem = os.path.splitext(pdf_path)[0]
    eps_pages = []
    for i, canvas in enumerate([summary, canvas_mll, canvas_mll_log]):
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


# ── Main fit ──────────────────────────────────────────────────────────────────

def fit_one(era_tag, force=False):
    out_json = config.zpeak_fit_json(era_tag)
    out_pdf  = config.zpeak_fit_pdf(era_tag)
    out_log  = config.zpeak_fit_log(era_tag)
    out_ws   = config.zpeak_fit_ws(era_tag)

    if not force and os.path.exists(out_json):
        print(f"  EXISTS (skip): {os.path.basename(out_json)}")
        return True

    years = config.era_years(era_tag)
    mll_var = ROOT.RooRealVar("mll", "m(ll) [GeV]", config.MLL_LO, config.MLL_HI)

    ds, n_mc = load_peaking_mc(years, mll_var)
    if ds is None:
        print(f"  SKIP: no Z-peak CRZ background MC loaded "
              f"(run make_inputs.py --mode zpeak first)")
        return False
    sum_w = ds.sumEntries()
    print(f"  Loaded {n_mc} Z-peak CRZ background MC entries (sum_w={sum_w:.4f})")

    sfx = era_tag
    mean_mll   = ROOT.RooRealVar(f"peak_mean_mll_{sfx}",   "peak_mean_mll",   90, 80, 100)
    sigmal_mll = ROOT.RooRealVar(f"peak_sigmal_mll_{sfx}", "peak_sigmal_mll",  5,  1,  20)
    sigmar_mll = ROOT.RooRealVar(f"peak_sigmar_mll_{sfx}", "peak_sigmar_mll",  5,  1,  20)
    alphal_mll = ROOT.RooRealVar(f"peak_alphal_mll_{sfx}", "peak_alphal_mll",  4, 0.01, 10)
    nl_mll     = ROOT.RooRealVar(f"peak_nl_mll_{sfx}",     "peak_nl_mll",      3,  1,  10)
    alphar_mll = ROOT.RooRealVar(f"peak_alphar_mll_{sfx}", "peak_alphar_mll",  5, 0.01, 10)
    nr_mll     = ROOT.RooRealVar(f"peak_nr_mll_{sfx}",     "peak_nr_mll",      3,  1,  10)
    peak_dcb_mll = ROOT.RooCrystalBall(f"peak_dcb_mll_{sfx}", "peak_dcb_mll",
                                        mll_var, mean_mll, sigmal_mll, sigmar_mll,
                                        alphal_mll, nl_mll, alphar_mll, nr_mll)

    # Weighted MC -> SumW2Error for correctly-scaled parameter uncertainties.
    result = peak_dcb_mll.fitTo(ds, ROOT.RooFit.Save(True),
                                ROOT.RooFit.SumW2Error(True), ROOT.RooFit.PrintLevel(-1))
    print(f"  Fit: status={result.status()}  covQual={result.covQual()}  "
          f"EDM={result.edm():.2e}  minNLL={result.minNll():.4f}")
    if result.status() != 0:
        print(f"  WARNING: fit did not converge cleanly")

    pars = result_to_dict(result)
    for name, (val, err) in pars.items():
        print(f"    {name:<28} = {val:.4f} +/- {err:.4f}")

    # Freeze the shape -- this is meant to be imported as a FIXED background
    # component by fit_background.py, not refit downstream.
    for v in [mean_mll, sigmal_mll, sigmar_mll, alphal_mll, nl_mll, alphar_mll, nr_mll]:
        v.setConstant(True)

    ws = ROOT.RooWorkspace(f"ws_zpeak_{era_tag}")
    ws.Import(peak_dcb_mll)
    os.makedirs(os.path.dirname(out_ws), exist_ok=True)
    ws.writeToFile(out_ws)

    mean_v, mean_e = pars[f"peak_mean_mll_{sfx}"]
    make_plots(ds, peak_dcb_mll, mll_var, n_mc, sum_w, era_tag, out_pdf, out_log,
               mean_v, mean_e)
    print(f"  Saved {out_pdf}")

    out_data = {
        "era":        era_tag,
        "n_mc":       n_mc,
        "sum_weights": sum_w,
        "mll_range":  [config.MLL_LO, config.MLL_HI],
        "fit_status": result.status(),
        "cov_qual":   result.covQual(),
        "edm":        result.edm(),
        "min_nll":    result.minNll(),
        "params":     {n: {"val": v, "err": e} for n, (v, e) in pars.items()},
        "high_corr":  [[p1, p2, c] for p1, p2, c in high_correlations(result)],
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"  Saved {out_json}")

    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit the Z-peak m(ll) lineshape to the peaking background-MC category")
    parser.add_argument("--era",   choices=["run2", "run3", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    eras = ["run2", "run3"] if args.era == "both" else [args.era]

    n_ok, n_fail = 0, 0
    for era in eras:
        print(f"\n--- era={era} ---")
        try:
            ok = fit_one(era, force=args.force)
            if ok: n_ok += 1
            else:  n_fail += 1
        except Exception as ex:
            print(f"  ERROR: {ex}")
            import traceback; traceback.print_exc()
            n_fail += 1

    print(f"\nDone: {n_ok} fitted  {n_fail} skipped/failed")


if __name__ == "__main__":
    main()
