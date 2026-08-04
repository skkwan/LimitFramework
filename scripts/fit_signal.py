#!/usr/bin/env python3
"""
fit_signal.py
Fits a 2D signal model using RooKeysPdf (KDE) independently for mgg and MET.
No MINUIT minimization — shape is built directly from the dataset.

Usage:
    python3 scripts/fit_signal.py --sigtype HH --mchi 300 --mlsp 0 --era run2
    python3 scripts/fit_signal.py --sigtype both --era run2 --force
"""

import ROOT
import json
import logging
import os
import sys
import argparse
import tempfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))
import config

RHO_MGG = 0.1
RHO_MET = 0.17


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


def high_correlations(result, threshold=0.6):
    """Return list of (p1, p2, corr) for |corr| > threshold."""
    pairs = []
    n     = result.floatParsFinal().getSize()
    for i in range(n):
        for j in range(i):
            p1   = result.floatParsFinal().at(i).GetName()
            p2   = result.floatParsFinal().at(j).GetName()
            corr = result.correlation(p1, p2)
            if abs(corr) > threshold:
                pairs.append((p1, p2, corr))
    return pairs


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(sigtype, mchi, mlsp, years):
    sig_years = list(dict.fromkeys(config.signal_year_for(y) for y in years))
    files = config.slim_sig_files(sigtype, mchi, mlsp, sig_years)
    if not files:
        return None, None, None

    chain = ROOT.TChain("tree")
    for f in files:
        chain.Add(f)

    mgg = ROOT.RooRealVar("mgg", "m_{#gamma#gamma} [GeV]", config.MGG_LO, config.MGG_HI)
    met = ROOT.RooRealVar("met", "p_{T}^{miss} [GeV]",     config.MET_LO, config.MET_HI)
    w   = ROOT.RooRealVar("w_lumi", "w_lumi", -1e6, 1e6)
    ds  = ROOT.RooDataSet("ds", "ds", chain, ROOT.RooArgSet(mgg, met, w), "", "w_lumi")

    if ds.numEntries() < config.MIN_EVENTS_FOR_FIT:
        return None, None, None
    return ds, mgg, met


# ── Plotting ──────────────────────────────────────────────────────────────────

def draw_projection(var, ds, pdf, xlabel, xmax=None, tag=""):
    lo = var.getMin()
    hi = xmax if xmax else var.getMax()
    if var.GetName() == "mgg":
        lo, hi = 115, 135

    frame     = var.frame(ROOT.RooFit.Range(lo, hi), ROOT.RooFit.Title(" "))
    res_frame = var.frame(ROOT.RooFit.Range(lo, hi), ROOT.RooFit.Title(" "))

    ds.plotOn(frame,  ROOT.RooFit.Name("data"))
    pdf.plotOn(frame, ROOT.RooFit.Name("pdf"),
               ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.LineWidth(3))

    cname  = f"c_{tag}"
    canvas = ROOT.TCanvas(cname, cname, 650, 700)

    pad1 = ROOT.TPad("p1_" + cname, "", 0, 0.28, 1, 1.0)
    pad1.SetBottomMargin(0.02); pad1.SetLeftMargin(0.14)
    pad1.Draw(); pad1.cd()
    frame.GetXaxis().SetLabelSize(0)
    frame.GetXaxis().SetTitleSize(0)
    frame.GetYaxis().SetTitleSize(0.055)
    frame.GetYaxis().SetLabelSize(0.048)
    frame.Draw()

    leg = ROOT.TLegend(0.55, 0.70, 0.92, 0.88)
    leg.AddEntry(frame.findObject("data"), "Signal MC", "PE")
    leg.AddEntry(frame.findObject("pdf"),  "RooKeysPdf", "L")
    leg.SetBorderSize(0); leg.SetTextSize(0.040)
    leg.Draw()

    canvas.cd()
    pad2 = ROOT.TPad("p2_" + cname, "", 0, 0.0, 1, 0.28)
    pad2.SetTopMargin(0.02); pad2.SetBottomMargin(0.40); pad2.SetLeftMargin(0.14)
    pad2.Draw(); pad2.cd()

    residual = frame.residHist("data", "pdf")
    res_frame.addPlotable(residual, "P")
    res_frame.GetYaxis().SetTitle("Residual")
    res_frame.GetYaxis().SetNdivisions(505)
    res_frame.GetYaxis().SetTitleSize(0.14)
    res_frame.GetYaxis().SetTitleOffset(0.30)
    res_frame.GetYaxis().SetLabelSize(0.12)
    res_frame.GetXaxis().SetTitle(xlabel)
    res_frame.GetXaxis().SetTitleSize(0.14)
    res_frame.GetXaxis().SetLabelSize(0.12)
    res_frame.Draw()

    line0 = ROOT.TLine(lo, 0, hi, 0)
    line0.SetLineColor(ROOT.kBlue); line0.SetLineWidth(2)
    line0.Draw()

    path = os.path.join(tempfile.mkdtemp(), f"{tag}.png")
    canvas.SaveAs(path)
    canvas._refs = [pad1, pad2, frame, res_frame, leg, residual, line0]
    return path, canvas


def make_plots(pdf_mgg, pdf_met, ds, mgg, met, mchi, pdf_path, log_file):
    logger = setup_logger(log_file)
    logger.info(f"RooKeysPdf  rho_mgg={RHO_MGG}  rho_met={RHO_MET}")
    logger.info(f"n_entries={ds.numEntries()}  sum_weights={ds.sumEntries():.4f}")

    mgg_ds = ds.reduce(ROOT.RooArgSet(mgg))
    met_ds = ds.reduce(ROOT.RooArgSet(met))

    path_mgg, _ = draw_projection(mgg, mgg_ds, pdf_mgg,
                                   r"m_{\gamma\gamma} [GeV]", tag="mgg")
    path_met, _ = draw_projection(met, met_ds, pdf_met,
                                   r"p_{T}^{miss} [GeV]",
                                   xmax=min(1.2 * mchi, config.MET_HI), tag="met")

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        # Page 1: summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        lines = [f"RooKeysPdf  rho_mgg={RHO_MGG}  rho_met={RHO_MET}",
                 f"n_entries={ds.numEntries()}  sum_weights={ds.sumEntries():.4f}"]
        ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        pdf.savefig(fig); plt.close(fig)

        # Pages 2 & 3: projections
        for path, title, rho in [(path_mgg, r"$m_{\gamma\gamma}$", RHO_MGG),
                                  (path_met, r"$p_T^{\rm miss}$",   RHO_MET)]:
            fig, ax = plt.subplots(figsize=(7, 7.5))
            ax.imshow(mpimg.imread(path)); ax.axis('off')
            ax.set_title(f"{title}  (rho={rho})", fontsize=14)
            pdf.savefig(fig); plt.close(fig)

    logger.info(f"PDF saved: {pdf_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def fit_one(sigtype, mchi, mlsp, years, era_tag, force=False):
    out_json = config.sig_fit_json(sigtype, mchi, mlsp, era_tag)
    out_pdf  = config.sig_fit_pdf(sigtype, mchi, mlsp, era_tag)
    out_log  = config.sig_fit_log(sigtype, mchi, mlsp, era_tag)
    out_ws   = config.sig_fit_ws(sigtype, mchi, mlsp, era_tag)

    if not force and os.path.exists(out_json):
        print(f"  EXISTS (skip): {os.path.basename(out_json)}")
        return True

    ds, mgg, met = load_data(sigtype, mchi, mlsp, years)
    if ds is None:
        print(f"  SKIP: too few events")
        return False

    print(f"  Loaded {ds.numEntries()} entries (sum_w={ds.sumEntries():.4f})")

    mgg_ds = ds.reduce(ROOT.RooArgSet(mgg))
    met_ds = ds.reduce(ROOT.RooArgSet(met))

    pdf_mgg = ROOT.RooKeysPdf("pdf_mgg", "pdf_mgg",
                               mgg, mgg_ds, ROOT.RooKeysPdf.NoMirror, RHO_MGG)
    pdf_met = ROOT.RooKeysPdf("pdf_met", "pdf_met",
                               met, met_ds, ROOT.RooKeysPdf.NoMirror, RHO_MET)

    sig_2d  = ROOT.RooProdPdf("sig_2d", "sig_2d",
                               ROOT.RooArgList(pdf_mgg, pdf_met))

    # Save workspace
    sig_2d.SetName(f"sig_pdf_{sigtype}_mchi{mchi}_mlsp{mlsp}")
    ws = ROOT.RooWorkspace(f"ws_sig_{sigtype}_mchi{mchi}_mlsp{mlsp}")
    ws.Import(sig_2d)
    os.makedirs(os.path.dirname(out_ws), exist_ok=True)
    ws.writeToFile(out_ws)

    # Save JSON
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({
            "sigtype":     sigtype,
            "mchi":        mchi,
            "mlsp":        mlsp,
            "era":         era_tag,
            "rho_mgg":     RHO_MGG,
            "rho_met":     RHO_MET,
            "n_entries":   ds.numEntries(),
            "sum_weights": ds.sumEntries(),
        }, f, indent=2)
    print(f"  Saved {out_json}")

    make_plots(pdf_mgg, pdf_met, ds, mgg, met, mchi, out_pdf, out_log)
    print(f"  Saved {out_pdf}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigtype", choices=["HH", "ZH", "both"], default="both")
    parser.add_argument("--mchi",    type=int, default=None)
    parser.add_argument("--mlsp",    type=int, default=None)
    parser.add_argument("--era",     choices=["run2", "run3", "run2run3"], default="run2")
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    years    = config.era_years(args.era)
    sigtypes = ["HH", "ZH"] if args.sigtype == "both" else [args.sigtype]

    mass_points = [(st, mc, ml)
                   for st in sigtypes
                   for mc, ml in config.SIGNAL_GRID[st]
                   if (args.mchi is None or mc == args.mchi)
                   and (args.mlsp is None or ml == args.mlsp)]

    print(f"Fitting {len(mass_points)} mass points  [era={args.era}, rho_mgg={RHO_MGG}, rho_met={RHO_MET}]")

    n_ok, n_skip, n_fail = 0, 0, 0
    for sigtype, mchi, mlsp in mass_points:
        print(f"\n--- {sigtype} mchi={mchi} mlsp={mlsp} ---")
        try:
            ok = fit_one(sigtype, mchi, mlsp, years, args.era, force=args.force)
            if ok: n_ok   += 1
            else:  n_skip += 1
        except Exception as ex:
            print(f"  ERROR: {ex}")
            n_fail += 1

    print(f"\nDone: {n_ok} fitted  {n_skip} skipped  {n_fail} failed")


if __name__ == "__main__":
    main()