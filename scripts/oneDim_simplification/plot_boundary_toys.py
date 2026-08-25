#!/usr/bin/env python3
"""
plot_boundary_toys.py
Reads a results_*.root file produced by do_1d_mll_toys.py (fitResult_i / genData_i
per toy) and plots the first N toys whose best-fit n_sig sits at the upper edge of
its fit range (n_sig_hi) -- i.e. the fit ran into the boundary rather than converging
to an interior minimum.

Usage:
    python3 plot_boundary_toys.py --results /path/to/results_..._N5000_seed1.root \
        --m1 650 --m2 1 -n 3
"""

import ROOT
import os
import sys
import argparse

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))
from do_1d_mll_toys import build_model_1d


def find_boundary_toys(results_path, n_max):
    """Return (indices of first n_max toys with n_sig pegged at the max n_sig
    seen across all toys, that max value)."""
    f = ROOT.TFile.Open(results_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open {results_path}")

    keys = [k.GetName() for k in f.GetListOfKeys() if k.GetName().startswith("fitResult_")]
    n_experiments = len(keys)

    vals = {}
    for i in range(n_experiments):
        fr = f.Get(f"fitResult_{i}")
        if not fr:
            continue
        par = fr.floatParsFinal().find("n_sig")
        vals[i] = par.getVal()

    n_sig_hi = max(vals.values())
    boundary_idx = [i for i in sorted(vals) if abs(vals[i] - n_sig_hi) < 1e-4]
    print(f"n_sig upper boundary (max best-fit n_sig across {n_experiments} toys): {n_sig_hi:.4f}")
    print(f"{len(boundary_idx)}/{n_experiments} toys pegged at that boundary")

    return f, boundary_idx[:n_max], n_sig_hi


def plot_toy(f, mll, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg, plots_dir, tag, toy_index):
    data = f.Get(f"genData_{toy_index}")
    fr = f.Get(f"fitResult_{toy_index}")
    fitted_pars = fr.floatParsFinal()
    sig_par = fitted_pars.find("n_sig")
    bkg_par = fitted_pars.find("n_bkg")
    n_sig.setVal(sig_par.getVal())
    n_bkg.setVal(bkg_par.getVal())

    frame = mll.frame(ROOT.RooFit.Title(f"Toy #{toy_index}: data + best fit (n_sig at boundary)"))
    data.plotOn(frame, ROOT.RooFit.Name("data"))
    total_pdf.plotOn(frame, ROOT.RooFit.Name("total"), ROOT.RooFit.LineColor(ROOT.kBlue + 1))
    total_pdf.plotOn(frame, ROOT.RooFit.Name("bkg"),
                     ROOT.RooFit.Components(bkg_pdf.GetName()),
                     ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed + 1))
    total_pdf.plotOn(frame, ROOT.RooFit.Name("sig"),
                     ROOT.RooFit.Components(sig_pdf.GetName()),
                     ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen + 2))

    canv = ROOT.TCanvas(f"canv_boundary_toy_{toy_index}_{tag}", "", 700, 600)
    frame.Draw()

    leg = ROOT.TLegend(0.60, 0.56, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.032)
    leg.AddEntry(frame.findObject("data"), "Toy data", "pe")
    leg.AddEntry(frame.findObject("total"), "S+B fit", "l")
    leg.AddEntry(frame.findObject("bkg"), "Background", "l")
    leg.AddEntry(frame.findObject("sig"), "Signal", "l")
    leg.AddEntry(ROOT.nullptr, f"n_{{sig}} = {sig_par.getVal():.2f} #pm {sig_par.getError():.2f} (at bound)", "")
    leg.AddEntry(ROOT.nullptr, f"n_{{bkg}} = {bkg_par.getVal():.2f} #pm {bkg_par.getError():.2f}", "")
    leg.Draw()

    os.makedirs(plots_dir, exist_ok=True)
    outpath = os.path.join(plots_dir, f"boundary_toy{toy_index}_{tag}.pdf")
    canv.SaveAs(outpath)
    canv.SaveAs(outpath.replace(".pdf", ".png"))
    print(f"  Created {outpath} (+.png) "
         f"[n_sig_fit={sig_par.getVal():.3f}, n_bkg_fit={bkg_par.getVal():.3f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Path to results_*.root from do_1d_mll_toys.py")
    parser.add_argument("--m1", type=int, default=650)
    parser.add_argument("--m2", type=int, default=1)
    parser.add_argument("-n", "--n-toys", type=int, default=3,
                        help="Number of boundary toys to plot")
    parser.add_argument("--outdir", default=None,
                        help="Plots dir; defaults to <results dir>/plots")
    args = parser.parse_args()

    plots_dir = args.outdir or os.path.join(os.path.dirname(args.results), "plots")
    tag = os.path.splitext(os.path.basename(args.results))[0].replace("results_", "")

    f, boundary_idx, n_sig_hi = find_boundary_toys(args.results, args.n_toys)
    if not boundary_idx:
        print("No boundary toys found.")
        return
    print(f"Plotting first {len(boundary_idx)} boundary toys: {boundary_idx}")

    mll, sig_pdf, bkg_pdf, components = build_model_1d(args.m1, args.m2)
    n_sig = ROOT.RooRealVar("n_sig", "n_sig", 0, -10 * abs(n_sig_hi) - 10, n_sig_hi + 1)
    n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", 0, 0, 1000)
    total_pdf = ROOT.RooAddPdf("total_pdf_1d_mll_boundary", "total_pdf_1d_mll_boundary",
                               ROOT.RooArgList(sig_pdf, bkg_pdf),
                               ROOT.RooArgList(n_sig, n_bkg))

    for idx in boundary_idx:
        plot_toy(f, mll, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg, plots_dir, tag, idx)


if __name__ == "__main__":
    main()
