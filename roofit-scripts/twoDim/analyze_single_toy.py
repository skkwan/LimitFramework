#!/usr/bin/env python3
"""
analyze_single_toy.py
Re-open one toy dataset saved by get_scatter_2d_both_bkg_components.py (the .root files it
writes to <outdir>/example_outlier_toys/ for every plotted outlier), rebuild the identical
2D (m(ll), MET) model, and re-run the same RooFit fit on it.

Reports, for the refit: the fit status, the three best-fit yields, the extended NLL at the
minimum, and the chi^2/ndf of the m(ll) and MET marginals -- then plots it with
helper.make_example_toy_plot_both_bkg's core routine.

Afterwards it prompts for your own n_sig / n_bkg_peaking / n_bkg_nonpeak values, and for
each set reports the NLL (and dNLL relative to the fitted minimum), the two chi^2/ndf
values, and writes the corresponding plot. That is the point of the script: to check by
hand whether a pathological best fit (e.g. a large negative n_sig) really is a better
likelihood than the values you would have expected.

Usage:
    python3 analyze_single_toy.py /eos/cms/store/group/phys_susy/skkwan/toys/2d_mll_met_both_bkg/scatter_plot_results/example_outlier_toys/outlier_toy70_two_dim_both_bkg_nsig0_nbkgpeak13_nbkgnonpeak132_seed1.root
    python3 analyze_single_toy.py <toy>.root --yields 0 4 44       
    python3 analyze_single_toy.py <toy>.root --no-refit --yields 0 4 44
"""

import ROOT
import os
import argparse

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

from helper import (
    build_model_2d_both_bkg, load_toy_dataset, make_nll, plot_toy_both_bkg,
)

DEFAULT_PLOT_DIR = ("/eos/user/s/skkwan/www/higgsino/studies/mll-MET-fit-2D/two_dimensional_fitTo_handling/both_bkg/refit")


def read_meta(path):
    """Peek at the stored meta JSON before the model is built -- the mass point and yield
    ranges live in there, and they are needed to build the model in the first place."""
    import json
    f = ROOT.TFile.Open(path, "READ")
    if not f or f.IsZombie():
        raise IOError(f"Could not open toy file {path}")
    obj = f.Get("meta")
    if not obj:
        raise IOError(f"No 'meta' TNamed in {path} -- not a saved toy file?")
    meta = json.loads(obj.GetTitle())
    f.Close()
    return meta


def build_fit_model(mll, met, sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf, meta):
    """The same RooAddPdf of three independently-floating yields that
    helper.run_mcstudy_2d_both_bkg builds, over the ranges this toy was originally fit
    with. Kept in the same argument order as that function so the two stay in sync."""
    n_sig = ROOT.RooRealVar("n_sig", "n_sig", meta["n_sig_gen"],
                            meta["n_sig_lo"], meta["n_sig_hi"])
    n_bkg_peak = ROOT.RooRealVar("n_bkg_peak", "n_bkg_peak", meta["n_bkg_peaking_gen"],
                                 meta["n_bkg_lo"], meta["n_bkg_hi"])
    n_bkg_nonpeak = ROOT.RooRealVar("n_bkg_nonpeak", "n_bkg_nonpeak", meta["n_bkg_nonpeak_gen"],
                                    meta["n_bkg_lo"], meta["n_bkg_hi"])
    total_pdf = ROOT.RooAddPdf("total_pdf_2d_mll_met", "total_pdf_2d_mll_met",
                               ROOT.RooArgList(sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf),
                               ROOT.RooArgList(n_sig, n_bkg_peak, n_bkg_nonpeak))
    return n_sig, n_bkg_peak, n_bkg_nonpeak, total_pdf


def report(header, n_sig, n_bkg_peak, n_bkg_nonpeak, nll_val, chi2, nll_ref=None, errs=None):
    errs = errs or {}

    def fmt(var, key):
        err = errs.get(key)
        return f"{var.getVal():10.3f}" + (f" +/- {err:7.3f}" if err is not None else "")

    print(f"\n{header}")
    print(f"  n_sig          = {fmt(n_sig, 'n_sig')}")
    print(f"  n_bkg_peaking  = {fmt(n_bkg_peak, 'n_bkg_peak')}")
    print(f"  n_bkg_nonpeak  = {fmt(n_bkg_nonpeak, 'n_bkg_nonpeak')}")
    print(f"  total          = {n_sig.getVal() + n_bkg_peak.getVal() + n_bkg_nonpeak.getVal():10.3f}")
    print(f"  NLL            = {nll_val:12.5f}"
          + (f"   (dNLL vs best fit = {nll_val - nll_ref:+.5f})" if nll_ref is not None else ""))
    for obs_name in ("mll", "met"):
        c = chi2[obs_name]
        print(f"  chi2/ndf ({obs_name:3s}) = {c['chi2']:8.2f} / {c['ndf']:3d} = {c['chi2_ndf']:.3f}")


def evaluate(values, data, mll, met, model, nll, gen_counts, plot_dir, tag, label,
             title, nll_ref):
    """Pin the three yields at `values`, evaluate the shared NLL there, plot, and report.
    The yields are held constant for the duration so the plot's chi2 ndf reflects that
    nothing was fitted, then released again."""
    n_sig, n_bkg_peak, n_bkg_nonpeak, total_pdf, sig_pdf, bkg_peak_pdf, bkg_nonpeak_pdf = model
    varlist = (n_sig, n_bkg_peak, n_bkg_nonpeak)

    for var, val in zip(varlist, values):
        if not (var.getMin() <= val <= var.getMax()):
            print(f"  [warn] {var.GetName()} = {val:g} is outside the fit range "
                  f"[{var.getMin():g}, {var.getMax():g}]; widening the range to accept it.")
            var.setRange(min(var.getMin(), val), max(var.getMax(), val))
        var.setVal(val)
        var.setConstant(True)

    try:
        chi2 = plot_toy_both_bkg(data, mll, met, total_pdf, sig_pdf, bkg_peak_pdf, bkg_nonpeak_pdf,
                                 n_sig, n_bkg_peak, n_bkg_nonpeak,
                                 yield_errs=None, gen_counts=gen_counts,
                                 plot_dir=plot_dir, tag=tag, label=label, title=title)
        report("User-specified yields:", n_sig, n_bkg_peak, n_bkg_nonpeak,
               nll.getVal(), chi2, nll_ref=nll_ref)
    finally:
        for var in varlist:
            var.setConstant(False)


def parse_yields(text):
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("expected 3 numbers: n_sig n_bkg_peaking n_bkg_nonpeak")
    return [float(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(
        description="Re-fit one saved toy dataset, then evaluate the NLL and marginal "
                    "chi^2 at your own n_sig / n_bkg_peaking / n_bkg_nonpeak values.")
    parser.add_argument("toy_file", help="A .root file from <outdir>/example_outlier_toys/")
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR,
                        help="Plots go to <plot-dir>/single_toy/")
    parser.add_argument("--yields", nargs=3, type=float, metavar=("NSIG", "NPEAK", "NNONPEAK"),
                        help="Evaluate this one set of yields and exit, instead of prompting")
    parser.add_argument("--no-refit", action="store_true",
                        help="Skip the refit; take the best-fit values stored in the file "
                             "as the NLL reference point instead")
    args = parser.parse_args()

    meta = read_meta(args.toy_file)
    print(f"Toy {args.toy_file}")
    print(f"  mass point ({meta['m1']}, {meta['m2']}), tag {meta.get('tag')}, "
          f"toy index {meta.get('toy_index')}, seed {meta.get('seed')}")
    print(f"  generated: n_sig = {meta['n_sig_gen']}, n_bkg_peaking = {meta['n_bkg_peaking_gen']}, "
          f"n_bkg_nonpeak = {meta['n_bkg_nonpeak_gen']}")
    print(f"  original best fit: n_sig = {meta['n_sig_val']:.3f}, "
          f"n_bkg_peak = {meta['n_bkg_peak_val']:.3f}, "
          f"n_bkg_nonpeak = {meta['n_bkg_nonpeak_val']:.3f} (status {meta['status']})")

    mll, met, sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf, components = (
        build_model_2d_both_bkg(meta["m1"], meta["m2"]))
    data, _ = load_toy_dataset(args.toy_file, mll, met)
    print(f"  {data.numEntries()} events read back")

    n_sig, n_bkg_peak, n_bkg_nonpeak, total_pdf = build_fit_model(
        mll, met, sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf, meta)
    model = (n_sig, n_bkg_peak, n_bkg_nonpeak, total_pdf,
             sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf)

    # One NLL object, re-evaluated at every point below -- absolute NLL values from
    # different objects are not comparable.
    nll = make_nll(total_pdf, data)

    gen_counts = (meta["n_sig_gen"], meta["n_bkg_peaking_gen"], meta["n_bkg_nonpeak_gen"])
    plot_dir = os.path.join(args.plot_dir, "single_toy")
    base_tag = meta.get("tag", "toy")
    toy_index = meta.get("toy_index", 0)

    if args.no_refit:
        n_sig.setVal(meta["n_sig_val"])
        n_bkg_peak.setVal(meta["n_bkg_peak_val"])
        n_bkg_nonpeak.setVal(meta["n_bkg_nonpeak_val"])
        nll_ref = nll.getVal()
        print(f"\nStored best-fit values, NLL = {nll_ref:.5f} (no refit)")
    else:
        fr = total_pdf.fitTo(data, ROOT.RooFit.Extended(), ROOT.RooFit.Save(True),
                             ROOT.RooFit.PrintLevel(-1))
        nll_ref = nll.getVal()
        pars = fr.floatParsFinal()
        errs = {name: pars.find(name).getError()
                for name in ("n_sig", "n_bkg_peak", "n_bkg_nonpeak")}
        chi2 = plot_toy_both_bkg(data, mll, met, total_pdf, sig_pdf, bkg_peaking_pdf,
                                 bkg_nonpeaking_pdf, n_sig, n_bkg_peak, n_bkg_nonpeak,
                                 yield_errs=errs, gen_counts=gen_counts,
                                 plot_dir=plot_dir, tag=f"refit_{base_tag}",
                                 label=f"single_toy{toy_index}",
                                 title=f"Toy #{toy_index}: data + refit")
        report(f"Refit (status {fr.status()}, covQual {fr.covQual()}):",
               n_sig, n_bkg_peak, n_bkg_nonpeak, nll_ref, chi2, errs=errs)

        # The refit starts from the generated values rather than wherever the scan's
        # previous toy left the parameters, so agreement is only ever to within Minuit's
        # tolerance -- compare against the fitted uncertainty, not to the last decimal.
        drift = max(abs(var.getVal() - meta[key]) / max(errs[var.GetName()], 1e-6)
                    for var, key in ((n_sig, "n_sig_val"),
                                     (n_bkg_peak, "n_bkg_peak_val"),
                                     (n_bkg_nonpeak, "n_bkg_nonpeak_val")))
        if drift > 0.05:
            print(f"  [warn] refit differs from the stored best fit by up to {drift:.2f}"
                  f" sigma -- the reloaded toy is not reproducing the original fit, so "
                  f"treat everything below with suspicion.")

    def run_one(values):
        label_tag = (f"user_nsig{values[0]:g}_npeak{values[1]:g}_nnonpeak{values[2]:g}"
                     .replace(".", "p").replace("-", "m"))
        evaluate(values, data, mll, met, model, nll, gen_counts, plot_dir,
                 tag=f"{label_tag}_{base_tag}", label=f"single_toy{toy_index}",
                 title=f"Toy #{toy_index}: data + user-specified yields", nll_ref=nll_ref)

    if args.yields is not None:
        run_one(args.yields)
    else:
        print("\nEnter your own yields as three numbers: n_sig n_bkg_peaking n_bkg_nonpeak")
        print("(blank line or 'q' to quit)")
        while True:
            try:
                line = input("yields > ").strip()
            except EOFError:
                print()
                break
            if line == "" or line.lower() in ("q", "quit", "exit"):
                break
            try:
                values = parse_yields(line)
            except ValueError as e:
                print(f"  {e}")
                continue
            run_one(values)

    website_dir = plot_dir.replace("/eos/user/s/skkwan/www/", "https://skkwan.web.cern.ch/")
    print(f"\nPlots in {website_dir}")


if __name__ == "__main__":
    main()
