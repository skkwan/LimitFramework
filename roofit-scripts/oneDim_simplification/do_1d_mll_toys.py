#!/usr/bin/env python3
"""
do_1d_mll_toys.py
Generates and fits 1D m(ll)-only toy datasets from the signal + non-peaking-background
model, reusing the same fitted shapes as do_toys.py / toy_noPkgBkg.py's
get_signal_model / get_background_model, but keeping only the m(ll) marginal pdfs
(sig_dcb_mll, bkgnonpeak_mll) instead of the full (m(ll), MET) product pdfs, and fitting
with a bare RooMCStudy (like toy_noPkgBkg.py's own __main__ block) rather than through
real Combine commands -- this is a quick closure/recovery check, not a limit-setting
study, so there is no rMin/rMax POI machinery and no AsymptoticLimits step.

Model: sig_pdf = m(ll) Crystal Ball (fitted signal shape at one mass point)
       bkg_pdf = m(ll) Exponential (non-peaking-only background shape)
       total_pdf = n_sig * sig_pdf + n_bkg * bkg_pdf   (both yields floating)

Default: inject n_sig = 0 (background-only), n_bkg = 48 (config.BKG_NORM['run3']),
generate + fit 100 toys, and report how well n_sig and n_bkg are recovered (mean,
std, and pulls (fit - truth) / fit_error across the toys).

Usage:
    python3 do_1d_mll_toys.py
    python3 do_1d_mll_toys.py -N 500 --n-sig-in 0 --n-bkg-in 48
    python3 do_1d_mll_toys.py --m1 650 --m2 1 --seed 42
"""

import ROOT
import os
import sys
import argparse

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))

# Only the model builders are imported from the hbb_zll study area, same as do_toys.py.
HBB_ZLL_TOY_DIR = ("/afs/cern.ch/work/s/skkwan/public/zhmet/CMSSW_14_0_21/src/"
                    "2DFit_higgsinos/hbb_zll/toy_experiments")
sys.path.insert(0, HBB_ZLL_TOY_DIR)
import toy_noPkgBkg as toy_ref


# ── Model construction ──────────────────────────────────────────────────────────────

def _find_component(components, name):
    """Pull one named RooFit object out of a get_signal_model/get_background_model
    components list (both already build the m(ll)-only marginal pdf internally --
    sig_dcb_mll / bkgnonpeak_mll -- before taking the product with the MET pdf, and
    list it among their returned components)."""
    for c in components:
        if hasattr(c, "GetName") and c.GetName() == name:
            return c
    raise RuntimeError(f"component '{name}' not found among {[c.GetName() for c in components if hasattr(c, 'GetName')]}")


def build_model_1d(m1, m2):
    """
    Build (mll, sig_pdf_mll, bkg_pdf_mll, components) for one mass point: the m(ll)-only
    Crystal Ball signal shape and Exponential non-peaking-background shape, reusing
    toy_noPkgBkg.py's fitted shapes (same setup/chdir pattern as do_toys.py's
    build_model -- get_background_model needs a met var, even though it's discarded
    here, and get_signal_model/get_background_model read paths relative to
    hbb_zll/toy_experiments/ and a bare module-level `mll` global).
    Caller must keep the returned components alive for as long as the model is in use.
    """
    prev_cwd = os.getcwd()
    try:
        os.chdir(HBB_ZLL_TOY_DIR)
        mll = ROOT.RooRealVar("m_ll", "m_ll", 60, 120)
        toy_ref.mll = mll
        _, met, sig_components = toy_ref.get_signal_model(m1, m2)
        _, bkg_components = toy_ref.get_background_model(met)
    finally:
        os.chdir(prev_cwd)

    sig_pdf_mll = _find_component(sig_components, "sig_dcb_mll")
    bkg_pdf_mll = _find_component(bkg_components, "bkgnonpeak_mll")
    return mll, sig_pdf_mll, bkg_pdf_mll, sig_components + bkg_components


# ── Toy generation + fitting ─────────────────────────────────────────────────────────

def run_mcstudy(mll, sig_pdf, bkg_pdf, n_sig_in, n_bkg_in, n_sig_range, n_bkg_range,
                n_experiments, seed, binned=False, bins=60):
    n_sig = ROOT.RooRealVar("n_sig", "n_sig", n_sig_in, *n_sig_range)
    n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", n_bkg_in, *n_bkg_range)
    total_pdf = ROOT.RooAddPdf("total_pdf_1d_mll", "total_pdf_1d_mll",
                               ROOT.RooArgList(sig_pdf, bkg_pdf),
                               ROOT.RooArgList(n_sig, n_bkg))

    if binned:
        mll.setBins(bins)

    ROOT.RooRandom.randomGenerator().SetSeed(seed)
    mcs = ROOT.RooMCStudy(
        total_pdf,
        ROOT.RooArgSet(mll),
        ROOT.RooFit.Extended(),
        ROOT.RooFit.Silence(),
        ROOT.RooFit.Binned(binned),
        ROOT.RooFit.FitOptions(ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1)),
    )
    print(f"Running {n_experiments} toy experiments "
         f"(n_sig_in={n_sig_in}, n_bkg_in={n_bkg_in}, seed={seed}, "
         f"binned={binned}{f', bins={bins}' if binned else ''})...", flush=True)
    mcs.generateAndFit(n_experiments, 0, True)
    print("Done.", flush=True)
    return mcs, n_sig, n_bkg, total_pdf


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


def summarize(rows, n_sig_in, n_bkg_in):
    import statistics as stats

    ok = [r for r in rows if r['status'] == 0]
    print(f"\nConverged fits: {len(ok)}/{len(rows)}")
    if not ok:
        print("!! No converged fits -- nothing to summarize.")
        return

    def report(label, vals_key, err_key, truth):
        vals  = [r[vals_key] for r in ok]
        pulls = [(r[vals_key] - truth) / r[err_key] for r in ok if r[err_key] > 0]
        mean_val = stats.mean(vals)
        std_val  = stats.pstdev(vals) if len(vals) > 1 else 0.0
        mean_pull = stats.mean(pulls) if pulls else float('nan')
        std_pull  = stats.pstdev(pulls) if len(pulls) > 1 else float('nan')
        print(f"  {label}: truth={truth:g}  "
             f"recovered mean={mean_val:.3f} +/- {std_val:.3f} (spread over toys)  "
             f"pull mean={mean_pull:.3f} std={std_pull:.3f}")

    print(f"\n=== Recovery summary (converged toys only) ===")
    report("n_sig", 'n_sig_val', 'n_sig_err', n_sig_in)
    report("n_bkg", 'n_bkg_val', 'n_bkg_err', n_bkg_in)


# ── Plots (pull + recovered-value distributions, via RooMCStudy's own plotters) ─────

def make_summary_plots(mcs, n_sig, n_bkg, plots_dir, tag):
    os.makedirs(plots_dir, exist_ok=True)
    for var, label in [(n_sig, "n_sig"), (n_bkg, "n_bkg")]:
        for kind, plotter in [("param", mcs.plotParam), ("pull", mcs.plotPull)]:
            if kind == "pull":
                frame = plotter(var, ROOT.RooFit.Bins(40), ROOT.RooFit.FitGauss(True))
            else:
                frame = plotter(var, ROOT.RooFit.Bins(40))
            canv = ROOT.TCanvas(f"canv_{kind}_{label}_{tag}", "", 700, 600)
            frame.Draw()
            outpath = os.path.join(plots_dir, f"{kind}_{label}_{tag}.pdf")
            canv.SaveAs(outpath)
            canv.SaveAs(outpath.replace(".pdf", ".png"))
            print(f"  Created {outpath} (+.png)")


def make_example_toy_plot(mcs, mll, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
                          plots_dir, tag, toy_index):
    """Plot one generated toy dataset with the total/signal/background curves
    overlaid at that toy's own best-fit yields (not the injected truth)."""
    os.makedirs(plots_dir, exist_ok=True)

    data = mcs.genData(toy_index)
    fr = mcs.fitResult(toy_index)
    fitted_pars = fr.floatParsFinal()
    sig_par = fitted_pars.find("n_sig")
    bkg_par = fitted_pars.find("n_bkg")
    n_sig.setVal(sig_par.getVal())
    n_bkg.setVal(bkg_par.getVal())

    frame = mll.frame(ROOT.RooFit.Title(f"Toy #{toy_index}: data + best fit"))
    data.plotOn(frame, ROOT.RooFit.Name("data"))
    total_pdf.plotOn(frame, ROOT.RooFit.Name("total"), ROOT.RooFit.LineColor(ROOT.kBlue + 1))
    total_pdf.plotOn(frame, ROOT.RooFit.Name("bkg"),
                     ROOT.RooFit.Components(bkg_pdf.GetName()),
                     ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed + 1))
    total_pdf.plotOn(frame, ROOT.RooFit.Name("sig"),
                     ROOT.RooFit.Components(sig_pdf.GetName()),
                     ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen + 2))

    canv = ROOT.TCanvas(f"canv_example_toy_{tag}", "", 700, 600)
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

    outpath = os.path.join(plots_dir, f"example_toy{toy_index}_{tag}.pdf")
    canv.SaveAs(outpath)
    canv.SaveAs(outpath.replace(".pdf", ".png"))
    print(f"  Created {outpath} (+.png) "
         f"[n_sig_fit={n_sig.getVal():.3f}, n_bkg_fit={n_bkg.getVal():.3f}]")


def _first_converged_toys(rows, n):
    return [i for i, r in enumerate(rows) if r['status'] == 0][:n]


# ── Entry point ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate and fit 1D m(ll)-only toy datasets from the signal + "
                    "non-peaking-background-only model via RooMCStudy, reusing "
                    "toy_noPkgBkg.py's fitted shapes, and report n_sig/n_bkg recovery.")
    parser.add_argument("--m1", type=int, default=650)
    parser.add_argument("--m2", type=int, default=1)
    parser.add_argument("--n-sig-in", type=float, default=0.0,
                        help="Injected signal yield")
    parser.add_argument("--n-bkg-in", type=float, default=48.0,
                        help="Injected non-peaking background yield")
    parser.add_argument("--n-sig-lo", type=float, default=-10.0)
    parser.add_argument("--n-sig-hi", type=float, default=10.0)
    parser.add_argument("--n-bkg-lo", type=float, default=0.0)
    parser.add_argument("--n-bkg-hi", type=float, default=100.0)
    parser.add_argument("-N", "--n-experiments", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--binned", action="store_true",
                        help="Generate and fit binned m(ll) data instead of unbinned")
    parser.add_argument("--bins", type=int, default=60,
                        help="Number of m(ll) bins to use when --binned is set")
    parser.add_argument("--outdir", default="/eos/cms/store/group/phys_susy/skkwan/toys/1d_mll")
    args = parser.parse_args()

    mll, sig_pdf, bkg_pdf, components = build_model_1d(args.m1, args.m2)

    tag = f"{args.m1}_{args.m2}_nsig{args.n_sig_in:g}_nbkg{args.n_bkg_in:g}_N{args.n_experiments}_seed{args.seed}"
    if args.binned:
        tag += f"_binned{args.bins}"
    print(f"=== 1D m(ll) toys: m1={args.m1} m2={args.m2} "
         f"n_sig_in={args.n_sig_in} n_bkg_in={args.n_bkg_in} N={args.n_experiments} ===", flush=True)

    mcs, n_sig, n_bkg, total_pdf = run_mcstudy(
        mll, sig_pdf, bkg_pdf, args.n_sig_in, args.n_bkg_in,
        (args.n_sig_lo, args.n_sig_hi), (args.n_bkg_lo, args.n_bkg_hi),
        args.n_experiments, args.seed, binned=args.binned, bins=args.bins)

    rows = collect_results(mcs, args.n_experiments, args.n_sig_in, args.n_bkg_in)
    summarize(rows, args.n_sig_in, args.n_bkg_in)

    os.makedirs(args.outdir, exist_ok=True)
    results_path = os.path.join(args.outdir, f"results_{tag}.root")
    f_out = ROOT.TFile(results_path, "RECREATE")
    for i in range(args.n_experiments):
        mcs.fitResult(i).Write(f"fitResult_{i}")
        mcs.genData(i).Write(f"genData_{i}")
    f_out.Close()
    print(f"\nSaved {args.n_experiments} fit results + generated datasets to {results_path}")

    plots_dir = os.path.join(args.outdir, "plots")
    make_summary_plots(mcs, n_sig, n_bkg, plots_dir, tag)

    for example_idx in _first_converged_toys(rows, 5):
        make_example_toy_plot(mcs, mll, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
                              plots_dir, tag, example_idx)


if __name__ == "__main__":
    main()
