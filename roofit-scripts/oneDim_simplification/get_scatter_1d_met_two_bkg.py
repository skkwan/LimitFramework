#!/usr/bin/env python3
"""
get_scatter_1d_met_two_bkg.py
Sibling of get_scatter_1d_met.py, but the background is the combination of both
MET shapes -- non-peaking-in-m(ll) and peaking-in-m(ll) -- instead of the
non-peaking-only shape:

    bkg_pdf = r * bkgpeaking_met + (1 - r) * bkgnonpeak_met

r (the peaking fraction) is fixed/constant, not fitted; n_sig and n_bkg (the total
background yield) remain the only two floating fit parameters. Reuses toy.py's
get_background_model (unlike do_1d_met_toys.py, which uses toy_noPkgBkg.py's
non-peaking-only version) and do_1d_met_toys.py's run_mcstudy/collect_results, which
are agnostic to how bkg_pdf was built.

Usage:
    python3 get_scatter_1d_met_two_bkg.py
    python3 get_scatter_1d_met_two_bkg.py --n-sig-list 0 5 10 15 20 25 -N 5000 --n-bkg-in 48 --r 0.088
    python3 get_scatter_1d_met_two_bkg.py --force
"""

import ROOT
import cmsstyle as CMS
import os
import sys
import json
import argparse

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))
from do_1d_met_toys import (
    _find_component, _find_component_prefix, run_mcstudy, collect_results, make_example_toy_plot,
    _first_converged_toys,
)
from get_scatter_1d_met import (
    add_pulls, aggregate_point, make_scatter_plot, make_distribution_plot,
)

HBB_ZLL_TOY_DIR = ("/afs/cern.ch/work/s/skkwan/public/zhmet/CMSSW_14_0_21/src/"
                    "2DFit_higgsinos/hbb_zll/toy_experiments")
sys.path.insert(0, HBB_ZLL_TOY_DIR)
import toy as toy_ref  # full model (peaking + non-peaking bkg), unlike toy_noPkgBkg

DATATAG = "one_dim_met_only_two_bkg"

# Plotted x-range for the n_sig distribution plots -- narrower than the fit range
# (n_sig_fit_range below) so the fit itself stays unconstrained over +/-500, while
# toys landing outside +/-100 get clamped into the edge bin as a visible
# underflow/overflow bin instead of being fit over a needlessly wide, mostly-empty axis.
N_SIG_PLOT_RANGE = (-100.0, 100.0)

def n_sig_fit_range(n_sig_in):
    # if n_sig_in == 0:
    #     return (-500.0, 500.0)
    # return (-200 + n_sig_in, 200 + n_sig_in)
    # return (-10 * abs(n_sig_in), 10 * abs(n_sig_in))
    return (-500 + n_sig_in, 500 + n_sig_in)

# ── Model construction ──────────────────────────────────────────────────────────────

def build_model_1d_two_bkg(m1, m2, r):
    """
    Build (met, sig_pdf_met, bkg_pdf_met, components) for one mass point, where
    bkg_pdf_met = r * bkgpeaking_met + (1 - r) * bkgnonpeak_met, with r fixed. Reuses
    toy.py's get_signal_model/get_background_model (same chdir/mll-global pattern as
    do_1d_met_toys.py's build_model_1d). Caller must keep the returned components
    alive for as long as the model is in use.
    """
    prev_cwd = os.getcwd()
    try:
        os.chdir(HBB_ZLL_TOY_DIR)
        mll = ROOT.RooRealVar("m_ll", "m_ll", 60, 120)
        toy_ref.mll = mll
        _, met, sig_components = toy_ref.get_signal_model(m1, m2)
        _, bkg_components, ratio_peaking = toy_ref.get_background_model(met, fix_r=True)
    finally:
        os.chdir(prev_cwd)

    sig_pdf_met = _find_component_prefix(sig_components, "pdf_of_spline")
    bkgpeaking_met = _find_component(bkg_components, "bkgpeaking_met")
    bkgnonpeak_met = _find_component(bkg_components, "bkgnonpeak_met")

    ratio_peaking.setVal(r)
    ratio_peaking.setConstant(True)

    bkg_pdf_met = ROOT.RooAddPdf("bkgtot_met_two_bkg", "bkgtot_met_two_bkg",
                                 ROOT.RooArgList(bkgpeaking_met, bkgnonpeak_met),
                                 ratio_peaking)

    components = sig_components + bkg_components + [bkg_pdf_met]
    return met, sig_pdf_met, bkg_pdf_met, components


# ── Scan across injected n_sig values ────────────────────────────────────────────────

def run_scan(args):
    met, sig_pdf, bkg_pdf, components = build_model_1d_two_bkg(args.m1, args.m2, args.r)

    n_bkg_lo, n_bkg_hi = -100.0, 300.0

    outlier_plot_dir = os.path.join(args.plot_dir, "example_toys_outliers")
    example_plot_dir = os.path.join(args.plot_dir, "example_toys")

    points = []
    for i, n_sig_in in enumerate(args.n_sig_list):
        n_sig_lo, n_sig_hi = n_sig_fit_range(n_sig_in)
        seed = args.seed + i
        mcs, n_sig, n_bkg, total_pdf = run_mcstudy(
            met, sig_pdf, bkg_pdf, n_sig_in, args.n_bkg_in,
            (n_sig_lo, n_sig_hi), (n_bkg_lo, n_bkg_hi),
            args.n_experiments, seed)
        rows = collect_results(mcs, args.n_experiments, n_sig_in, args.n_bkg_in)
        add_pulls(rows, n_sig_in, args.n_bkg_in)

        tag = f"{DATATAG}_nsig{n_sig_in:g}_nbkg{args.n_bkg_in:g}_r{args.r:g}_seed{seed}"

        # Budgets reset per injected n_sig point, so every point gets its own set of
        # outlier/example plots instead of the whole scan sharing one global budget.
        n_outliers_plotted = 0
        for idx, row in enumerate(rows):
            if n_outliers_plotted >= args.n_outlier_examples:
                break
            if row["status"] == 0 and row["n_sig_val"] < args.outlier_n_sig_max:
                make_example_toy_plot(mcs, met, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
                                      outlier_plot_dir, tag, idx)
                n_outliers_plotted += 1

        n_examples_plotted = 0
        for idx in _first_converged_toys(rows, args.n_example_toys):
            make_example_toy_plot(mcs, met, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
                                  example_plot_dir, tag, idx)
            n_examples_plotted += 1

        print(f"n_sig_in={n_sig_in:g}: made {n_outliers_plotted} outlier plot(s) "
             f"(n_sig_val < {args.outlier_n_sig_max:g}) and {n_examples_plotted} "
             f"converged-fit example plot(s)")

        points.append({
            "n_sig_in": n_sig_in,
            "n_sig_lo": n_sig_lo,
            "n_sig_hi": n_sig_hi,
            "n_bkg_lo": n_bkg_lo,
            "n_bkg_hi": n_bkg_hi,
            "seed": seed,
            "toys": rows,
        })

    print(f"Outlier plots in {outlier_plot_dir}, example plots in {example_plot_dir}")

    return {
        "m1": args.m1,
        "m2": args.m2,
        "n_bkg_in": args.n_bkg_in,
        "r": args.r,
        "n_experiments": args.n_experiments,
        "seed": args.seed,
        "points": points,
    }


def load_or_run(args):
    json_path = os.path.join(args.outdir, "results_met_two_bkg.json")

    if os.path.exists(json_path) and not args.force:
        print(f"Found existing {json_path}, loading it (pass --force to re-run the scan).")
        with open(json_path) as f:
            return json.load(f), json_path

    os.makedirs(args.outdir, exist_ok=True)
    result = run_scan(args)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {json_path}")
    return result, json_path


# ── Plotting (same conventions as get_scatter_1d_met.py, legend also reports r) ─────

def make_per_point_plots(point, n_bkg_in, r, n_experiments, mass_point, plot_dir):
    """Best-fit-value and pull distributions for n_sig/n_bkg at one injected point."""
    n_sig_in = point["n_sig_in"]
    tag = f"{DATATAG}_nsig{n_sig_in:g}_nbkg{n_bkg_in:g}_r{r:g}"
    toys = [t for t in point["toys"] if t["status"] == 0]

    for key, label, name, rng, refline_val in [
            ("n_sig_val", "Best-fit n_{sig}", "nsig", N_SIG_PLOT_RANGE, n_sig_in),
            ("n_bkg_val", "Best-fit n_{bkg}", "nbkg", (point["n_bkg_lo"], point["n_bkg_hi"]), n_bkg_in)]:
        vals = [t[key] for t in toys]
        make_distribution_plot(vals, label, f"distribution_{name}_{tag}", plot_dir,
                               mass_point, n_sig_in, n_bkg_in, n_experiments, x_range=rng,
                               refline_val=refline_val, nbins=100)

    for key, label, name in [("n_sig_pull", "Pull(n_{sig})", "nsig"),
                             ("n_bkg_pull", "Pull(n_{bkg})", "nbkg")]:
        vals = [t[key] for t in toys if t[key] is not None]
        make_distribution_plot(vals, label, f"pull_{name}_{tag}", plot_dir,
                               mass_point, n_sig_in, n_bkg_in, n_experiments, fit_gaussian=True)


def make_plots(result, plot_dir):
    mass_point = (result["m1"], result["m2"])
    n_bkg_in = result["n_bkg_in"]
    r = result["r"]
    n_experiments = result["n_experiments"]
    points = result["points"]

    n_sig_in_vals = [p["n_sig_in"] for p in points]

    n_sig_means, n_sig_errs = [], []
    n_bkg_means, n_bkg_errs = [], []
    for p in points:
        mv, me = aggregate_point(p, "n_sig_val")
        n_sig_means.append(mv)
        n_sig_errs.append(me)
        mv, me = aggregate_point(p, "n_bkg_val")
        n_bkg_means.append(mv)
        n_bkg_errs.append(me)

        make_per_point_plots(p, n_bkg_in, r, n_experiments, mass_point, plot_dir)

    make_scatter_plot(
        n_sig_in_vals, n_sig_means, n_sig_errs,
        xlabel="Injected n_{sig}", ylabel="Average best-fit n_{sig}",
        refline=lambda x: x, refline_label="y=x",
        plotname="scatter_1d_met_two_bkg_nsig", plot_dir=plot_dir, mass_point=mass_point,
        n_bkg_in=n_bkg_in, n_experiments=n_experiments)

    make_scatter_plot(
        n_sig_in_vals, n_bkg_means, n_bkg_errs,
        xlabel="Injected n_{sig}", ylabel="Average best-fit n_{bkg}",
        refline=lambda x: n_bkg_in,
        plotname="scatter_1d_met_two_bkg_nbkg", plot_dir=plot_dir, mass_point=mass_point,
        n_bkg_in=n_bkg_in, n_experiments=n_experiments)


# ── Entry point ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scan injected n_sig values for the 1D MET toy fit with a "
                    "two-component background (non-peaking + peaking-in-m(ll), mixed "
                    "with a fixed peaking fraction r), save every toy's best-fit "
                    "n_sig/n_bkg to JSON, and make injected-vs-fitted scatter plots.")
    parser.add_argument("--n-sig-list", nargs="+", type=float, default=[0, 5, 10, 15, 20, 25])
    parser.add_argument("--n-bkg-in", type=float, default=48.0)
    parser.add_argument("--r", type=float, default=0.088,
                        help="Fixed fraction of the total background that is "
                             "peaking-in-m(ll) (not a fit parameter)")
    parser.add_argument("--outlier-n-sig-max", type=float, default=-80.0,
                        help="Make example toy plots (data + best-fit signal/"
                             "background curves) for converged toys whose best-fit "
                             "n_sig falls below this value")
    parser.add_argument("--n-outlier-examples", type=int, default=5,
                        help="Max number of such example toy plots to make, per "
                             "injected n_sig point")
    parser.add_argument("--n-example-toys", type=int, default=5,
                        help="Make example toy plots (data + best-fit signal/"
                             "background curves) for the first N converged toys, "
                             "per injected n_sig point")
    parser.add_argument("-N", "--n-experiments", type=int, default=5000)
    parser.add_argument("--m1", type=int, default=650)
    parser.add_argument("--m2", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", default="/eos/cms/store/group/phys_susy/skkwan/toys/1d_met_two_bkg/scatter_plot_results")
    parser.add_argument("--plot-dir", default="/eos/user/s/skkwan/www/higgsino/studies/mll-MET-fit-2D/toys/one_dimension_two_bkg")
    parser.add_argument("--force", action="store_true",
                        help="Re-run the scan even if results_met_two_bkg.json already exists")
    args = parser.parse_args()

    result, json_path = load_or_run(args)

    print(f"\nMaking scatter plots from {json_path} ...")
    make_plots(result, args.plot_dir)

    print(f"\n Check https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/toys/one_dimension_two_bkg/ for outputs")
    print(f"\n Check https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/toys/one_dimension_two_bkg/example_toys_outliers/ for outlier plots")
    print(f"\n Check https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/toys/one_dimension_two_bkg/example_toys/ for random examples that converged")

if __name__ == "__main__":
    main()
