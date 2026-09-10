#!/usr/bin/env python3
"""
get_scatter_2d_only_nonpeaking_bkg.py
2D (m(ll), MET) analogue of oneDim_simplification/get_scatter_1d_met.py: wraps
do_2d_toys_nonpeaking_bkg_only.py's model-building/toy-running functions to scan
several injected n_sig values, save every toy's best-fit n_sig/n_bkg (value + error)
to a single JSON file, make CMS-styled (cmsstyle "Private work") scatter plots of
injected vs. recovered yield, and (like get_scatter_1d_met_two_bkg.py) dump per-toy
example fit plots to example_toys/ and example_toys_outliers/ (one plot per observable,
m(ll) and MET, per toy). Same signal + non-peaking-only-background model as
get_scatter_1d_met.py, but fit over both (m(ll), MET) observables instead of the MET
marginal only.

Usage:
    python3 get_scatter_2d_only_nonpeaking_bkg.py
    python3 get_scatter_2d_only_nonpeaking_bkg.py --n-sig-list 0 5 10 15 20 25 -N 5000 --n-bkg-in 48
    python3 get_scatter_2d_only_nonpeaking_bkg.py --force
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
from do_2d_toys_nonpeaking_bkg_only import (
    build_model_2d, run_mcstudy_2d, collect_results, make_example_toy_plot,
    _first_converged_toys,
)

ONE_DIM_DIR = os.path.join(os.path.dirname(__file__), "..", "oneDim_simplification")
sys.path.insert(0, ONE_DIM_DIR)
from get_scatter_1d_met import (
    add_pulls, aggregate_point, make_scatter_plot, make_distribution_plot,
)

DATATAG = "two_dim_peakingBkgOnly"

# Plotted x-range for the n_sig distribution plots -- narrower than the fit range
# (n_sig_fit_range below) so the fit itself stays unconstrained over +/-500, while
# toys landing outside +/-100 get clamped into the edge bin as a visible
# underflow/overflow bin instead of being fit over a needlessly wide, mostly-empty axis.
N_SIG_PLOT_RANGE = (-100.0, 100.0)


# ── n_sig fit range per injected point ───────────────────────────────────────────────

def n_sig_fit_range(n_sig_in):
    return (-500 + n_sig_in, 500 + n_sig_in)


# ── Scan across injected n_sig values ────────────────────────────────────────────────

def run_scan(args):
    mll, met, sig_pdf, bkg_pdf, components = build_model_2d(args.m1, args.m2)

    n_bkg_lo, n_bkg_hi = 0.0, 100.0

    outlier_plot_dir = os.path.join(args.plot_dir, "example_toys_outliers")
    example_plot_dir = os.path.join(args.plot_dir, "example_toys")

    points = []
    for i, n_sig_in in enumerate(args.n_sig_list):
        n_sig_lo, n_sig_hi = n_sig_fit_range(n_sig_in)
        seed = args.seed + i
        mcs, n_sig, n_bkg, total_pdf = run_mcstudy_2d(
            mll, met, sig_pdf, bkg_pdf, n_sig_in, args.n_bkg_in,
            (n_sig_lo, n_sig_hi), (n_bkg_lo, n_bkg_hi),
            args.n_experiments, seed)
        rows = collect_results(mcs, args.n_experiments, n_sig_in, args.n_bkg_in)
        add_pulls(rows, n_sig_in, args.n_bkg_in)

        tag = f"{DATATAG}_nsig{n_sig_in:g}_nbkg{args.n_bkg_in:g}_seed{seed}"

        # Budgets reset per injected n_sig point, so every point gets its own set of
        # outlier/example plots instead of the whole scan sharing one global budget.
        n_outliers_plotted = 0
        for idx, row in enumerate(rows):
            if n_outliers_plotted >= args.n_outlier_examples:
                break
            if row["status"] == 0 and row["n_sig_val"] < args.outlier_n_sig_max:
                make_example_toy_plot(mcs, mll, met, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
                                      outlier_plot_dir, tag, idx)
                n_outliers_plotted += 1

        n_examples_plotted = 0
        for idx in _first_converged_toys(rows, args.n_example_toys):
            make_example_toy_plot(mcs, mll, met, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
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
        "n_experiments": args.n_experiments,
        "seed": args.seed,
        "points": points,
    }


def load_or_run(args):
    json_path = os.path.join(args.outdir, "results_2d_mll_met.json")

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


# ── Plotting (same conventions as get_scatter_1d_met.py) ────────────────────────────

def make_per_point_plots(point, n_bkg_in, n_experiments, mass_point, plot_dir):
    """Best-fit-value and pull distributions for n_sig/n_bkg at one injected point.
    Returns {val_key: gaussian_fit_result} for the best-fit-value distributions, so
    the caller can use the fit's sigma as the scatter-plot error bar for that point."""
    n_sig_in = point["n_sig_in"]
    tag = f"{DATATAG}_nsig{n_sig_in:g}_nbkg{n_bkg_in:g}"
    toys = [t for t in point["toys"] if t["status"] == 0]

    fit_results = {}
    for key, label, name, rng, refline_val in [
            ("n_sig_val", "Best-fit n_{sig}", "nsig", N_SIG_PLOT_RANGE, n_sig_in),
            ("n_bkg_val", "Best-fit n_{bkg}", "nbkg", (point["n_bkg_lo"], point["n_bkg_hi"]), n_bkg_in)]:
        vals = [t[key] for t in toys]
        fit_results[key] = make_distribution_plot(vals, label, f"distribution_{name}_{tag}", plot_dir,
                               mass_point, n_sig_in, n_bkg_in, n_experiments, x_range=rng,
                               refline_val=refline_val, nbins=100, fit_gaussian=True)

    for key, label, name in [("n_sig_pull", "Pull(n_{sig})", "nsig"),
                             ("n_bkg_pull", "Pull(n_{bkg})", "nbkg")]:
        vals = [t[key] for t in toys if t[key] is not None]
        make_distribution_plot(vals, label, f"pull_{name}_{tag}", plot_dir,
                               mass_point, n_sig_in, n_bkg_in, n_experiments, fit_gaussian=True)

    return fit_results


def make_plots(result, plot_dir):
    mass_point = (result["m1"], result["m2"])
    n_bkg_in = result["n_bkg_in"]
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

        make_per_point_plots(p, n_bkg_in, n_experiments, mass_point, plot_dir)

    make_scatter_plot(
        n_sig_in_vals, n_sig_means, n_sig_errs,
        xlabel="Injected n_{sig}", ylabel="Average best-fit n_{sig}",
        refline=lambda x: x, refline_label="y=x",
        plotname="scatter_2d_mll_met_nonPeakBkgOnly_nsig", plot_dir=plot_dir, mass_point=mass_point,
        n_bkg_in=n_bkg_in, n_experiments=n_experiments, label_points=True)

    make_scatter_plot(
        n_sig_in_vals, n_bkg_means, n_bkg_errs,
        xlabel="Injected n_{sig}", ylabel="Average best-fit n_{bkg}",
        refline=lambda x: n_bkg_in,
        plotname="scatter_2d_mll_met_nonPeakBkgOnly_nbkg", plot_dir=plot_dir, mass_point=mass_point,
        n_bkg_in=n_bkg_in, n_experiments=n_experiments, label_points=True)


# ── Entry point ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scan injected n_sig values for the 2D (m(ll), MET) toy fit "
                    "(full signal product pdf, non-peaking-only background product "
                    "pdf), save every toy's best-fit n_sig/n_bkg to JSON, and make "
                    "injected-vs-fitted scatter plots.")
    parser.add_argument("--n-sig-list", nargs="+", type=float, default=[0, 5, 10, 15, 20, 25])
    parser.add_argument("--n-bkg-in", type=float, default=48.0)
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
    parser.add_argument("--outdir", default="/eos/cms/store/group/phys_susy/skkwan/toys/2d_mll_met/scatter_plot_results")
    parser.add_argument("--plot-dir", default="/eos/user/s/skkwan/www/higgsino/studies/mll-MET-fit-2D/toys/two_dimension_only_nonpeaking_bkg")
    parser.add_argument("--force", action="store_true",
                        help="Re-run the scan even if results_2d_mll_met.json already exists")
    args = parser.parse_args()

    result, json_path = load_or_run(args)

    print(f"\nMaking scatter plots from {json_path} ...")
    make_plots(result, args.plot_dir)

    print(f"\n Check https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/toys/two_dimension_only_nonpeaking_bkg/ for outputs")
    print(f"\n Check https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/toys/two_dimension_only_nonpeaking_bkg/example_toys_outliers/ for outlier plots")
    print(f"\n Check https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/toys/two_dimension_only_nonpeaking_bkg/example_toys/ for random examples that converged")

if __name__ == "__main__":
    main()
