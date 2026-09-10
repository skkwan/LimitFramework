#!/usr/bin/env python3
"""
get_scatter_2d_both_bkg_components.py
2D (m(ll), MET) analogue of oneDim_simplification/get_scatter_1d_met_two_bkg.py and
get_scatter_1d_met_two_bkg_fit_r.py, merged into a single script with a --floatR
switch instead of two sibling files. Wraps do_2d_toys_both_bkg.py's
model-building/toy-running functions (full 2D signal pdf + BOTH background
components, peaking-in-m(ll) and non-peaking-in-m(ll), mixed by r) to scan several
injected n_sig values, save every toy's best-fit n_sig/n_bkg (and r, iff --floatR)
to a single JSON file, make CMS-styled (cmsstyle "Private work") scatter plots of
injected vs. recovered yield, and dump per-toy example fit plots to example_toys/
and example_toys_outliers/ (one plot per observable, m(ll) and MET, per toy).

Default (no --floatR): r is generated at --r and held constant in the fit -- only
n_sig and n_bkg float, and only their distributions/scatter plots are made.

--floatR: r is generated at --r but left floating in the fit (fit range --r-lo to
--r-hi) alongside n_sig and n_bkg. r's distribution/scatter plot is made too, and
every output (JSON, plot files) is tagged with a "_floatR" suffix.

Usage:
    python3 get_scatter_2d_both_bkg_components.py
    python3 get_scatter_2d_both_bkg_components.py --floatR
    python3 get_scatter_2d_both_bkg_components.py --n-sig-list 0 5 10 15 20 25 -N 5000 --n-bkg-in 145 --r 0.088 
    python3 get_scatter_2d_both_bkg_components.py --n-sig-list 0 5 10 15 20 25 -N 5000 --n-bkg-in 145 --r 0.088 --floatR
    python3 get_scatter_2d_both_bkg_components.py --floatR --force
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
from do_2d_toys_both_bkg import (
    build_model_2d_both_bkg, collect_results_2d, add_pulls_2d, make_example_toy_plot,
    _first_converged_toys,
)
from do_2d_toys_nonpeaking_bkg_only import run_mcstudy_2d

ONE_DIM_DIR = os.path.join(os.path.dirname(__file__), "..", "oneDim_simplification")
sys.path.insert(0, ONE_DIM_DIR)
from get_scatter_1d_met import aggregate_point, make_scatter_plot, make_distribution_plot

DATATAG = "two_dim_both_bkg"

# Plotted x-range for the n_sig distribution plots -- narrower than the fit range
# (n_sig_fit_range below) so the fit itself stays unconstrained over +/-500, while
# toys landing outside +/-100 get clamped into the edge bin as a visible
# underflow/overflow bin instead of being fit over a needlessly wide, mostly-empty axis.
N_SIG_PLOT_RANGE = (-100.0, 100.0)


def n_sig_fit_range(n_sig_in):
    return (-500 + n_sig_in, 500 + n_sig_in)


# ── Scan across injected n_sig values ────────────────────────────────────────────────

def run_scan(args):
    mll, met, sig_pdf, bkg_pdf, ratio_peaking, components = build_model_2d_both_bkg(
        args.m1, args.m2, args.r, args.floatR, (args.r_lo, args.r_hi))

    n_bkg_lo, n_bkg_hi = -100.0, 300.0
    tag_suffix = "_floatR" if args.floatR else ""

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
        rows = collect_results_2d(mcs, args.n_experiments, args.floatR)
        add_pulls_2d(rows, n_sig_in, args.n_bkg_in, args.r, args.floatR)

        tag = f"{DATATAG}{tag_suffix}_nsig{n_sig_in:g}_nbkg{args.n_bkg_in:g}_r{args.r:g}_seed{seed}"

        # Budgets reset per injected n_sig point, so every point gets its own set of
        # outlier/example plots instead of the whole scan sharing one global budget.
        n_outliers_plotted = 0
        print(f"For {n_sig_in}: Outlying examples: {args.n_outlier_examples}")
        for idx, row in enumerate(rows):
            if n_outliers_plotted >= args.n_outlier_examples:
                break
            if row["status"] == 0 and args.floatR and row["r_val"] > 0.35:
                make_example_toy_plot(mcs, mll, met, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
                                      ratio_peaking, args.floatR, outlier_plot_dir, tag, idx)
                n_outliers_plotted += 1
            elif row["status"] == 0 and not args.floatR:
                make_example_toy_plot(mcs, mll, met, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
                                      ratio_peaking, args.floatR, outlier_plot_dir, tag, idx)
                n_outliers_plotted += 1

        n_examples_plotted = 0
        for idx in _first_converged_toys(rows, args.n_example_toys):
            make_example_toy_plot(mcs, mll, met, total_pdf, sig_pdf, bkg_pdf, n_sig, n_bkg,
                                  ratio_peaking, args.floatR, example_plot_dir, tag, idx)
            n_examples_plotted += 1

        print(f"n_sig_in={n_sig_in:g}: made {n_outliers_plotted} outlier plot(s) "
             f"(r_val > 0.4) and {n_examples_plotted} converged-fit example plot(s)")

        points.append({
            "n_sig_in": n_sig_in,
            "n_sig_lo": n_sig_lo,
            "n_sig_hi": n_sig_hi,
            "n_bkg_lo": n_bkg_lo,
            "n_bkg_hi": n_bkg_hi,
            "r_lo": args.r_lo,
            "r_hi": args.r_hi,
            "seed": seed,
            "toys": rows,
        })

    print(f"Outlier plots in {outlier_plot_dir}, example plots in {example_plot_dir}")

    return {
        "m1": args.m1,
        "m2": args.m2,
        "n_bkg_in": args.n_bkg_in,
        "r": args.r,
        "floatR": args.floatR,
        "n_experiments": args.n_experiments,
        "seed": args.seed,
        "points": points,
    }


def load_or_run(args):
    tag_suffix = "_floatR" if args.floatR else ""
    json_path = os.path.join(args.outdir, f"results_2d_mll_met_both_bkg{tag_suffix}.json")

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


# ── Plotting (same conventions as the oneDim two-bkg siblings) ──────────────────────

def make_per_point_plots(point, n_bkg_in, r, floatR, n_experiments, mass_point, plot_dir):
    """Best-fit-value and pull distributions for n_sig/n_bkg (+ r iff floatR) at one
    injected point. Returns {val_key: gaussian_fit_result} for the best-fit-value
    distributions, so the caller can use the fit's sigma as the scatter-plot error
    bar for that point."""
    n_sig_in = point["n_sig_in"]
    tag_suffix = "_floatR" if floatR else ""
    tag = f"{DATATAG}{tag_suffix}_nsig{n_sig_in:g}_nbkg{n_bkg_in:g}_r{r:g}"
    toys = [t for t in point["toys"] if t["status"] == 0]

    dist_specs = [
        ("n_sig_val", "Best-fit n_{sig}", "nsig", N_SIG_PLOT_RANGE, n_sig_in),
        ("n_bkg_val", "Best-fit n_{bkg}", "nbkg", (point["n_bkg_lo"], point["n_bkg_hi"]), n_bkg_in),
    ]
    pull_specs = [
        ("n_sig_pull", "Pull(n_{sig})", "nsig"),
        ("n_bkg_pull", "Pull(n_{bkg})", "nbkg"),
    ]
    if floatR:
        dist_specs.append(("r_val", "Best-fit r", "r", (point["r_lo"], point["r_hi"]), r))
        pull_specs.append(("r_pull", "Pull(r)", "r"))

    fit_results = {}
    for key, label, name, rng, refline_val in dist_specs:
        vals = [t[key] for t in toys]
        fit_results[key] = make_distribution_plot(vals, label, f"distribution_{name}_{tag}", plot_dir,
                               mass_point, n_sig_in, n_bkg_in, n_experiments, x_range=rng,
                               refline_val=refline_val, nbins=100, fit_gaussian=True)

    for key, label, name in pull_specs:
        vals = [t[key] for t in toys if t[key] is not None]
        make_distribution_plot(vals, label, f"pull_{name}_{tag}", plot_dir,
                               mass_point, n_sig_in, n_bkg_in, n_experiments, fit_gaussian=True)

    return fit_results


def make_plots(result, plot_dir):
    mass_point = (result["m1"], result["m2"])
    n_bkg_in = result["n_bkg_in"]
    r = result["r"]
    floatR = result["floatR"]
    tag_suffix = "_floatR" if floatR else ""
    n_experiments = result["n_experiments"]
    points = result["points"]

    n_sig_in_vals = [p["n_sig_in"] for p in points]

    n_sig_means, n_sig_errs = [], []
    n_bkg_means, n_bkg_errs = [], []
    r_means, r_errs = [], []
    for p in points:
        mv, me = aggregate_point(p, "n_sig_val")
        n_sig_means.append(mv)
        n_sig_errs.append(me)
        mv, me = aggregate_point(p, "n_bkg_val")
        n_bkg_means.append(mv)
        n_bkg_errs.append(me)
        if floatR:
            mv, me = aggregate_point(p, "r_val")
            r_means.append(mv)
            r_errs.append(me)

        make_per_point_plots(p, n_bkg_in, r, floatR, n_experiments, mass_point, plot_dir)

    make_scatter_plot(
        n_sig_in_vals, n_sig_means, n_sig_errs,
        xlabel="Injected n_{sig}", ylabel="Average best-fit n_{sig}",
        refline=lambda x: x, refline_label="y = x",
        plotname=f"scatter_2d_mll_met_both_bkg{tag_suffix}_nsig", plot_dir=plot_dir, mass_point=mass_point,
        n_bkg_in=n_bkg_in, n_experiments=n_experiments, label_points=True)

    make_scatter_plot(
        n_sig_in_vals, n_bkg_means, n_bkg_errs,
        xlabel="Injected n_{sig}", ylabel="Average best-fit n_{bkg}",
        refline=lambda x: n_bkg_in, # refline_label="Injected n_{bkg}",
        plotname=f"scatter_2d_mll_met_both_bkg{tag_suffix}_nbkg", plot_dir=plot_dir, mass_point=mass_point,
        n_bkg_in=n_bkg_in, n_experiments=n_experiments, label_points=True)

    if floatR:
        make_scatter_plot(
            n_sig_in_vals, r_means, r_errs,
            xlabel="Injected n_{sig}", ylabel="Average best-fit r",
            refline=lambda x: r, # refline_label="Injected r",
            plotname=f"scatter_2d_mll_met_both_bkg{tag_suffix}_r", plot_dir=plot_dir, mass_point=mass_point,
            n_bkg_in=n_bkg_in, n_experiments=n_experiments, label_points=True)


# ── Entry point ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scan injected n_sig values for the 2D (m(ll), MET) toy fit "
                    "with both background components (peaking-in-m(ll) + "
                    "non-peaking-in-m(ll)), save every toy's best-fit n_sig/n_bkg "
                    "(and r, with --floatR) to JSON, and make injected-vs-fitted "
                    "scatter plots.")
    parser.add_argument("--floatR", action="store_true",
                        help="Leave r (ratio_peaking) floating in the fit instead of "
                             "fixing it at --r; also plots r's distribution/scatter "
                             "and tags every output with _floatR")
    parser.add_argument("--n-sig-list", nargs="+", type=float, default=[0, 5, 10, 15, 20, 25])
    parser.add_argument("--n-bkg-in", type=float, default=48.0)
    parser.add_argument("--r", type=float, default=0.088,
                        help="Injected peaking fraction of the total background")
    parser.add_argument("--r-lo", type=float, default=0.0, help="Fit range lower bound for r (with --floatR)")
    parser.add_argument("--r-hi", type=float, default=1.0, help="Fit range upper bound for r (with --floatR)")
    parser.add_argument("--n-outlier-examples", type=int, default=5,
                        help="Max number of example toy plots (data + best-fit "
                             "signal/background curves) to make for converged toys "
                             "with best-fit r > 0.4 (only with --floatR), per "
                             "injected n_sig point")
    parser.add_argument("--n-example-toys", type=int, default=5,
                        help="Make example toy plots (data + best-fit signal/"
                             "background curves) for the first N converged toys, "
                             "per injected n_sig point")
    parser.add_argument("-N", "--n-experiments", type=int, default=5000)
    parser.add_argument("--m1", type=int, default=650)
    parser.add_argument("--m2", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", default="/eos/cms/store/group/phys_susy/skkwan/toys/2d_mll_met_both_bkg/scatter_plot_results")
    parser.add_argument("--plot-dir", default="/eos/user/s/skkwan/www/higgsino/studies/mll-MET-fit-2D/toys/two_dimension_both_bkg")
    parser.add_argument("--force", action="store_true",
                        help="Re-run the scan even if the results JSON already exists")
    args = parser.parse_args()

    result, json_path = load_or_run(args)

    print(f"\nMaking scatter plots from {json_path} ...")
    make_plots(result, args.plot_dir)

    print(f"\n Check https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/toys/two_dimension_both_bkg/ for outputs")
    print(f"\n Check https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/toys/two_dimension_both_bkg/example_toys_outliers/ for outlier plots")
    print(f"\n Check https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/toys/two_dimension_both_bkg/example_toys/ for random examples that converged")

if __name__ == "__main__":
    main()
