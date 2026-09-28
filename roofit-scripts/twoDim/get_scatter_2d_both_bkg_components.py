#!/usr/bin/env python3
"""
get_scatter_2d_both_bkg_components.py
2D (m(ll), MET) analogue of the one-dimensional two-background scatter studies.
Wraps helper.py's model-building/toy-running functions (full 2D signal pdf + BOTH
background components, peaking-in-m(ll) and non-peaking-in-m(ll)) to scan several
injected n_sig values, save every toy's best-fit n_sig/n_bkg_peak/n_bkg_nonpeak to a
single JSON file, make CMS-styled (cmsstyle "Private work") scatter plots of injected
vs. recovered yield, and dump per-toy example fit plots to example_toys/ and
example_toys_outliers/ (one plot per observable, m(ll) and MET, per toy).

Every toy generates exactly --n-bkg-peaking-gen peaking-in-m(ll) background events and
exactly --n-bkg-nonpeak-gen non-peaking-in-m(ll) background events (fixed counts, no
Poisson fluctuation, no probabilistic mixture split -- the two components are generated
independently). The fit model fits n_sig, n_bkg_peak, and n_bkg_nonpeak as three fully
independent yields (no shape-mixture ratio parameter at all -- peaking and non-peaking
are separate RooAddPdf components, each with its own yield).

Usage:
    python3 get_scatter_2d_both_bkg_components.py
    python3 get_scatter_2d_both_bkg_components.py --n-sig-list 0 5 10 15 20 25 -N 100 --n-bkg-peaking-gen 13 --n-bkg-nonpeak-gen 132
    python3 get_scatter_2d_both_bkg_components.py --force
"""

import ROOT
import os
import json
import argparse

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

from helper import (
    build_model_2d_both_bkg, run_mcstudy_2d_both_bkg, collect_results_2d, add_pulls_2d,
    make_example_toy_plot_both_bkg, _first_converged_toys, save_toy_dataset,
    aggregate_point, make_scatter_plot, make_distribution_plot,
)

DATATAG = "two_dim_both_bkg"

# Plotted x-range for the n_sig distribution plots -- narrower than the fit range
# (n_sig_fit_range below) so the fit itself stays unconstrained over +/-500, while
# toys landing outside +/-100 get clamped into the edge bin as a visible
# underflow/overflow bin instead of being fit over a needlessly wide, mostly-empty axis.
N_SIG_PLOT_RANGE = (-100.0, 100.0)

# Criteria for the outlier toy plots
OUTLIER_NSIG_MAX = -40


def n_sig_fit_range(n_sig_in):
    return (-100 + n_sig_in, 100 + n_sig_in)


def _r_fit(row):
    total = row["n_bkg_peak_val"] + row["n_bkg_nonpeak_val"]
    return row["n_bkg_peak_val"] / total if total > 0 else None


# ── Scan across injected n_sig values ────────────────────────────────────────────────

def run_scan(args):
    n_bkg_total_gen = args.n_bkg_peaking_gen + args.n_bkg_nonpeak_gen

    mll, met, sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf, components = (
        build_model_2d_both_bkg(args.m1, args.m2))

    n_bkg_lo, n_bkg_hi = -100.0, 300.0

    outlier_plot_dir = os.path.join(args.plot_dir, "example_toys_outliers")
    example_plot_dir = os.path.join(args.plot_dir, "example_toys")
    # The toy datasets themselves (not plots) go next to the results JSON, so
    # analyze_single_toy.py can re-fit any outlier long after the scan has exited.
    outlier_toy_dir = os.path.join(args.outdir, "example_outlier_toys")

    points = []
    for i, n_sig_in in enumerate(args.n_sig_list):
        n_sig_lo, n_sig_hi = n_sig_fit_range(n_sig_in)
        seed = args.seed + i
        mcs, n_sig, n_bkg_peak, n_bkg_nonpeak, total_pdf = run_mcstudy_2d_both_bkg(
            mll, met, sig_pdf, bkg_peaking_pdf, bkg_nonpeaking_pdf,
            n_sig_in, args.n_bkg_peaking_gen, args.n_bkg_nonpeak_gen,
            (n_sig_lo, n_sig_hi), (n_bkg_lo, n_bkg_hi), (n_bkg_lo, n_bkg_hi),
            args.n_experiments, seed)
        rows = collect_results_2d(mcs, args.n_experiments)
        add_pulls_2d(rows, n_sig_in, args.n_bkg_peaking_gen, args.n_bkg_nonpeak_gen)

        tag = (f"{DATATAG}_nsig{n_sig_in:g}_nbkgpeak{args.n_bkg_peaking_gen:g}"
              f"_nbkgnonpeak{args.n_bkg_nonpeak_gen:g}_seed{seed}")

        # Budgets reset per injected n_sig point, so every point gets its own set of
        # outlier/example plots instead of the whole scan sharing one global budget.
        n_outliers_plotted = 0
        print(f"For {n_sig_in}: Outlying examples: {args.n_outlier_examples}")
        for idx, row in enumerate(rows):
            if n_outliers_plotted >= args.n_outlier_examples:
                break
            if row["status"] == 0 and row["n_sig_val"] < OUTLIER_NSIG_MAX:
                make_example_toy_plot_both_bkg(mcs, mll, met, total_pdf, sig_pdf,
                                      bkg_peaking_pdf, bkg_nonpeaking_pdf,
                                      n_sig, n_bkg_peak, n_bkg_nonpeak, outlier_plot_dir, tag, idx)
                meta = dict(row)
                meta.update({
                    "m1": args.m1,
                    "m2": args.m2,
                    "n_sig_in": n_sig_in,
                    "n_sig_lo": n_sig_lo,
                    "n_sig_hi": n_sig_hi,
                    "n_bkg_lo": n_bkg_lo,
                    "n_bkg_hi": n_bkg_hi,
                    "seed": seed,
                    "toy_index": idx,
                    "tag": tag,
                })
                toy_path = save_toy_dataset(mcs.genData(idx), meta, outlier_toy_dir,
                                            f"outlier_toy{idx}_{tag}")
                print(f"  Saved toy dataset to {toy_path}")
                n_outliers_plotted += 1

        n_examples_plotted = 0
        for idx in _first_converged_toys(rows, args.n_example_toys):
            make_example_toy_plot_both_bkg(mcs, mll, met, total_pdf, sig_pdf,
                                  bkg_peaking_pdf, bkg_nonpeaking_pdf,
                                  n_sig, n_bkg_peak, n_bkg_nonpeak, example_plot_dir, tag, idx)
            n_examples_plotted += 1

        print(f"n_sig_in={n_sig_in:g}: made {n_outliers_plotted} outlier plot(s) "
             f"(best-fit n_sig < {OUTLIER_NSIG_MAX:g}) and "
             f"{n_examples_plotted} converged-fit example plot(s)")

        points.append({
            "n_sig_in": n_sig_in,
            "n_sig_lo": n_sig_lo,
            "n_sig_hi": n_sig_hi,
            "n_bkg_lo": n_bkg_lo,
            "n_bkg_hi": n_bkg_hi,
            "n_bkg_peaking_gen": args.n_bkg_peaking_gen,
            "n_bkg_nonpeak_gen": args.n_bkg_nonpeak_gen,
            "seed": seed,
            "toys": rows,
        })

    print(f"Outlier plots in {outlier_plot_dir}, example plots in {example_plot_dir}")
    print(f"Outlier toy datasets in {outlier_toy_dir} "
         f"(re-fit one with: python3 analyze_single_toy.py <file>.root)")

    return {
        "m1": args.m1,
        "m2": args.m2,
        "n_bkg_peaking_gen": args.n_bkg_peaking_gen,
        "n_bkg_nonpeak_gen": args.n_bkg_nonpeak_gen,
        "n_experiments": args.n_experiments,
        "seed": args.seed,
        "points": points,
    }


def load_or_run(args):
    json_path = os.path.join(args.outdir, "results_2d_mll_met_both_bkg.json")

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

def make_per_point_plots(point, n_bkg_peak_in, n_bkg_nonpeak_in, n_experiments, mass_point, plot_dir):
    """Best-fit-value and pull distributions for n_sig/n_bkg_peak/n_bkg_nonpeak at one
    injected point. Returns {val_key: gaussian_fit_result} for the best-fit-value
    distributions, so the caller can use the fit's sigma as the scatter-plot error
    bar for that point."""
    n_sig_in = point["n_sig_in"]
    tag = (f"{DATATAG}_nsig{n_sig_in:g}_nbkgpeak{point['n_bkg_peaking_gen']:g}"
          f"_nbkgnonpeak{point['n_bkg_nonpeak_gen']:g}")
    toys = [t for t in point["toys"] if t["status"] == 0]
    n_bkg_total_in = n_bkg_peak_in + n_bkg_nonpeak_in

    dist_specs = [
        ("n_sig_val", "Best-fit n_{sig}", "nsig", N_SIG_PLOT_RANGE, n_sig_in),
        ("n_bkg_peak_val", "Best-fit n_{bkg peak}", "nbkgpeak", (point["n_bkg_lo"], point["n_bkg_hi"]), n_bkg_peak_in),
        ("n_bkg_nonpeak_val", "Best-fit n_{bkg nonpeak}", "nbkgnonpeak",
         (point["n_bkg_lo"], point["n_bkg_hi"]), n_bkg_nonpeak_in),
    ]
    pull_specs = [
        ("n_sig_pull", "Pull(n_{sig})", "nsig"),
        ("n_bkg_peak_pull", "Pull(n_{bkg peak})", "nbkgpeak"),
        ("n_bkg_nonpeak_pull", "Pull(n_{bkg nonpeak})", "nbkgnonpeak"),
    ]

    fit_results = {}
    for key, label, name, rng, refline_val in dist_specs:
        vals = [t[key] for t in toys]
        fit_results[key] = make_distribution_plot(vals, label, f"distribution_{name}_{tag}", plot_dir,
                               mass_point, n_sig_in, n_bkg_total_in, n_experiments, x_range=rng,
                               refline_val=refline_val, nbins=100, fit_gaussian=True)

    for key, label, name in pull_specs:
        vals = [t[key] for t in toys if t[key] is not None]
        make_distribution_plot(vals, label, f"pull_{name}_{tag}", plot_dir,
                               mass_point, n_sig_in, n_bkg_total_in, n_experiments, fit_gaussian=True)

    return fit_results


def make_plots(result, plot_dir):
    mass_point = (result["m1"], result["m2"])
    n_bkg_peak_in = result["n_bkg_peaking_gen"]
    n_bkg_nonpeak_in = result["n_bkg_nonpeak_gen"]
    n_bkg_total_in = n_bkg_peak_in + n_bkg_nonpeak_in
    n_experiments = result["n_experiments"]
    points = result["points"]

    n_sig_in_vals = [p["n_sig_in"] for p in points]

    n_sig_means, n_sig_errs = [], []
    n_bkg_peak_means, n_bkg_peak_errs = [], []
    n_bkg_nonpeak_means, n_bkg_nonpeak_errs = [], []
    for p in points:
        mv, me = aggregate_point(p, "n_sig_val")
        n_sig_means.append(mv)
        n_sig_errs.append(me)
        mv, me = aggregate_point(p, "n_bkg_peak_val")
        n_bkg_peak_means.append(mv)
        n_bkg_peak_errs.append(me)
        mv, me = aggregate_point(p, "n_bkg_nonpeak_val")
        n_bkg_nonpeak_means.append(mv)
        n_bkg_nonpeak_errs.append(me)

        make_per_point_plots(p, n_bkg_peak_in, n_bkg_nonpeak_in, n_experiments, mass_point, plot_dir)

    make_scatter_plot(
        n_sig_in_vals, n_sig_means, n_sig_errs,
        xlabel="Injected n_{sig}", ylabel="Average best-fit n_{sig}",
        refline=lambda x: x, refline_label="y = x",
        plotname=f"scatter_2d_mll_met_both_bkg_nsig", plot_dir=plot_dir, mass_point=mass_point,
        n_bkg_in=n_bkg_total_in, n_experiments=n_experiments, label_points=True)

    make_scatter_plot(
        n_sig_in_vals, n_bkg_peak_means, n_bkg_peak_errs,
        xlabel="Injected n_{sig}", ylabel="Average best-fit n_{bkg peak}",
        refline=lambda x: n_bkg_peak_in, # refline_label="Injected n_{bkg peak}",
        plotname=f"scatter_2d_mll_met_both_bkg_nbkgpeak", plot_dir=plot_dir, mass_point=mass_point,
        n_bkg_in=n_bkg_total_in, n_experiments=n_experiments, label_points=True)

    make_scatter_plot(
        n_sig_in_vals, n_bkg_nonpeak_means, n_bkg_nonpeak_errs,
        xlabel="Injected n_{sig}", ylabel="Average best-fit n_{bkg nonpeak}",
        refline=lambda x: n_bkg_nonpeak_in, # refline_label="Injected n_{bkg nonpeak}",
        plotname=f"scatter_2d_mll_met_both_bkg_nbkgnonpeak", plot_dir=plot_dir, mass_point=mass_point,
        n_bkg_in=n_bkg_total_in, n_experiments=n_experiments, label_points=True)


# ── Entry point ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scan injected n_sig values for the 2D (m(ll), MET) toy fit "
                    "with both background components (peaking-in-m(ll) + "
                    "non-peaking-in-m(ll)) fit as independent yields, save every "
                    "toy's best-fit n_sig/n_bkg_peak/n_bkg_nonpeak to JSON, and make "
                    "injected-vs-fitted scatter plots.")
    parser.add_argument("--n-sig-list", nargs="+", type=float, default=[0, 5, 10, 15, 20, 25])
    parser.add_argument("--n-bkg-peaking-gen", type=float, default=4.0,
                        help="Exact number of peaking-in-m(ll) background events "
                             "generated every toy (fixed, not Poisson-fluctuated)")
    parser.add_argument("--n-bkg-nonpeak-gen", type=float, default=44.0,
                        help="Exact number of non-peaking-in-m(ll) background events "
                             "generated every toy (fixed, not Poisson-fluctuated)")
    parser.add_argument("--n-outlier-examples", type=int, default=10,
                        help="Max number of example toy plots (data + best-fit "
                             "signal/background curves) to make for converged toys "
                             f"with fitted peaking fraction > {OUTLIER_NSIG_MAX:g}, "
                             "per injected n_sig point")
    parser.add_argument("--n-example-toys", type=int, default=10,
                        help="Make example toy plots (data + best-fit signal/"
                             "background curves) for the first N converged toys, "
                             "per injected n_sig point")
    parser.add_argument("-N", "--n-experiments", type=int, default=5000)
    parser.add_argument("--m1", type=int, default=650)
    parser.add_argument("--m2", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", default="/eos/cms/store/group/phys_susy/skkwan/toys/2d_mll_met_both_bkg/scatter_plot_results")
    parser.add_argument("--plot-dir", default="/eos/user/s/skkwan/www/higgsino/studies/mll-MET-fit-2D/two_dimensional_fitTo_handling/both_bkg")
    parser.add_argument("--force", action="store_true",
                        help="Re-run the scan even if the results JSON already exists")
    args = parser.parse_args()

    result, json_path = load_or_run(args)

    print(f"\nMaking scatter plots from {json_path} ...")
    make_plots(result, args.plot_dir)

    print(f"\n Check https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/two_dimensional_fitTo_handling/both_bkg/ for outputs")
    print(f"\n Check https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/two_dimensional_fitTo_handling/both_bkg/example_toys_outliers/ for outlier plots")
    print(f"\n Check https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/two_dimensional_fitTo_handling/both_bkg/example_toys/ for random examples that converged")

if __name__ == "__main__":
    main()
