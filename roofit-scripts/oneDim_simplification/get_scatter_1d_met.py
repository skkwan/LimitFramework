#!/usr/bin/env python3
"""
get_scatter_1d_met.py
Wraps do_1d_met_toys.py's model-building/toy-running functions to scan several
injected n_sig values, save every toy's best-fit n_sig/n_bkg (value + error) to
a single JSON file, and make CMS-styled (cmsstyle "Private work") scatter plots
of injected vs. recovered yield. Sibling of get_scatter_1d_mll.py, same structure,
MET instead of m(ll).

Usage:
    python3 get_scatter_1d_met.py
    python3 get_scatter_1d_met.py --n-sig-list 0 5 10 15 20 25 -N 5000 --n-bkg-in 48
    python3 get_scatter_1d_met.py --force
"""

import ROOT
import cmsstyle as CMS
import os
import sys
import json
import math
import argparse

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))
from do_1d_met_toys import build_model_1d, run_mcstudy, collect_results

DATATAG = "one_dim_met_only"

# Plotted x-range for the n_sig distribution plots -- narrower than the fit range
# (n_sig_fit_range below) so the fit itself stays unconstrained over +/-500, while
# toys landing outside +/-100 get clamped into the edge bin as a visible
# underflow/overflow bin instead of being fit over a needlessly wide, mostly-empty axis.
N_SIG_PLOT_RANGE = (-100.0, 100.0)


# ── n_sig fit range per injected point ───────────────────────────────────────────────

def n_sig_fit_range(n_sig_in):
    # if n_sig_in == 0:
    #     return (-12.0, 12.0)
    # return (-3 * abs(n_sig_in) + n_sig_in, 3 * abs(n_sig_in) + n_sig_in)
    return (-500 + n_sig_in, 500 + n_sig_in)


# ── Pulls: (fit - truth) / fit_error, per toy, added in-place to each row ───────────

def add_pulls(rows, n_sig_in, n_bkg_in):
    for r in rows:
        r["n_sig_pull"] = (r["n_sig_val"] - n_sig_in) / r["n_sig_err"] if r["n_sig_err"] > 0 else None
        r["n_bkg_pull"] = (r["n_bkg_val"] - n_bkg_in) / r["n_bkg_err"] if r["n_bkg_err"] > 0 else None
    return rows


# ── Scan across injected n_sig values ────────────────────────────────────────────────

def run_scan(args):
    met, sig_pdf, bkg_pdf, components = build_model_1d(args.m1, args.m2)

    n_bkg_lo, n_bkg_hi = 0.0, 100.0

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
        points.append({
            "n_sig_in": n_sig_in,
            "n_sig_lo": n_sig_lo,
            "n_sig_hi": n_sig_hi,
            "n_bkg_lo": n_bkg_lo,
            "n_bkg_hi": n_bkg_hi,
            "seed": seed,
            "toys": rows,
        })

    return {
        "m1": args.m1,
        "m2": args.m2,
        "n_bkg_in": args.n_bkg_in,
        "n_experiments": args.n_experiments,
        "seed": args.seed,
        "points": points,
    }


def load_or_run(args):
    json_path = os.path.join(args.outdir, "results_met.json")

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


# ── Aggregation: mean +/- per-toy spread (stdev of the fitted values) over converged toys ──

def aggregate_point(point, val_key):
    import statistics as stats

    toys = [t for t in point["toys"] if t["status"] == 0]

    vals = [t[val_key] for t in toys]
    mean_val = stats.mean(vals)
    spread = stats.pstdev(vals) if len(vals) > 1 else 0.0
    print(f">>> {mean_val}, {spread}")
    return mean_val, spread


# ── Plotting (PyROOT + cmsstyle, matching plot_utils.py conventions) ────────────────

def _finalize_canvas(canv, plotname, plot_dir):
    """Shared eps->pdf conversion (keeps Greek symbols intact, unlike a direct .pdf
    SaveAs) + .png + move-to-plot_dir tail, used by every plot in this script."""
    canv.SaveAs(f"{plotname}.eps")
    os.system(f"gs -q -dBATCH -dNOPAUSE -dSAFER -dEPSCrop -dPDFSETTINGS=/prepress -sDEVICE=pdfwrite "
              f"-dEmbedAllFonts=true -dSubsetFonts=true -sOutputFile={plotname}.pdf {plotname}.eps && rm {plotname}.eps")

    canv.SaveAs(f"{plotname}.png")
    print(f"Created {plotname}.pdf / .png")
    os.makedirs(plot_dir, exist_ok=True)
    os.system(f"mv {plotname}.pdf {plotname}.png {plot_dir}/")


def make_scatter_plot(x_vals, y_vals, y_errs, xlabel, ylabel, refline, plotname,
                       plot_dir, mass_point, n_bkg_in, n_experiments, refline_label=None,
                       label_points=False):
    m1, m2 = mass_point
    n = len(x_vals)
    graph = ROOT.TGraphErrors(n)
    for i, (x, y, ey) in enumerate(zip(x_vals, y_vals, y_errs)):
        graph.SetPoint(i, x, y)
        graph.SetPointError(i, 0.0, ey)
    graph.SetMarkerStyle(ROOT.kFullCircle)
    graph.SetMarkerColor(ROOT.kRed + 1)
    graph.SetLineColor(ROOT.kRed + 1)
    graph.SetMarkerSize(1.1)

    CMS.SetExtraText("Private work")
    CMS.SetCmsText("CMS", font=62, size=0.76)
    CMS.SetLumi(250, unit="fb", run="2018")

    x_min = min(0.0, min(x_vals)) - 0.1 * (max(x_vals) - min(x_vals) + 1)
    x_max = max(x_vals) * 1.15 + 1
    y_min = min(0.0, min(y - ey for y, ey in zip(y_vals, y_errs))) - 2
    y_max = max(y + ey for y, ey in zip(y_vals, y_errs)) * 1.25 + 2

    if "nsig" in plotname:
        y_max *= 1.1

    canv = CMS.cmsCanvas(f"canv_{plotname}",
                          x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                          nameXaxis=xlabel, nameYaxis=ylabel,
                          square=False, extraSpace=0.02, iPos=0.)
    canv.SetRightMargin(0.05)
    CMS.UpdatePad(canv)

    line = ROOT.TLine(x_min, refline(x_min), x_max, refline(x_max))
    line.SetLineStyle(ROOT.kDashed)
    line.SetLineColor(ROOT.kBlack)
    line.Draw("SAME")

    CMS.cmsObjectDraw(graph, "P")

    leg = ROOT.TLegend(0.15, 0.78, 0.55, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.032)
    leg.AddEntry(ROOT.nullptr, f"Signal ({m1}, {m2}) GeV", "")
    leg.AddEntry(ROOT.nullptr, f"n_{{bkg}}^{{in}} = {n_bkg_in:g}, N_{{toys}} = {n_experiments}", "")
    if refline_label is not None:
        leg.AddEntry(line, refline_label, "l")
    CMS.cmsObjectDraw(leg)

    if label_points:
        # Keep the TLatex objects alive past this function's return (PyROOT drops
        # unreferenced objects), by hanging them off the canvas.
        canv.point_labels = []
        y_offset = 0.03 * (y_max - y_min)
        for x, y, ey in zip(x_vals, y_vals, y_errs):
            latex = ROOT.TLatex(x, y + ey + y_offset, f"{y:.2f} #pm {ey:.2f}")
            latex.SetTextFont(42)
            latex.SetTextSize(0.032)
            latex.SetTextAlign(21)
            latex.Draw("SAME")
            canv.point_labels.append(latex)

    CMS.UpdatePad(canv)

    _finalize_canvas(canv, plotname, plot_dir)


GAUSSIAN_FIT_COLOR = "#5790fc"


def make_distribution_plot(vals, xlabel, plotname, plot_dir, mass_point, n_sig_in,
                            n_bkg_in, n_experiments, x_range=None, fit_gaussian=False,
                            nbins=40, refline_val=None, fit_color=GAUSSIAN_FIT_COLOR):
    """CMS-styled plot (black points + error bars) of per-toy values (best-fit yield
    or pull) for one injected point. x_range fixes the axis to that (lo, hi) instead
    of the data-driven padded range; fit_gaussian overlays a Gaussian fit and reports
    its mean/sigma/chi2/ndf. refline_val, if given, draws a thin black dashed vertical
    line at that x position (e.g. the injected yield). Values outside [lo, hi] (e.g. a
    fit range wider than the plotted range) are clamped into the first/last bin, so
    that bin doubles as a visible underflow/overflow bin instead of silently vanishing
    into TH1's internal (unplotted) under/overflow bins.

    Returns a dict of the Gaussian fit results (mu, mu_err, sigma, sigma_err, chi2,
    ndf) if fit_gaussian is True, else None -- callers use the sigma as the toy-spread
    error bar on the corresponding scatter-plot point."""
    m1, m2 = mass_point
    if x_range is not None:
        lo, hi = x_range
    else:
        lo, hi = min(vals), max(vals)
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - pad, hi + pad

    hist = ROOT.TH1D(f"h_{plotname}", "", nbins, lo, hi)
    bin_width = (hi - lo) / nbins
    eps = bin_width * 1e-6
    for v in vals:
        hist.Fill(min(max(v, lo + eps), hi - eps))
    hist.SetStats(False)
    hist.SetMarkerStyle(ROOT.kFullCircle)
    hist.SetMarkerColor(ROOT.kBlack)
    hist.SetLineColor(ROOT.kBlack)

    fit = None
    fit_result = None
    if fit_gaussian:
        fit = ROOT.TF1(f"fit_{plotname}", "gaus", lo, hi)
        hist.Fit(fit, "QR")
        fit_result = {
            "mu": fit.GetParameter(1), "mu_err": fit.GetParError(1),
            "sigma": fit.GetParameter(2), "sigma_err": fit.GetParError(2),
            "chi2": fit.GetChisquare(), "ndf": fit.GetNDF(),
        }

    CMS.SetExtraText("Private work")
    CMS.SetCmsText("CMS", font=62, size=0.76)
    CMS.SetLumi(250, unit="fb", run="2018")

    peak = max(hist.GetMaximum(), fit.GetMaximum(lo, hi)) if fit is not None else hist.GetMaximum()
    y_max = peak * (2.0 if fit_gaussian else 1.35)

    canv = CMS.cmsCanvas(f"canv_{plotname}",
                          x_min=lo, x_max=hi, y_min=0.0, y_max=y_max,
                          nameXaxis=xlabel, nameYaxis="Toys",
                          square=False, extraSpace=0.02, iPos=0.)
    canv.SetRightMargin(0.05)
    CMS.UpdatePad(canv)

    CMS.cmsObjectDraw(hist, "PE")
    if fit is not None:
        fit.SetLineColor(ROOT.TColor.GetColor(fit_color))
        fit.SetLineWidth(2)
        fit.Draw("SAME")

    refline = None
    if refline_val is not None:
        refline = ROOT.TLine(refline_val, 0.0, refline_val, y_max)
        refline.SetLineStyle(ROOT.kDashed)
        refline.SetLineColor(ROOT.kBlack)
        refline.SetLineWidth(1)
        refline.Draw("SAME")

    leg = ROOT.TLegend(0.58, 0.76, 0.93, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.032)
    leg.AddEntry(ROOT.nullptr, f"Signal ({m1}, {m2}) GeV", "")
    leg.AddEntry(ROOT.nullptr, f"n_{{sig}}^{{in}} = {n_sig_in:g}, n_{{bkg}}^{{in}} = {n_bkg_in:g}", "")
    leg.AddEntry(ROOT.nullptr, f"N_{{toys}} = {n_experiments}", "")
    CMS.cmsObjectDraw(leg)

    if fit_result is not None:
        mu, mu_err = fit_result["mu"], fit_result["mu_err"]
        sigma, sigma_err = fit_result["sigma"], fit_result["sigma_err"]
        chi2, ndf = fit_result["chi2"], fit_result["ndf"]
        chi2_ndf = chi2 / ndf if ndf > 0 else float("nan")
        box = ROOT.TPaveText(0.58, 0.51, 0.93, 0.74, "NDC")
        box.SetBorderSize(0)
        box.SetFillStyle(0)
        box.SetTextSize(0.032)
        box.SetTextAlign(12)
        box.AddText("Fit parameters:")
        box.AddText(f"#mu: {mu:.3g} #pm {mu_err:.2g}")
        box.AddText(f"#sigma: {sigma:.3g} #pm {sigma_err:.2g}")
        box.AddText(f"#chi^{{2}} / ndf = {chi2:.2f} / {ndf} = {chi2_ndf:.2f}")
        box.Draw()

    CMS.UpdatePad(canv)

    _finalize_canvas(canv, plotname, plot_dir)

    return fit_result


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
        refline=lambda x: x, refline_label="y = x",
        plotname="scatter_1d_met_nsig", plot_dir=plot_dir, mass_point=mass_point,
        n_bkg_in=n_bkg_in, n_experiments=n_experiments)

    make_scatter_plot(
        n_sig_in_vals, n_bkg_means, n_bkg_errs,
        xlabel="Injected n_{sig}", ylabel="Average best-fit n_{bkg}",
        refline=lambda x: n_bkg_in, # refline_label="Injected n_{bkg}",
        plotname="scatter_1d_met_nbkg", plot_dir=plot_dir, mass_point=mass_point,
        n_bkg_in=n_bkg_in, n_experiments=n_experiments)


# ── Entry point ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scan injected n_sig values for the 1D MET toy fit, save every "
                    "toy's best-fit n_sig/n_bkg to JSON, and make injected-vs-fitted "
                    "scatter plots.")
    parser.add_argument("--n-sig-list", nargs="+", type=float, default=[0, 5, 10, 15, 20, 25])
    parser.add_argument("--n-bkg-in", type=float, default=48.0)
    parser.add_argument("-N", "--n-experiments", type=int, default=5000)
    parser.add_argument("--m1", type=int, default=650)
    parser.add_argument("--m2", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", default="/eos/cms/store/group/phys_susy/skkwan/toys/1d_met/scatter_plot_results")
    parser.add_argument("--plot-dir", default="/eos/user/s/skkwan/www/higgsino/studies/mll-MET-fit-2D/toys/one_dimension")
    parser.add_argument("--force", action="store_true",
                        help="Re-run the scan even if results_met.json already exists")
    args = parser.parse_args()

    result, json_path = load_or_run(args)

    print(f"\nMaking scatter plots from {json_path} ...")
    make_plots(result, args.plot_dir)


if __name__ == "__main__":
    main()
