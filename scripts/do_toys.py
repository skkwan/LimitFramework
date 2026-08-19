#!/usr/bin/env python3
"""
do_toys.py
Generates and fits toy (m(ll), MET) datasets from the signal + non-peaking-background-only
model at one mass point, reusing the model from
2DFit_higgsinos/hbb_zll/toy_experiments/toy_noPkgBkg.py's get_signal_model /
get_background_model, but generating and fitting each toy through real Combine commands
(combine -M FitDiagnostics --saveToys) instead of a bare RooMCStudy, and reporting an
expected limit via combine -M AsymptoticLimits the way the rest of this framework does.

Model: sig_pdf = m(ll) Crystal Ball x MET spline (fitted signal shape at one mass point)
       bkg_pdf = m(ll) Exponential x MET Gumbel (non-peaking-only background shape)
       total_pdf = n_sig * sig_pdf + n_bkg * bkg_pdf

Per toy (i = 1..N):
    combine -M FitDiagnostics -d <card> -t 1 -s <seed_i> --expectSignal {0,1} \
            --rMin -2 --rMax 5 --toysNoSystematics --saveToys -n _<toy_tag>
reads back:
  - the generated toy dataset from higgsCombine_<toy_tag>.FitDiagnostics.mH*.<seed>.root:
    toys/toy_1
  - best-fit r (-> n_sig_yield = r * n_sig_in) and the floating "n_bkg" rateParam
    (-> n_bkg_yield) from fitDiagnostics_<toy_tag>.root's fit_s RooFitResult

Once per (m1, m2, mu):
    combine -M AsymptoticLimits -d <card> --run expected -t -1 --expectSignal {0,1} \
            --rMin -2 --rMax 5 --noFitAsimov -n _<run_tag>
reports the expected limit (log only, not plotted).

Plots (m(ll), MET; linear + log scale) reuse the drawing logic from
2DFit_higgsinos/hbb_zll/toy_experiments/plot_utils_noPkgBkg.py's _draw_toy_frame /
make_toy_plot / _obs_configs -- copied below (not imported), so this script has no
runtime dependency on that unrelated CMSSW_14_0_21 tree beyond the one-time model
construction via toy_noPkgBkg.py.

Usage:
    python3 do_toys.py --mu 0 -N 1
    python3 do_toys.py --mu 1 -N 1 --force
"""

import ROOT
import os
import sys
import glob
import argparse
import subprocess
import cmsstyle as CMS

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))
import config
import run_combine as rc

# Only the model builders are imported from the hbb_zll study area -- the plotting
# helpers below are copied, not imported, so this script has no runtime dependency on
# that unrelated CMSSW_14_0_21 tree beyond the one-time model construction.
HBB_ZLL_TOY_DIR = ("/afs/cern.ch/work/s/skkwan/public/zhmet/CMSSW_14_0_21/src/"
                    "2DFit_higgsinos/hbb_zll/toy_experiments")
sys.path.insert(0, HBB_ZLL_TOY_DIR)
import toy_noPkgBkg as toy_ref


# ── Model construction (wraps toy_ref.get_signal_model / get_background_model) ────────

def build_model(m1, m2):
    """
    Build (mll, met, sig_pdf, bkg_pdf, components) for one mass point, reusing
    toy_noPkgBkg.py's model builders. Those builders open their input ROOT files with
    paths relative to hbb_zll/toy_experiments/, and get_background_model reads a bare
    module-level `mll` global from toy_noPkgBkg -- both are only valid when called the
    way toy_noPkgBkg.py's own __main__ block calls them, so replicate that setup here
    (same injection pattern plot_utils_noPkgBkg.py's find_and_plot_selected_toy uses).
    Caller must keep the returned vars/pdfs/components alive for as long as the model
    is in use (they hold the only references keeping the underlying RooFit/TFile
    objects from being garbage-collected).
    """
    prev_cwd = os.getcwd()
    try:
        os.chdir(HBB_ZLL_TOY_DIR)
        mll = ROOT.RooRealVar("m_ll", "m_ll", 60, 120)
        toy_ref.mll = mll
        sig_pdf, met, sig_components = toy_ref.get_signal_model(m1, m2)
        bkg_pdf, bkg_components = toy_ref.get_background_model(met)
    finally:
        os.chdir(prev_cwd)
    return mll, met, sig_pdf, bkg_pdf, sig_components + bkg_components


# ── Plotting helpers, copied from plot_utils_noPkgBkg.py (not imported) ───────────────

def _obs_configs(mll, met, plotname_prefix):
    return [
        (mll, 40,  60.,  120., "m(ll) [GeV]", f"mll_{plotname_prefix}"),
        (met, 60, 200., 1200., "MET [GeV]",   f"met_{plotname_prefix}"),
    ]


def _draw_toy_frame(toys, model, obs, nBins, xmin, xmax, n_tot_val):
    """
    Build a RooPlot frame with the total/signal/background model curves (normalized to
    n_tot_val) and the toy data overlaid.
    """
    frame = obs.frame(ROOT.RooFit.Bins(nBins), ROOT.RooFit.Range(xmin, xmax), ROOT.RooFit.Title(""))

    model.plotOn(frame,
                 ROOT.RooFit.Name("total"),
                 ROOT.RooFit.Normalization(n_tot_val, ROOT.RooAbsReal.NumEvent),
                 ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#9c9ca1")),
                 ROOT.RooFit.LineWidth(2))

    model.plotOn(frame,
                 ROOT.RooFit.Components("sigtot_mll_met_2dpdf"),
                 ROOT.RooFit.Name("signal"),
                 ROOT.RooFit.Normalization(n_tot_val, ROOT.RooAbsReal.NumEvent),
                 ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#bd1f01")),
                 ROOT.RooFit.LineStyle(ROOT.kDashed),
                 ROOT.RooFit.LineWidth(2))

    model.plotOn(frame,
                 ROOT.RooFit.Components("bkgnonpeak_mll_met_2dpdf"),
                 ROOT.RooFit.Name("bkg_nonpeak"),
                 ROOT.RooFit.Normalization(n_tot_val, ROOT.RooAbsReal.NumEvent),
                 ROOT.RooFit.LineColor(ROOT.TColor.GetColor("#3f90da")),
                 ROOT.RooFit.LineStyle(ROOT.kDashed),
                 ROOT.RooFit.LineWidth(2))

    toys.plotOn(frame,
                ROOT.RooFit.Binning(nBins),
                ROOT.RooFit.Name("data"),
                ROOT.RooFit.MarkerColor(ROOT.kBlack),
                ROOT.RooFit.LineColor(ROOT.kBlack),
                ROOT.RooFit.MarkerStyle(ROOT.kFullCircle),
                ROOT.RooFit.MarkerSize(0.8))

    return frame


def make_toy_plot(frame, obs, nBins, xmin, xmax, xlabel, plotname, doLog,
                  n_sig, n_bkg, n_nonpeak_val, n_nonpeak_err, pull_n_sig,
                  fit_status, cov_qual, mass_point, eos_dir="."):
    if doLog:
        plotname = f"{plotname}_logscale"
        y_min = 1e-6
        y_max = frame.GetMaximum() * 1e10
        frame.SetMinimum(y_min)
        frame.SetMaximum(y_max)
    else:
        y_min = 0
        y_max = 2.2 * frame.GetMaximum()

    pull_hist = frame.pullHist("data", "total")
    pull_hist.SetMarkerStyle(ROOT.kFullCircle)
    pull_hist.SetMarkerSize(0.8)

    frame_pull = obs.frame(ROOT.RooFit.Bins(nBins), ROOT.RooFit.Range(xmin, xmax), ROOT.RooFit.Title(""))
    frame_pull.addPlotable(pull_hist, "P")

    CMS.SetExtraText("Private work")
    CMS.SetCmsText("CMS", font=62, size=0.76)
    CMS.SetLumi(250, unit="fb", run="2018")

    canv = CMS.cmsDiCanvas("canv_" + plotname, x_min=xmin, x_max=xmax, y_min=y_min, y_max=y_max,
                           r_min=-2, r_max=2,
                           nameXaxis=f"{xlabel}",
                           nameYaxis="Events",
                           nameRatio="Pull",
                           square=True, extraSpace=0.01, iPos=0.)
    canv.SetRightMargin(0.05)
    CMS.UpdatePad(canv)

    canv.cd(1)
    if doLog:
        ROOT.gPad.SetLogy()

    m1, m2 = mass_point
    leg = ROOT.TLegend(0.25, 0.45, 0.90, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.AddEntry(frame.findObject("data"),        "Toy data", "PE")
    leg.AddEntry(frame.findObject("total"),       "Total model", "L")
    leg.AddEntry(frame.findObject("signal"),      f"Signal ({m1}, {m2}) GeV (n_{{sig}} = {n_sig.getVal():.2f} +/- {n_sig.getError():.2f})", "L")
    leg.AddEntry(frame.findObject("bkg_nonpeak"), f"Non-peaking background (n_{{nonpeak}} = {n_nonpeak_val:.2f} +/- {n_nonpeak_err:.2f})", "L")
    r_dummy = ROOT.TLine()
    r_dummy.SetLineWidth(0)
    r_dummy.SetLineColor(0)
    leg.AddEntry(r_dummy, f"n_{{bkg}} = {n_bkg.getVal():.2f} +/- {n_bkg.getError():.2f}", "L")
    leg.AddEntry(ROOT.nullptr, f"n_{{sig}} #sigma: {n_sig.getError():.2f}, pull: {pull_n_sig:.2f}", "")
    leg.AddEntry(ROOT.nullptr, f"Fit status: {fit_status}, covQual: {cov_qual}", "")

    frame.Draw("SAME")

    canv.cd(2)
    frame_pull.Draw("SAME")
    zero_line = ROOT.TLine(xmin, 0, xmax, 0)
    zero_line.SetLineColor(ROOT.kBlack)
    zero_line.SetLineWidth(1)
    zero_line.SetLineStyle(ROOT.kDashed)
    zero_line.Draw("SAME")

    canv.cd(1)
    CMS.cmsObjectDraw(leg)
    CMS.UpdatePad(canv)

    canv.SaveAs(f"{plotname}.eps")
    os.system(f"gs -q -dBATCH -dNOPAUSE -dSAFER -dEPSCrop -dPDFSETTINGS=/prepress -sDEVICE=pdfwrite "
              f"-dEmbedAllFonts=true -dSubsetFonts=true -sOutputFile={plotname}.pdf {plotname}.eps && rm {plotname}.eps")
    canv.SaveAs(f"{plotname}.png")
    print(f"Created {plotname}.pdf / .png")
    if eos_dir != ".":
        os.system(f"mv {plotname}.pdf {eos_dir}/")
        os.system(f"mv {plotname}.png {eos_dir}/")


# ── Datacard / workspace ────────────────────────────────────────────────────────────

def datacard_and_workspace_paths(outdir, m1, m2, n_sig_in, n_bkg_in):
    model_base = f"{m1}_{m2}_nsig{n_sig_in:g}_nbkg{n_bkg_in:g}"
    ws_name   = f"ws_toy_{m1}_{m2}"
    ws_path   = os.path.join(outdir, "workspaces", f"workspace_{model_base}.root")
    card_path = os.path.join(outdir, "datacards", f"datacard_{model_base}.txt")
    return model_base, ws_name, ws_path, card_path


def build_datacard_workspace(m1, m2, n_sig_in, n_bkg_in, ws_name, ws_path, card_path, force=False):
    if not force and os.path.exists(ws_path) and os.path.exists(card_path):
        print(f"  EXISTS (skip): {os.path.basename(ws_path)}, {os.path.basename(card_path)}")
        return

    mll, met, sig_pdf, bkg_pdf, components = build_model(m1, m2)
    bkg_range_lo = -1 * n_bkg_in
    bkg_range_hi = 2 * n_bkg_in

    ws = ROOT.RooWorkspace(ws_name, ws_name)
    ws.Import(sig_pdf, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    ws.Import(bkg_pdf, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())

    ds_obs = bkg_pdf.generate(ROOT.RooArgSet(mll, met), int(round(n_bkg_in)))
    ds_obs.SetName("data_obs")
    ds_obs.SetTitle("data_obs")
    ws.Import(ds_obs)

    # POI (r) + floating n_bkg + full s+b model, so this workspace is self-contained and
    # carries its own ModelConfig/ModelConfig_bonly rather than relying on combine's
    # on-the-fly text-datacard conversion, which only builds "ModelConfig" and then
    # auto-derives "ModelConfig_bonly" at runtime by cloning it with r fixed to 0.
    r = ROOT.RooRealVar("r", "signal strength", 1, -2, 5)
    n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", n_bkg_in, bkg_range_lo, bkg_range_hi)
    n_sig_scaled = ROOT.RooProduct("n_sig_scaled", "n_sig_scaled",
                                   ROOT.RooArgList(r, ROOT.RooFit.RooConst(n_sig_in)))
    total_pdf = ROOT.RooAddPdf("total_pdf", "total_pdf",
                               ROOT.RooArgList(sig_pdf, bkg_pdf),
                               ROOT.RooArgList(n_sig_scaled, n_bkg))
    # ModelConfig_bonly's pdf is the background-only 2D pdf directly (extended by n_bkg),
    # not a r=0 clone of total_pdf -- no signal term appears in it at all.
    bkg_only_pdf = ROOT.RooExtendPdf("bkg_only_extpdf", "bkg_only_extpdf", bkg_pdf, n_bkg)

    observables = ROOT.RooArgSet(mll, met)

    mc = ROOT.RooStats.ModelConfig("ModelConfig", ws)
    mc.SetPdf(total_pdf)
    mc.SetObservables(observables)
    mc.SetParametersOfInterest(ROOT.RooArgSet(r))
    ws.Import(mc)

    mc_bonly = ROOT.RooStats.ModelConfig("ModelConfig_bonly", ws)
    mc_bonly.SetPdf(bkg_only_pdf)
    mc_bonly.SetObservables(observables)
    ws.Import(mc_bonly)

    os.makedirs(os.path.dirname(ws_path), exist_ok=True)
    ws.writeToFile(ws_path)
    print(f"  Saved workspace: {ws_path}")

    os.makedirs(os.path.dirname(card_path), exist_ok=True)
    with open(card_path, 'w') as f:
        f.write("# Standalone toy datacard (toy_noPkgBkg.py model reused via do_toys.py)\n")
        f.write(f"# m1={m1} m2={m2}  n_sig_in={n_sig_in}  n_bkg_in={n_bkg_in}\n\n")
        f.write("imax 1\njmax *\nkmax *\n\n---\n\n")
        f.write(f"shapes sig      SR  {ws_path}  {ws_name}:sigtot_mll_met_2dpdf\n")
        f.write(f"shapes bkg      SR  {ws_path}  {ws_name}:bkgnonpeak_mll_met_2dpdf\n")
        f.write(f"shapes data_obs SR  {ws_path}  {ws_name}:data_obs\n")
        f.write("\n---\n\n")
        f.write("bin         SR\n")
        f.write("observation -1\n")
        f.write("\n---\n\n")
        f.write("bin      SR   SR\n")
        f.write("process  sig  bkg\n")
        f.write("process  0    1\n")
        f.write(f"rate     {n_sig_in:.6f}  1.0\n")
        f.write(f"\nn_bkg  rateParam  SR  bkg  {n_bkg_in:g}  [{bkg_range_lo:g},{bkg_range_hi:g}]\n")
    print(f"  Wrote datacard: {card_path}")


# ── Combine runners ──────────────────────────────────────────────────────────────────

def run_fit_diagnostics(ws_path, ws_name, mu, seed_i, toy_tag, fits_dir, timeout, t=1):
    """
    Generate + fit one toy via combine -M FitDiagnostics --saveToys -t <t>. One combine
    call per toy index (not a single -t N call): FitDiagnostics -t N only persists a
    single fit_s/fit_b RooFitResult (the last toy processed), not one per toy.

    Loads the workspace (ws_path/ws_name) directly rather than the .txt datacard, so
    combine picks up the workspace's own ModelConfig/ModelConfig_bonly instead of
    parsing the datacard on the fly and auto-deriving ModelConfig_bonly at runtime.

    t: combine's -t flag. 1 = generate and fit one toy. 0 = fit data_obs directly
    (no toy generated; --saveToys then has nothing to save).
    """
    os.makedirs(fits_dir, exist_ok=True)
    cmd = [
        "combine", "-M", "FitDiagnostics",
        "-d", ws_path,
        "-w", ws_name,
        "-t", str(t),
        "-s", str(seed_i),
        "--expectSignal", str(mu),
        "--rMin", "-2", "--rMax", "5",
        "--toysNoSystematics",
        "--saveToys",
        "-n", f"_{toy_tag}",
    ]
    print(f">>>>>> {cmd}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=fits_dir, timeout=timeout)
    except FileNotFoundError:
        return {'status': 'combine_not_found'}
    except subprocess.TimeoutExpired:
        return {'status': 'timeout'}

    if proc.returncode != 0:
        return {'status': 'failed', 'stderr': proc.stderr[-600:]}

    fitdiag = glob.glob(os.path.join(fits_dir, f"fitDiagnostics_{toy_tag}.root"))
    higgs   = glob.glob(os.path.join(fits_dir, f"higgsCombine_{toy_tag}.FitDiagnostics.mH*.{seed_i}.root"))
    if not fitdiag or not higgs:
        return {'status': 'output_not_found', 'stdout': proc.stdout[-300:]}

    return {'status': 'ok', 'fitdiag_path': fitdiag[0], 'higgs_path': higgs[0]}


def read_fit_result(fitdiag_path, n_sig_in):
    """
    Read the best-fit r (-> n_sig_yield = r * n_sig_in) and the floating "n_bkg"
    rateParam (-> n_bkg_yield) from fitDiagnostics.root's fit_s RooFitResult.
    """
    f = ROOT.TFile.Open(fitdiag_path, "READ")
    fit_s = f.Get("fit_s")
    if not fit_s:
        f.Close()
        raise RuntimeError(f"fit_s not found in {fitdiag_path}")

    r_par    = fit_s.floatParsFinal().find("r")
    nbkg_par = fit_s.floatParsFinal().find("n_bkg")
    if not r_par or not nbkg_par:
        f.Close()
        raise RuntimeError(f"r / n_bkg not found among floating parameters in {fitdiag_path}")

    result = {
        'n_sig_val':  r_par.getVal() * n_sig_in,
        'n_sig_err':  r_par.getError() * n_sig_in,
        'n_bkg_val':  nbkg_par.getVal(),
        'n_bkg_err':  nbkg_par.getError(),
        'fit_status': fit_s.status(),
        'cov_qual':   fit_s.covQual(),
    }
    f.Close()
    return result


def read_toy_dataset(higgs_path):
    """
    Returns (dataset, tfile) -- caller must keep tfile open (and Close() it when done)
    for as long as the dataset is used, since the RooDataSet is backed by this file.
    """
    f = ROOT.TFile.Open(higgs_path, "READ")
    toy_dir = f.Get("toys")
    if not toy_dir:
        f.Close()
        raise RuntimeError(f"'toys' directory not found in {higgs_path}")
    toy_ds = toy_dir.Get("toy_1")
    if not toy_ds:
        f.Close()
        raise RuntimeError(f"toy_1 not found in {higgs_path}")
    return toy_ds, f


def run_asymptotic_limits(ws_path, ws_name, mu, run_tag, limits_dir, timeout):
    """
    Asimov-based expected limit for this (workspace, mu) -- deterministic given the
    workspace's fixed rates, so run once per (m1, m2, mu), not once per toy. Log-only,
    does not feed the plot. Loads ws_path/ws_name directly (see run_fit_diagnostics).
    """
    os.makedirs(limits_dir, exist_ok=True)
    cmd = [
        "combine", "-M", "AsymptoticLimits",
        "-d", ws_path,
        "-w", ws_name,
        "--run", "expected",
        "-t", "-1",
        "--expectSignal", str(mu),
        "--rMin", "-2", "--rMax", "5",
        "--noFitAsimov",
        "-n", f"_{run_tag}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=limits_dir, timeout=timeout)
    except FileNotFoundError:
        return {'status': 'combine_not_found'}
    except subprocess.TimeoutExpired:
        return {'status': 'timeout'}

    if proc.returncode != 0:
        return {'status': 'failed', 'stderr': proc.stderr[-600:]}

    limits = rc.parse_limits(proc.stdout)
    if not limits:
        return {'status': 'parse_failed', 'stdout': proc.stdout[-300:]}
    return {'status': 'ok', **limits}


# ── Plotting entry point ────────────────────────────────────────────────────────────

def plot_toy(m1, m2, n_sig_in, n_bkg_in, mu, fit_info, toy_ds, toy_tag, plots_dir):
    mll, met, sig_pdf, bkg_pdf, components = build_model(m1, m2)

    n_sig = ROOT.RooRealVar("n_sig", "n_sig", fit_info['n_sig_val'], -2 * n_sig_in, 5 * n_sig_in)
    n_sig.setError(fit_info['n_sig_err'])
    n_bkg = ROOT.RooRealVar("n_bkg", "n_bkg", fit_info['n_bkg_val'], -2 * n_bkg_in, 5 * n_bkg_in)
    n_bkg.setError(fit_info['n_bkg_err'])
    model = ROOT.RooAddPdf("total_pdf", "total_pdf",
                           ROOT.RooArgList(sig_pdf, bkg_pdf),
                           ROOT.RooArgList(n_sig, n_bkg))

    n_tot_val    = fit_info['n_sig_val'] + fit_info['n_bkg_val']
    injected_sig = mu * n_sig_in
    pull_n_sig   = ((fit_info['n_sig_val'] - injected_sig) / fit_info['n_sig_err']
                    if fit_info['n_sig_err'] else 0.0)

    os.makedirs(plots_dir, exist_ok=True)
    prev_cwd = os.getcwd()
    try:
        os.chdir(plots_dir)
        for obs, nBins, xmin, xmax, xlabel, plotname in _obs_configs(mll, met, toy_tag):
            frame = _draw_toy_frame(toy_ds, model, obs, nBins, xmin, xmax, n_tot_val)
            plot_kwargs = dict(obs=obs, nBins=nBins, xmin=xmin, xmax=xmax, xlabel=xlabel,
                               plotname=plotname,
                               n_sig=n_sig, n_bkg=n_bkg,
                               n_nonpeak_val=fit_info['n_bkg_val'], n_nonpeak_err=fit_info['n_bkg_err'],
                               pull_n_sig=pull_n_sig,
                               fit_status=fit_info['fit_status'], cov_qual=fit_info['cov_qual'],
                               mass_point=(m1, m2),
                               eos_dir=plots_dir)
            make_toy_plot(frame, doLog=False, **plot_kwargs)
            make_toy_plot(frame, doLog=True,  **plot_kwargs)
    finally:
        os.chdir(prev_cwd)


# ── Entry point ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate and fit toy (m(ll), MET) datasets from the signal + "
                    "non-peaking-background-only model via Combine, reusing "
                    "toy_noPkgBkg.py's model and reporting an AsymptoticLimits expected limit.")
    parser.add_argument("--m1", type=int, default=650)
    parser.add_argument("--m2", type=int, default=1)
    parser.add_argument("--n-sig-in", type=float, default=2.4,
                        help="Nominal signal yield at signal strength mu=1")
    parser.add_argument("--n-bkg-in", type=float, default=42,
                        help="Nominal (floating) non-peaking background yield")
    parser.add_argument("-N", "--n-experiments", type=int, default=1)
    parser.add_argument("--mu", "--signal-strength", dest="mu", type=float,
                        choices=[0.0, 1.0], default=1.0,
                        help="Injected signal strength: 0 = background-only, "
                             "1 = background + n_sig_in signal events")  
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--outdir", default="/eos/cms/store/group/phys_susy/skkwan/toys/combine")
    parser.add_argument("--combine-timeout", type=int, default=600)
    args = parser.parse_args()

    model_base, ws_name, ws_path, card_path = datacard_and_workspace_paths(
        args.outdir, args.m1, args.m2, args.n_sig_in, args.n_bkg_in)

    print(f"=== Datacard/workspace: m1={args.m1} m2={args.m2} "
         f"n_sig_in={args.n_sig_in} n_bkg_in={args.n_bkg_in} ===", flush=True)
    build_datacard_workspace(args.m1, args.m2, args.n_sig_in, args.n_bkg_in,
                             ws_name, ws_path, card_path, force=args.force)

    run_tag    = f"{model_base}_mu{args.mu:g}_N{args.n_experiments}_seed{args.seed}"
    fits_dir   = os.path.join(args.outdir, "fits")
    limits_dir = os.path.join(args.outdir, "limits")
    plots_dir  = os.path.join(args.outdir, "plots")

    n_ok = n_fail = 0
    for i in range(1, args.n_experiments + 1):
        seed_i  = args.seed + i
        toy_tag = f"{run_tag}_toy{i}"
        print(f"\n--- Toy {i}/{args.n_experiments}  (seed={seed_i}) ---", flush=True)

        existing_plot = os.path.join(plots_dir, f"mll_{toy_tag}.pdf")
        if not args.force and os.path.exists(existing_plot):
            print(f"  EXISTS (skip): {os.path.basename(existing_plot)}")
            n_ok += 1
            continue

        fd_result = run_fit_diagnostics(ws_path, ws_name, args.mu, seed_i, toy_tag, fits_dir,
                                        args.combine_timeout)
        if fd_result['status'] == 'combine_not_found':
            print("  !! combine not found — source Combine environment")
            n_fail += 1
            continue
        if fd_result['status'] != 'ok':
            print(f"  !! FitDiagnostics failed: {fd_result}")
            n_fail += 1
            continue

        fit_info = read_fit_result(fd_result['fitdiag_path'], args.n_sig_in)
        print(f"  OK  n_sig_yield = {fit_info['n_sig_val']:.2f} +/- {fit_info['n_sig_err']:.2f}   "
             f"n_bkg_yield = {fit_info['n_bkg_val']:.2f} +/- {fit_info['n_bkg_err']:.2f}   "
             f"fit_status={fit_info['fit_status']} covQual={fit_info['cov_qual']}", flush=True)

        toy_ds, toy_file = read_toy_dataset(fd_result['higgs_path'])
        try:
            plot_toy(args.m1, args.m2, args.n_sig_in, args.n_bkg_in, args.mu,
                    fit_info, toy_ds, toy_tag, plots_dir)
        finally:
            toy_file.Close()
        n_ok += 1

    print(f"\nDone: {n_ok} toys OK, {n_fail} failed", flush=True)

    print(f"\n=== AsymptoticLimits: mu={args.mu} ===", flush=True)
    al_result = run_asymptotic_limits(ws_path, ws_name, args.mu, run_tag, limits_dir, args.combine_timeout)
    if al_result['status'] == 'combine_not_found':
        print("  !! combine not found — source Combine environment")
    elif al_result['status'] != 'ok':
        print(f"  !! AsymptoticLimits failed: {al_result}")
    else:
        print(f"  Expected limit (mu={args.mu}): "
             f"-2s={al_result['exp_m2']:.4f}  -1s={al_result['exp_m1']:.4f}  "
             f"median={al_result['exp']:.4f}  +1s={al_result['exp_p1']:.4f}  +2s={al_result['exp_p2']:.4f}",
             flush=True)


if __name__ == "__main__":
    main()
