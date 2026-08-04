#!/usr/bin/env python3
"""
fit_background.py
Fits the 2D background model to data sideband, per (signal_region, era).

Model:
  bkg_total = ratio * [Gumbel_real(MET) x Exp_real(mgg)]
            + (1-ratio) * [Gumbel_fake(MET) x Exp_fake(mgg)]

  Each Gumbel has location mu and scale b.
  Each Exp has slope a.
  ratio is the real-MET fraction.

Fit is performed on the mgg sideband [100,120] U [130,200] with full
MET range [0,1300]. The SR mgg window [120,130] is blinded -- and, since
make_slims.py hard-blinds data slims at creation, this script never even has
access to real SR-window events in the first place (the sideband cut applied
in load_data() is a defense-in-depth no-op, not the primary blind).

Per signal region the mbb cut is applied to data:
  HH SR: mbb in [100, 140]
  ZH SR: mbb in [ 60, 100]

Produces 4 fits total: {HH,ZH} x {run2,run3}.

Saves per fit:
  - JSON with fit parameters, errors, fit quality, n_data_sideband (real,
    blind-safe) and bkg_estimate_full (extrapolated from the sideband fit --
    NOT the real full-range data count, which would include the blind window)
  - Multi-page PDF: page 1 = params, pages 2-3 = mgg & MET projections
  - Workspace with FLOATING-parameter PDF (Combine refits to data)

Usage:
    python3 scripts/fit_background.py --sr HH --era run2
    python3 scripts/fit_background.py --era run2 --force
    python3 scripts/fit_background.py --force                  # all 4 combos
"""

import ROOT
import json
import os
import sys
import argparse
import logging
import tempfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Eval)
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))
import config


# Derive sideband and blinding window from config
SB_LOW   = config.SIDEBAND_WINDOWS[0]   # (100, 120)
SB_HIGH  = config.SIDEBAND_WINDOWS[1]   # (130, 200)
MGG_BLIND_LO = SB_LOW[1]                # 120
MGG_BLIND_HI = SB_HIGH[0]               # 130


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
    return logger


# ── Parameter helpers ─────────────────────────────────────────────────────────

def result_to_dict(fit_result):
    out = {}
    it  = fit_result.floatParsFinal().createIterator()
    var = it.Next()
    while var:
        out[var.GetName()] = (var.getVal(), var.getError())
        var = it.Next()
    return out


def high_correlations(fit_result, threshold=0.6):
    pairs = []
    n     = fit_result.floatParsFinal().getSize()
    for i in range(n):
        for j in range(i):
            p1   = fit_result.floatParsFinal().at(i).GetName()
            p2   = fit_result.floatParsFinal().at(j).GetName()
            corr = fit_result.correlation(p1, p2)
            if abs(corr) > threshold:
                pairs.append((p1, p2, corr))
    return pairs


# ── Model builder ─────────────────────────────────────────────────────────────

def build_background_model(mgg_var, met_var, region, era_tag):
    """
    2-component background:
      Real-MET: Gumbel(mu_real, b_real) x Exp(a_real_mgg)
      Fake-MET: Gumbel(mu_fake, b_fake) x Exp(a_fake_mgg)
      Mixture via ratio_realmet.

    Parameter names include region and era_tag for uniqueness when
    combining workspaces in datacards.

    Initial values from previous bbgg+MET analysis (Run 2). Bounds widened
    slightly to accommodate Run 3 differences.
    """
    sfx = f"{region}_{era_tag}"

    # ─── Fake-MET component ────────────────────────────────────────────────
    mu_fake = ROOT.RooRealVar(f"mu_fake_{sfx}",  f"mu_fake_{sfx}",   24.4, 5.0, 40.0)
    b_fake  = ROOT.RooRealVar(f"b_fake_{sfx}",   f"b_fake_{sfx}",    16.8, 5.0, 40.0)
    gumbel_fake = ROOT.RooGenericPdf(
        f"gumbel_fake_{sfx}", f"gumbel_fake_{sfx}",
        f"(1.0/b_fake_{sfx}) * "
        f"exp(-(@0 - mu_fake_{sfx})/b_fake_{sfx} "
        f"     - exp(-(@0 - mu_fake_{sfx})/b_fake_{sfx}))",
        ROOT.RooArgList(met_var, mu_fake, b_fake)
    )

    a_fake_mgg   = ROOT.RooRealVar(f"a_fake_mgg_{sfx}", f"a_fake_mgg_{sfx}",
                                    -0.01, -1.0, 1.0)
    exp_fake_mgg = ROOT.RooExponential(f"exp_fake_mgg_{sfx}", f"exp_fake_mgg_{sfx}",
                                        mgg_var, a_fake_mgg)

    bkg_fake = ROOT.RooProdPdf(f"bkg_fake_{sfx}", f"bkg_fake_{sfx}",
                                ROOT.RooArgList(exp_fake_mgg, gumbel_fake))

    # ─── Real-MET component ────────────────────────────────────────────────
    mu_real = ROOT.RooRealVar(f"mu_real_{sfx}",  f"mu_real_{sfx}",   53.4, 20.0, 100.0)
    b_real  = ROOT.RooRealVar(f"b_real_{sfx}",   f"b_real_{sfx}",    29.9, 10.0,  80.0)
    gumbel_real = ROOT.RooGenericPdf(
        f"gumbel_real_{sfx}", f"gumbel_real_{sfx}",
        f"(1.0/b_real_{sfx}) * "
        f"exp(-(@0 - mu_real_{sfx})/b_real_{sfx} "
        f"     - exp(-(@0 - mu_real_{sfx})/b_real_{sfx}))",
        ROOT.RooArgList(met_var, mu_real, b_real)
    )

    a_real_mgg   = ROOT.RooRealVar(f"a_real_mgg_{sfx}", f"a_real_mgg_{sfx}",
                                    -0.01, -1.0, 1.0)
    exp_real_mgg = ROOT.RooExponential(f"exp_real_mgg_{sfx}", f"exp_real_mgg_{sfx}",
                                        mgg_var, a_real_mgg)

    bkg_real = ROOT.RooProdPdf(f"bkg_real_{sfx}", f"bkg_real_{sfx}",
                                ROOT.RooArgList(exp_real_mgg, gumbel_real))

    # ─── Mixture ───────────────────────────────────────────────────────────
    ratio_realmet = ROOT.RooRealVar(f"ratio_realmet_{sfx}", f"ratio_realmet_{sfx}",
                                     0.1, 0.0, 1.0)
    bkg_total = ROOT.RooAddPdf(f"bkg_total_{sfx}", f"bkg_total_{sfx}",
                                ROOT.RooArgList(bkg_real, bkg_fake),
                                ROOT.RooArgList(ratio_realmet))

    params = [mu_fake, b_fake, a_fake_mgg,
              mu_real, b_real, a_real_mgg,
              ratio_realmet]
    live   = params + [gumbel_fake, exp_fake_mgg, bkg_fake,
                       gumbel_real, exp_real_mgg, bkg_real, bkg_total]
    return bkg_total, params, live


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(years, region):
    """
    Load data slims, apply mbb SR cut plus an explicit mgg-sideband cut.

    The mgg-sideband cut here is defense-in-depth: slims/data/ is already
    hard-blinded at creation (make_slims.py drops SR-window events from the
    file entirely), so this filter should normally be a no-op. It's kept
    explicit so this script is correct on its own terms even if ever pointed
    at an unblinded or externally-provided slim.
    """
    print(f"  -> looking up data files for {years}", flush=True)
    files = config.slim_data_files(years)
    if not files:
        print(f"  ERROR: no data slim files found for years {years}", flush=True)
        return None, None, None, 0
    print(f"  -> found {len(files)} data files", flush=True)

    chain = ROOT.TChain("tree")
    for f in files:
        chain.Add(f)
    print(f"  -> chain built, counting entries (may take a moment)...", flush=True)
    n_chain = chain.GetEntries()
    print(f"  -> chain has {n_chain} entries", flush=True)
    if n_chain == 0:
        return None, None, None, 0

    mbb_lo = config.SIGNAL_REGIONS[region]['mbb_lo']
    mbb_hi = config.SIGNAL_REGIONS[region]['mbb_hi']

    mgg_var = ROOT.RooRealVar("mgg", "m_{#gamma#gamma} [GeV]",
                               config.MGG_LO, config.MGG_HI)
    met_var = ROOT.RooRealVar("met", "p_{T}^{miss} [GeV]",
                               config.MET_LO, config.MET_HI)
    mbb_var = ROOT.RooRealVar("mbb", "m_{bb} [GeV]", 0.0, 500.0)

    # Named ranges on mgg
    mgg_var.setRange("sb_low",      SB_LOW[0],  SB_LOW[1])
    mgg_var.setRange("sb_high",     SB_HIGH[0], SB_HIGH[1])
    mgg_var.setRange("full",        config.MGG_LO, config.MGG_HI)
    mgg_var.setRange("blind",       MGG_BLIND_LO, MGG_BLIND_HI)
    # Projection-only ranges: defined ONLY on mgg so ProjectionRange("mgg_sb_low,mgg_sb_high")
    # doesn't conflict with the MET sb_low/sb_high ranges (which are both [0,1300]
    # and would be "overlapping" when RooFit tries to normalise over MET).
    mgg_var.setRange("mgg_sb_low",  SB_LOW[0],  SB_LOW[1])
    mgg_var.setRange("mgg_sb_high", SB_HIGH[0], SB_HIGH[1])

    # Same range NAMES on met (full MET span) so CutRange("sb_low,sb_high")
    # works on the MET frame too — it selects events by mgg in sideband
    # while leaving MET unconstrained.
    met_var.setRange("sb_low",  config.MET_LO, config.MET_HI)
    met_var.setRange("sb_high", config.MET_LO, config.MET_HI)
    met_var.setRange("full",    config.MET_LO, config.MET_HI)

    obs = ROOT.RooArgSet(mgg_var, met_var, mbb_var)
    cut = (f"mbb > {mbb_lo} && mbb < {mbb_hi} && "
           f"{config.sideband_cut_expr('mgg')}")
    print(f"  -> building RooDataSet with cut '{cut}'  (this is the slow step)",
          flush=True)
    ds  = ROOT.RooDataSet("ds_data", "data sideband (mbb-cut, mgg blinded)",
                           chain, obs, cut)
    n_sideband = ds.numEntries()
    print(f"  -> RooDataSet built: {n_sideband} sideband entries after mbb cut",
          flush=True)

    if n_sideband == 0:
        return None, None, None, 0

    return ds, mgg_var, met_var, n_sideband


# ── Plotting ──────────────────────────────────────────────────────────────────

# Colors matching previous bbgg+MET analysis style
COLOR_BROWN      = "#8B4513"   # real MET
COLOR_DARKORANGE = "#FF8C00"   # fake MET


def draw_projection(var, ds, pdf, region, era_tag, ratio_val,
                    is_mgg, xlabel, x_max=None, name_suffix=""):
    """
    Plot data + total bkg + real-MET + fake-MET components for a sideband fit.

    Pattern follows standard sideband-fit plotting:
      data:  plotted with CutRange("sb_low,sb_high") -> only events with mgg
             in the sideband (the same range names exist on MET with the full
             MET span, so this filter is effectively mgg-only)
      model: 
        - for mgg frame: Range("full") + NormRange("sb_low,sb_high") to draw
                         the extrapolation through the blinded window
        - for MET frame: ProjectionRange("sb_low,sb_high") to marginalise mgg
                         over the sideband only, matching the data subset

    RooFit handles all normalisation automatically — no NumEvent calls needed.
    """
    sfx = f"{region}_{era_tag}"
    real_name = f"bkg_real_{sfx}"
    fake_name = f"bkg_fake_{sfx}"

    if x_max:
        frame     = var.frame(ROOT.RooFit.Range(var.getMin(), x_max),
                              ROOT.RooFit.Title(" "))
        res_frame = var.frame(ROOT.RooFit.Range(var.getMin(), x_max),
                              ROOT.RooFit.Title(" "))
    else:
        frame     = var.frame(ROOT.RooFit.Title(" "))
        res_frame = var.frame(ROOT.RooFit.Title(" "))

    # ── Data — pre-reduced to sideband, just plotOn directly ──
    ds_sb = ds if hasattr(ds, '_is_sideband') else ds
    ds_sb.plotOn(frame,
                 ROOT.RooFit.Name("data"),
                 ROOT.RooFit.MarkerColor(ROOT.kBlack),
                 ROOT.RooFit.LineColor(ROOT.kBlack))

    if is_mgg:
        # Draw model in each sideband separately — single range per call,
        # never comma-separated (avoids fit_nll overlap error on met_var).
        for rng_idx, rng in enumerate(["sb_low", "sb_high"]):
            pdf.plotOn(frame,
                       ROOT.RooFit.Name(f"total_{rng_idx}"),
                       ROOT.RooFit.Range(rng),
                       ROOT.RooFit.LineColor(ROOT.kBlue),
                       ROOT.RooFit.LineWidth(4))
            pdf.plotOn(frame,
                       ROOT.RooFit.Name(f"realmet_{rng_idx}"),
                       ROOT.RooFit.Components(real_name),
                       ROOT.RooFit.Range(rng),
                       ROOT.RooFit.LineColor(ROOT.TColor.GetColor(COLOR_BROWN)),
                       ROOT.RooFit.LineStyle(ROOT.kDotted),
                       ROOT.RooFit.LineWidth(3))
            pdf.plotOn(frame,
                       ROOT.RooFit.Name(f"fakemet_{rng_idx}"),
                       ROOT.RooFit.Components(fake_name),
                       ROOT.RooFit.Range(rng),
                       ROOT.RooFit.LineColor(ROOT.TColor.GetColor(COLOR_DARKORANGE)),
                       ROOT.RooFit.LineStyle(ROOT.kDotted),
                       ROOT.RooFit.LineWidth(3))
        legend_total   = "total_0"
        legend_realmet = "realmet_0"
        legend_fakemet = "fakemet_0"
    else:
        # MET projection: PDF is factorised — MET shape is independent of mgg
        # range. Plot with NO range specification; RooFit normalises to the
        # data count automatically. Never use ProjectionRange with comma-
        # separated names here (triggers the fit_nll overlapping-ranges error).
        pdf.plotOn(frame,
                   ROOT.RooFit.Name("total"),
                   ROOT.RooFit.LineColor(ROOT.kBlue),
                   ROOT.RooFit.LineWidth(4))
        pdf.plotOn(frame,
                   ROOT.RooFit.Name("realmet"),
                   ROOT.RooFit.Components(real_name),
                   ROOT.RooFit.LineColor(ROOT.TColor.GetColor(COLOR_BROWN)),
                   ROOT.RooFit.LineStyle(ROOT.kDotted),
                   ROOT.RooFit.LineWidth(3))
        pdf.plotOn(frame,
                   ROOT.RooFit.Name("fakemet"),
                   ROOT.RooFit.Components(fake_name),
                   ROOT.RooFit.LineColor(ROOT.TColor.GetColor(COLOR_DARKORANGE)),
                   ROOT.RooFit.LineStyle(ROOT.kDotted),
                   ROOT.RooFit.LineWidth(3))
        legend_total   = "total"
        legend_realmet = "realmet"
        legend_fakemet = "fakemet"

    cname  = f"c_{var.GetName()}_{name_suffix}"
    canvas = ROOT.TCanvas(cname, cname, 700, 750)

    # ── Upper pad: fit ──
    pad1 = ROOT.TPad("p1_" + cname, "", 0, 0.30, 1, 1.0)
    pad1.SetBottomMargin(0.02); pad1.SetLeftMargin(0.14)
    pad1.Draw(); pad1.cd()
    frame.GetXaxis().SetLabelSize(0)
    frame.GetXaxis().SetTitleSize(0)
    frame.GetYaxis().SetTitleSize(0.055)
    frame.GetYaxis().SetLabelSize(0.048)
    frame.Draw()

    # Shade blinded mgg window
    blind_box = None
    if is_mgg:
        y_max = frame.GetMaximum() * 1.1
        blind_box = ROOT.TBox(MGG_BLIND_LO, 0, MGG_BLIND_HI, y_max)
        blind_box.SetFillColorAlpha(ROOT.kGray, 0.4)
        blind_box.SetLineColor(0)
        blind_box.Draw("same")
        frame.Draw("same")

    leg = ROOT.TLegend(0.55, 0.55, 0.92, 0.88)
    leg.AddEntry(frame.findObject("data"),         "Data (sideband)", "PE")
    leg.AddEntry(frame.findObject(legend_total),   "Total Bkg",       "L")
    leg.AddEntry(frame.findObject(legend_realmet),
                  f"Real p_{{T}}^{{miss}}  (r={ratio_val:.2f})", "L")
    leg.AddEntry(frame.findObject(legend_fakemet),
                  f"Fake p_{{T}}^{{miss}}  (1-r={1-ratio_val:.2f})", "L")
    if blind_box:
        leg.AddEntry(blind_box, "Blinded", "F")
    leg.SetBorderSize(0); leg.SetTextSize(0.034)
    leg.Draw()

    # ── Lower pad: residuals ──
    canvas.cd()
    pad2 = ROOT.TPad("p2_" + cname, "", 0, 0.0, 1, 0.30)
    pad2.SetTopMargin(0.02); pad2.SetBottomMargin(0.38)
    pad2.SetLeftMargin(0.14)
    pad2.Draw(); pad2.cd()

    if is_mgg:
        # Add residuals for each sideband separately
        for total_nm in ["total_0", "total_1"]:
            rh = frame.residHist("data", total_nm)
            if rh:
                rh.SetMarkerSize(0.8)
                rh.SetMarkerColor(ROOT.kBlack)
                res_frame.addPlotable(rh, "P")
    else:
        rh = frame.residHist("data", legend_total)
        if rh:
            rh.SetMarkerSize(0.8)
            rh.SetMarkerColor(ROOT.kBlack)
            res_frame.addPlotable(rh, "P")
    res_frame.SetMinimum(-3); res_frame.SetMaximum(3)
    res_frame.GetYaxis().SetTitle("Data #minus Bkg")
    res_frame.GetYaxis().SetNdivisions(505)
    res_frame.GetYaxis().SetTitleSize(0.13)
    res_frame.GetYaxis().SetTitleOffset(0.35)
    res_frame.GetYaxis().SetLabelSize(0.11)
    res_frame.GetXaxis().SetTitle(xlabel)
    res_frame.GetXaxis().SetTitleSize(0.13)
    res_frame.GetXaxis().SetLabelSize(0.11)
    res_frame.Draw()

    line0 = ROOT.TLine(var.getMin(), 0,
                       x_max if x_max else var.getMax(), 0)
    line0.SetLineColor(ROOT.kBlue); line0.SetLineWidth(2)
    line0.Draw()

    tmppath = os.path.join(tempfile.mkdtemp(),
                           f"{var.GetName()}_{name_suffix}.png")
    canvas.SaveAs(tmppath)
    canvas._refs = [pad1, pad2, frame, res_frame, leg, line0, blind_box]
    return tmppath, canvas


def make_bkg_plots(result, pdf, ds, mgg_var, met_var,
                   region, era_tag, n_sideband, bkg_estimate_full,
                   pdf_file, log_file):
    logger = setup_logger(log_file)

    pars = result_to_dict(result)
    hc   = high_correlations(result)

    logger.info(f"=== Background fit: region={region}, era={era_tag} ===")
    logger.info(f"  n_data_sideband (mbb-cut, mgg sideband only) = {n_sideband}")
    logger.info(f"  bkg_estimate_full (extrapolated, no blind-window data) "
                f"= {bkg_estimate_full:.1f}")
    logger.info(f"  fit range: mgg [{SB_LOW[0]:.0f},{SB_LOW[1]:.0f}] U "
                f"[{SB_HIGH[0]:.0f},{SB_HIGH[1]:.0f}], MET [0,1300]")
    for name, (val, err) in pars.items():
        logger.info(f"  {name:<30}: {val:.6f} +/- {err:.6f}")
    logger.info(f"  covQual={result.covQual()}  status={result.status()}  "
                f"EDM={result.edm():.2e}  minNLL={result.minNll():.4f}")
    for p1, p2, c in hc:
        logger.info(f"  HIGH CORR: {p1} <-> {p2} = {c:.4f}")

    ratio_name = f"ratio_realmet_{region}_{era_tag}"
    ratio_val  = pars[ratio_name][0]

    # ds is already sideband-only (load_data applies the blind-safe cut) --
    # no further reduction needed before plotting.
    ds_sb = ds
    n_sb  = n_sideband
    logger.info(f"  n_real ~ {ratio_val*n_sb:.1f}  "
                f"n_fake ~ {(1-ratio_val)*n_sb:.1f}")

    path_mgg, _ = draw_projection(
        mgg_var, ds_sb, pdf, region, era_tag, ratio_val,
        is_mgg=True, xlabel=r"m_{\gamma\gamma} [GeV]",
        name_suffix="mgg")

    path_met, _ = draw_projection(
        met_var, ds_sb, pdf, region, era_tag, ratio_val,
        is_mgg=False, xlabel=r"p_{T}^{miss} [GeV]",
        x_max=500.0, name_suffix="met")

    os.makedirs(os.path.dirname(pdf_file), exist_ok=True)
    with PdfPages(pdf_file) as pdf_out:
        # Page 1: param table
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')

        lines = [f"=== Background fit ===",
                 f"  region    = {region}",
                 f"  era       = {era_tag}",
                 f"  n_sb           = {n_sb}    (real data, sideband mgg only)",
                 f"  bkg_est_full   = {bkg_estimate_full:.1f}  "
                 f"(extrapolated from sideband fit -- no blind-window data)",
                 f"  fit range = mgg [{SB_LOW[0]:.0f},{SB_LOW[1]:.0f}] U "
                 f"[{SB_HIGH[0]:.0f},{SB_HIGH[1]:.0f}], MET [0,1300]",
                 ""]
        for n, (v, e) in pars.items():
            lines.append(f"  {n:<30} {v:>14.6f}  +/-  {e:>12.6f}")
        lines.append("")
        lines.append(f"  ratio_realmet = {ratio_val:.4f}  -> "
                     f"n_real~{ratio_val*n_sb:.1f}  n_fake~{(1-ratio_val)*n_sb:.1f}")
        lines.append("")
        lines.append(f"  covQual={result.covQual()}  status={result.status()}  "
                     f"EDM={result.edm():.2e}  minNLL={result.minNll():.4f}")
        if hc:
            for p1, p2, c in hc:
                lines.append(f"  HIGH CORR: {p1} <-> {p2} = {c:.4f}")
        else:
            lines.append("  No high correlations")

        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        pdf_out.savefig(fig); plt.close(fig)

        for path, title in [(path_mgg, r"$m_{\gamma\gamma}$ projection (blinded)"),
                            (path_met, r"$p_T^{\mathrm{miss}}$ projection (sideband mgg)")]:
            fig, ax = plt.subplots(figsize=(7, 7.5))
            ax.imshow(mpimg.imread(path))
            ax.axis('off')
            ax.set_title(title, fontsize=14)
            pdf_out.savefig(fig); plt.close(fig)

    logger.info(f"\nPDF saved: {pdf_file}")


# ── Main fit ──────────────────────────────────────────────────────────────────

def fit_one(region, years, era_tag, force=False):
    out_json = config.bkg_fit_json(region, era_tag)
    out_pdf  = config.bkg_fit_pdf(region, era_tag)
    out_log  = config.bkg_fit_log(region, era_tag)
    out_ws   = config.bkg_fit_ws(region, era_tag)

    if not force and os.path.exists(out_json):
        print(f"  EXISTS (skip): {os.path.basename(out_json)}")
        return True

    ds, mgg_var, met_var, n_sideband = load_data(years, region)
    if ds is None:
        print(f"  SKIP: no data loaded")
        return False

    mbb_lo = config.SIGNAL_REGIONS[region]['mbb_lo']
    mbb_hi = config.SIGNAL_REGIONS[region]['mbb_hi']
    print(f"  Loaded {n_sideband} sideband data entries  "
          f"(mbb cut applied: [{mbb_lo:.0f}, {mbb_hi:.0f}])")

    bkg_total, params, live = build_background_model(
        mgg_var, met_var, region, era_tag)

    # Fit on mgg sideband
    result = bkg_total.fitTo(
            ds,
            ROOT.RooFit.Range("sb_low,sb_high"),
            ROOT.RooFit.Save(True),
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.Warnings(False),
        )

    print(f"  Fit: status={result.status()}  covQual={result.covQual()}  "
          f"EDM={result.edm():.2e}  minNLL={result.minNll():.4f}")
    for p in params:
        print(f"    {p.GetName():<30} = {p.getVal():.4f} +/- {p.getError():.4f}")

    if result.status() != 0:
        print(f"  WARNING: fit did not converge cleanly")

    # Extrapolate the fitted shape to estimate the background across the
    # FULL mgg range (sidebands + blinded window) using only the sideband
    # fit -- this never touches real data in the blind window. Replaces the
    # previous approach of seeding the downstream workspace norm with the
    # literal (blinded, real) full-range data count.
    obs_set = ROOT.RooArgSet(mgg_var, met_var)
    frac_sb = bkg_total.createIntegral(
        obs_set, ROOT.RooFit.NormSet(obs_set),
        ROOT.RooFit.Range("sb_low,sb_high")).getVal()
    bkg_estimate_full = n_sideband / frac_sb if frac_sb > 0 else float(n_sideband)
    print(f"  Sideband fraction of full shape: {frac_sb:.4f}  ->  "
          f"extrapolated full-range bkg estimate = {bkg_estimate_full:.1f}  "
          f"(from n_sideband={n_sideband}, no blind-window data used)")

    # Save workspace — parameters left FLOATING for Combine
    ws = ROOT.RooWorkspace(f"ws_bkg_{region}_{era_tag}")
    ws.Import(bkg_total)
    os.makedirs(os.path.dirname(out_ws), exist_ok=True)
    ws.writeToFile(out_ws)

    # Save JSON
    pars = result_to_dict(result)
    out_data = {
        "region":            region,
        "era":               era_tag,
        "n_data_sideband":   n_sideband,
        "sideband_frac":     frac_sb,
        "bkg_estimate_full": bkg_estimate_full,
        "mbb_cut":     [mbb_lo, mbb_hi],
        "mgg_range":   [config.MGG_LO, config.MGG_HI],
        "met_range":   [config.MET_LO, config.MET_HI],
        "fit_range":   {"sb_low":  list(SB_LOW),
                        "sb_high": list(SB_HIGH)},
        "fit_status":  result.status(),
        "cov_qual":    result.covQual(),
        "edm":         result.edm(),
        "min_nll":     result.minNll(),
        "params":      {n: {"val": v, "err": e} for n, (v, e) in pars.items()},
        "high_corr":   [[p1, p2, c]
                        for p1, p2, c in high_correlations(result)],
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"  Saved {out_json}")

    make_bkg_plots(result, bkg_total, ds, mgg_var, met_var,
                   region, era_tag, n_sideband, bkg_estimate_full,
                   out_pdf, out_log)
    print(f"  Saved {out_pdf}")

    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit 2-component background to data sideband")
    parser.add_argument("--region", choices=["SRHH", "SRZH", "both"], default="both")
    parser.add_argument("--era",    choices=["run2", "run3", "both"],  default="both")
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()

    regions = ["SRHH", "SRZH"]    if args.region == "both" else [args.region]
    eras    = ["run2", "run3"]    if args.era    == "both" else [args.era]

    combos = [(r, era) for r in regions for era in eras]
    print(f"Fitting {len(combos)} background fits  "
          f"[mgg sideband, mbb-cut per region]", flush=True)

    n_ok, n_skip, n_fail = 0, 0, 0
    for region, era in combos:
        years = config.era_years(era)
        print(f"\n--- region={region}  era={era}  (years={years}) ---", flush=True)
        try:
            ok = fit_one(region, years, era, force=args.force)
            if ok:   n_ok   += 1
            else:    n_skip += 1
        except Exception as ex:
            print(f"  ERROR: {ex}")
            import traceback
            traceback.print_exc()
            n_fail += 1

    print(f"\nDone: {n_ok} fitted  {n_skip} skipped  {n_fail} failed")


if __name__ == "__main__":
    main()