#!/usr/bin/env python3
"""
plot.py
Produces CMS-style limit plots (ROOT + cmsstyle, "CMS Preliminary" header),
each saved as .png and .pdf:

  1D Brazil band (--plot 1d):
     x: m(χ̃₁⁰) [GeV]    y: cross section [pb]   (log scale)
     Green band ±1σ, yellow band ±2σ, black dashed = expected limit median
     Theory cross section overlaid

  2D exclusion contour (--plot 2d):
     Color map: expected cross section upper limit [fb]  (log scale)
     Red contours: expected median, ±1σ, ±2σ exclusion boundaries

Single signal process (TChiZH) -- there's no branching-fraction mixing in
this channel, so (unlike the H(gg)+H/Z(bb) channel this framework was
originally scaffolded for) there is no BF-exclusion plot mode here.

Usage:
    python3 plot.py --era run2 --m gmsb --plot 1d
    python3 plot.py --era run2 --m bino --mlsp 100 --plot 1d
    python3 plot.py --era run2 --m bino --plot 2d
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from contourpy import contour_generator
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import ROOT
import cmsstyle as CMS

sys.path.insert(0, os.path.dirname(__file__))
import config

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning  # silence the "file has been created" Info lines

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# Centre-of-mass energy for the header; the lumi is config.PROJECTED_LUMI for
# every era, since the datacard yields are projected to it
ENERGY_LABELS = {
    'run2':     '13 TeV',
    'run3':     '13.6 TeV',
    'run2run3': '13+13.6 TeV',
}


# ══════════════════════════════════════════════════════════════════════════════
#  cmsstyle helpers
# ══════════════════════════════════════════════════════════════════════════════

def cms_canvas(name, x_min, x_max, y_min, y_max, x_title, y_title, era_tag,
               right_margin=None, **kwargs):
    """
    cmsstyle canvas with "CMS Preliminary" at the top left, outside the frame,
    and "<lumi> (<energy>)" at the top right.
    """
    CMS.SetExtraText("Preliminary")
    CMS.SetLumi(config.PROJECTED_LUMI, run=None, round_lumi=0)
    CMS.SetEnergy(0, unit=ENERGY_LABELS.get(era_tag, era_tag))
    canv = CMS.cmsCanvas(name, x_min, x_max, y_min, y_max, x_title, y_title,
                         iPos=0, **kwargs)
    if right_margin is not None:
        # cmsCanvas has no right-margin option, and it has already placed the
        # right-aligned lumi text against the old margin, so move that too
        canv.SetRightMargin(right_margin)
        for obj in canv.GetListOfPrimitives():
            if obj.InheritsFrom('TLatex') and obj.GetTextAlign() == 31:
                obj.SetX(1 - right_margin)
    return canv


def make_graph(x, y):
    # TGraph reads the raw buffers, so pandas column views must be made contiguous
    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    return ROOT.TGraph(len(x), x, y)


WEB_DIR = "/eos/user/s/skkwan/www/higgsino/studies/mll-MET-fit-2D/limits"


def save_canvas(canv, out_path):
    """Save canv as out_path (.png) and as a .pdf next to it, then close it."""
    base = os.path.splitext(out_path)[0]
    paths = [base + '.png', base + '.pdf']
    CMS.UpdatePad(canv)
    for path in paths:
        canv.SaveAs(path)
    canv.Close()
    return paths


# ══════════════════════════════════════════════════════════════════════════════
#  Cross section helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_xsec(era_tag):
    """
    Per-mchi xsec_2d, xsec_1d [pb] for era_tag from signal_config.json.
    run2run3 is the luminosity-weighted run2/run3 average (config.signal_xsec).
    """
    rows = [{'mchi':    p['mchi'],
             'xsec_2d': config.signal_xsec(p, era_tag, '2d'),
             'xsec_1d': config.signal_xsec(p, era_tag, '1d')}
            for p in config.load_signal_config()]
    df = pd.DataFrame(rows).astype({'xsec_2d': float, 'xsec_1d': float})
    df = df.dropna(subset=['xsec_2d', 'xsec_1d'], how='all').drop_duplicates(subset=['mchi'])
    if df.empty:
        need = " (needs both run2 and run3)" if era_tag == 'run2run3' else ""
        raise ValueError(f"No {era_tag} cross sections in signal_config.json{need}")
    return df


def add_xsec(df_lim, model, era_tag):
    df_xsec = load_xsec(era_tag)
    df = df_lim.merge(df_xsec, on='mchi', how='left')
    xsec_col = f'xsec_{config.model_xsec_mode(model)}'
    df['sigma_ref_pb'] = df[xsec_col] * config.model_br(model)
    for band in ['exp', 'exp_m1', 'exp_p1', 'exp_m2', 'exp_p2']:
        df[f'sigma_{band}_pb'] = df[band] * df['sigma_ref_pb']
    return df


def find_csv(era_tag, model, mlsp=None):
    """Auto-find limits CSV for a given era and model (and mlsp, as run_combine.py names it)."""
    # search in era subdir first, then limits root
    for d in [os.path.join(config.LIMITS_DIR, era_tag), config.LIMITS_DIR]:
        pattern = os.path.join(d, f"limits_{era_tag}_{model}_mlsp{mlsp}.csv")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  1D Brazil band
# ══════════════════════════════════════════════════════════════════════════════

def plot_brazil(df, mlsp_val, model, era_tag, out_path):
    sub = df[df['mlsp'] == mlsp_val].sort_values('mchi').copy()
    if len(sub) < 2:
        print(f"  1D: not enough points for mlsp={mlsp_val}")
        return

    mchi         = sub['mchi'].values.astype(float)
    sigma_theory = sub['sigma_ref_pb'].values
    s_exp        = sub['sigma_exp_pb'].values
    s_exp_m1     = sub['sigma_exp_m1_pb'].values
    s_exp_p1     = sub['sigma_exp_p1_pb'].values
    s_exp_m2     = sub['sigma_exp_m2_pb'].values
    s_exp_p2     = sub['sigma_exp_p2_pb'].values

    all_vals = np.concatenate([s_exp_m2, s_exp_p2, sigma_theory])
    # log scale: the curves fill the lower 55% of the frame, the labels and legend sit above
    y_min = max(1e-4, all_vals.min() / 5)
    y_max = y_min * (all_vals.max() / y_min) ** (1 / 0.55)
    canv = cms_canvas('c_brazil', mchi.min() - 25, mchi.max() + 25, y_min, y_max,
                      'm(#tilde{#chi}^{0}_{1}) [GeV]', '#sigma(#tilde{#chi}^{0}_{1} #tilde{#chi}^{0}_{1} #rightarrow ZH) [pb]',
                      era_tag, square=False, yTitOffset=1.3, extraSpace=0.04)
    canv.SetLogy()

    # bands are closed polygons: out along the upper edge, back along the lower edge
    mchi_loop = np.concatenate([mchi, mchi[::-1]])
    g_2sig = make_graph(mchi_loop, np.concatenate([s_exp_p2, s_exp_m2[::-1]]))
    g_1sig = make_graph(mchi_loop, np.concatenate([s_exp_p1, s_exp_m1[::-1]]))
    g_exp  = make_graph(mchi, s_exp)
    g_th   = make_graph(mchi, sigma_theory)
    # the line colour only shows as the legend box outline
    CMS.cmsObjectDraw(g_2sig, 'F', FillColor=CMS.kLimit95, LineColor=CMS.kLimit95)
    CMS.cmsObjectDraw(g_1sig, 'F', FillColor=CMS.kLimit68, LineColor=CMS.kLimit68)
    CMS.cmsObjectDraw(g_exp,  'L', LineColor=ROOT.kBlack, LineStyle=ROOT.kDashed, LineWidth=3)
    CMS.cmsObjectDraw(g_th,   'L', LineColor=ROOT.kRed, LineWidth=3)

    x_right = 1 - canv.GetRightMargin() - 0.04
    y_top   = 1 - canv.GetTopMargin() - 0.04
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.SetTextAlign(33)
    latex.DrawLatex(x_right, y_top, ('GMSB, assuming' if model == 'gmsb' else 'Higgsino-Bino, assuming') + f' BR(#tilde{{#chi}}^{{0}}_{{1}} #tilde{{#chi}}^{{0}}_{{1}} #rightarrow ZH) = {config.model_br(model):.0%}, ')
    latex.DrawLatex(x_right, y_top - 0.06, f'm_{{LSP}} = {mlsp_val} GeV, 2018 MC scaled up')

    leg = CMS.cmsLeg(x_right - 0.32, y_top - 0.38, x_right, y_top - 0.13, textSize=0.04)
    leg.AddEntry(g_exp,  'Expected limit', 'L')
    leg.AddEntry(g_1sig, 'Expected #pm1#sigma', 'F')
    leg.AddEntry(g_2sig, 'Expected #pm2#sigma', 'F')
    leg.AddEntry(g_th,   'Theory', 'L')

    canv.RedrawAxis()  # the bands cover the frame ticks
    paths = save_canvas(canv, out_path)
    print(f"  Saved 1D: {', '.join(paths)}")


# ══════════════════════════════════════════════════════════════════════════════
#  2D exclusion contour
# ══════════════════════════════════════════════════════════════════════════════

def plot_2d_exclusion(df, model, era_tag, out_path):
    mchi_pts = df['mchi'].values.astype(float)
    mlsp_pts = df['mlsp'].values.astype(float)

    sigma_lim_fb    = df['sigma_exp_pb'].values    * 1000.0
    sigma_lim_m1fb  = df['sigma_exp_m1_pb'].values * 1000.0
    sigma_lim_p1fb  = df['sigma_exp_p1_pb'].values * 1000.0
    sigma_lim_m2fb  = df['sigma_exp_m2_pb'].values * 1000.0
    sigma_lim_p2fb  = df['sigma_exp_p2_pb'].values * 1000.0
    sigma_theory_fb = df['sigma_ref_pb'].values    * 1000.0

    if len(mchi_pts) < 4:
        print("  2D: not enough points"); return

    vmin_fb = max(10 ** np.floor(np.log10(sigma_lim_fb.min())), 0.1)
    vmax_fb = 10 ** np.ceil(np.log10(sigma_lim_fb.max()))

    mNLSP_range = np.linspace(mchi_pts.min(), mchi_pts.max(), 400)
    mLSP_range = np.linspace(0, mchi_pts.max(), 400)
    MX, MY = np.meshgrid(mNLSP_range, mLSP_range)

    def interp_log(vals):
        z = griddata((mchi_pts, mlsp_pts), np.log10(vals),
                     (MX, MY), method='cubic')
        z[MY >= MX - 5] = np.nan
        return z

    Z_lim  = interp_log(sigma_lim_fb)
    Z_th   = interp_log(sigma_theory_fb)
    Z_excl    = Z_lim  - Z_th
    Z_excl_m1 = interp_log(sigma_lim_m1fb) - Z_th
    Z_excl_p1 = interp_log(sigma_lim_p1fb) - Z_th
    Z_excl_m2 = interp_log(sigma_lim_m2fb) - Z_th
    Z_excl_p2 = interp_log(sigma_lim_p2fb) - Z_th

    mchi_uniq = np.array(sorted(df['mchi'].unique()), dtype=float)
    mlsp_uniq = np.array(sorted(df['mlsp'].unique()), dtype=float)
    step_mchi = np.diff(mchi_uniq).mean() if len(mchi_uniq) > 1 else 50.0
    step_mlsp = np.diff(mlsp_uniq).mean() if len(mlsp_uniq) > 1 else 50.0
    mchi_edges = np.append(mchi_uniq, mchi_uniq[-1] + step_mchi)
    mlsp_edges = np.append(mlsp_uniq, mlsp_uniq[-1] + step_mlsp)
    # y axis ends 200 GeV above the top edge of the highest mlsp row with a z value
    mlsp_top = df.loc[np.isfinite(df['sigma_exp_pb']), 'mlsp'].max()
    y_max = mlsp_edges[np.searchsorted(mlsp_uniq, mlsp_top) + 1] + 200


    print(">>>>", mchi_pts)
    canv = cms_canvas('c_2d', mchi_pts.min(), mchi_pts.max() + step_mchi,
                      0, y_max,
                      'm_{NLSP} [GeV]', 'm_{LSP} [GeV]',
                      era_tag, square=False, with_z_axis=True, right_margin=0.19,
                      yTitOffset=1.1, extraSpace=0.03)
    canv.SetLogz()
    CMS.SetCMSPalette()

    # one cell per mass point, with its lower edges at (mchi, mlsp)
    h_lim = ROOT.TH2D('h_lim', '', len(mchi_uniq), mchi_edges, len(mlsp_uniq), mlsp_edges)
    mchi_idx = {v: i + 1 for i, v in enumerate(mchi_uniq)}
    mlsp_idx = {v: i + 1 for i, v in enumerate(mlsp_uniq)}
    for _, row in df.iterrows():
        # COLZ leaves cells below the minimum blank, so clip them to the lowest colour
        h_lim.SetBinContent(mchi_idx[row['mchi']], mlsp_idx[row['mlsp']],
                            max(row['sigma_exp_pb'] * 1000.0, vmin_fb))
    h_lim.SetMinimum(vmin_fb)
    h_lim.SetMaximum(vmax_fb)
    h_lim.SetContour(255)
    h_lim.GetZaxis().SetTitle('Cross section upper limit (95% CL) [fb]')
    h_lim.GetZaxis().SetTitleSize(0.045)
    h_lim.GetZaxis().SetLabelSize(0.045)
    h_lim.GetZaxis().SetTitleOffset(1.1)
    CMS.cmsObjectDraw(h_lim, 'COLZ')
    r = canv.GetRightMargin()
    CMS.UpdatePalettePosition(h_lim, canv, X1=1 - r + 0.01, X2=1 - r + 0.04)

    graphs = []
    leg_entries = []
    # style 9 is a long dash
    for label, lw, ls, Zs in [('Expected median',     3, ROOT.kSolid,  [Z_excl]),
                              ('Expected #pm1#sigma', 2, 9,            [Z_excl_m1, Z_excl_p1]),
                              ('Expected #pm2#sigma', 1, ROOT.kDotted, [Z_excl_m2, Z_excl_p2])]:
        style = dict(LineColor=ROOT.kRed, LineWidth=lw, LineStyle=ls)
        for Z in Zs:
            for line in contour_generator(MX, MY, np.ma.masked_invalid(Z)).lines(0):
                # the PNG renderer restarts the dash pattern on every segment, so
                # the ~1 px segments of the fine grid draw as solid; thin them out
                line = np.vstack([line[:-1:6], line[-1:]])
                graphs.append(make_graph(line[:, 0], line[:, 1]))
                CMS.cmsObjectDraw(graphs[-1], 'L', **style)
        # style-only graph for the legend, since a contour can be absent from the plot range
        g_leg = ROOT.TGraph()
        CMS.setRootObjectProperties(g_leg, **style)
        graphs.append(g_leg)
        leg_entries.append((g_leg, label))

    xsec_lbl = '#sigma_{1D}' if model == 'gmsb' else '#sigma_{2D}'
    x_left  = canv.GetLeftMargin() + 0.03
    x_right = 1 - canv.GetRightMargin() - 0.03
    y_top   = 1 - canv.GetTopMargin() - 0.03

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.03)
    latex.SetTextAlign(13)
    latex.DrawLatex(x_left, y_top, ('GMSB, assuming' if model == 'gmsb' else 'Higgsino-Bino, assuming') + f' BR(#tilde{{#chi}}^{{0}}_{{1}} #tilde{{#chi}}^{{0}}_{{1}} #rightarrow ZH) = {config.model_br(model):.0%}')

    # drawn after the contours, and opaque, so it stays readable over the colour map
    leg = CMS.cmsLeg(x_left, y_top - 0.21, x_left + 0.24, y_top - 0.06, textSize=0.03)
    leg.SetFillStyle(1001)
    leg.SetBorderSize(0)
    for g_leg, label in leg_entries:
        leg.AddEntry(g_leg, label, 'L')

    canv.RedrawAxis()  # the colour map covers the frame ticks
    paths = save_canvas(canv, out_path)
    print(f"  Saved 2D: {', '.join(paths)}")


# ══════════════════════════════════════════════════════════════════════════════
#  BF exclusion (ATLAS-style) -- DEAD CODE for this single-process channel.
#  Kept only because plot_bf_exclusion below still references it; neither is
#  reachable from main() anymore (see the --plot choices), since a BF scan
#  requires branching-fraction mixing between two signal processes, which
#  doesn't exist for a pure TChiZH grid. Safe to delete once run_bf_scan.py
#  (also inapplicable now) is removed.
# ══════════════════════════════════════════════════════════════════════════════

def find_crossing(mchi_vals, mu_vals):
    mchi_vals = np.array(mchi_vals)
    mu_vals   = np.array(mu_vals)
    mask = np.isfinite(mu_vals)
    mchi_vals = mchi_vals[mask]
    mu_vals   = mu_vals[mask]
    if len(mu_vals) < 2:
        return np.nan
    if np.all(mu_vals < 1):
        return np.nan
    if np.all(mu_vals > 1):
        return np.nan
    for i in range(len(mu_vals) - 1):
        if (mu_vals[i] - 1) * (mu_vals[i+1] - 1) <= 0:
            t = (1.0 - mu_vals[i]) / (mu_vals[i+1] - mu_vals[i])
            return mchi_vals[i] + t * (mchi_vals[i+1] - mchi_vals[i])
    return np.nan


def plot_bf_exclusion(era_tag, xsec_mode, lumi_label, out_dir):
    pattern = os.path.join(out_dir, f"limits_{era_tag}_bf*_{xsec_mode}xsec.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        print(f"  BF plot: no CSVs found matching {pattern}")
        return
    print(f"  BF plot: found {len(csv_files)} BF scan CSVs")

    frames = []
    for f in csv_files:
        df_tmp = pd.read_csv(f)
        df_tmp = df_tmp[(df_tmp['status'] == 'ok') & (df_tmp['mlsp'] == 0)]
        frames.append(df_tmp)

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        print("  BF plot: no OK results with mlsp=0"); return

    df = add_xsec(df, xsec_mode, era_tag)

    bf_vals = np.sort(df['bf'].unique()).astype(float)
    bf_pct  = bf_vals * 100.0

    if len(bf_vals) < 2:
        print(f"  BF plot: need ≥2 BF points (got {len(bf_vals)})"); return

    mchi_exp    = []
    mchi_exp_m1 = []
    mchi_exp_p1 = []

    for bf in bf_vals:
        sub  = df[df['bf'] == bf].sort_values('mchi')
        mchi = sub['mchi'].values.astype(float)
        mchi_exp   .append(find_crossing(mchi, sub['exp'].values))
        mchi_exp_m1.append(find_crossing(mchi, sub['exp_m1'].values))
        mchi_exp_p1.append(find_crossing(mchi, sub['exp_p1'].values))

    mchi_exp    = np.array(mchi_exp)
    mchi_exp_m1 = np.array(mchi_exp_m1)
    mchi_exp_p1 = np.array(mchi_exp_p1)

    print(f"  BF [%]   mchi_m1   mchi_exp   mchi_p1")
    for i, bf in enumerate(bf_vals):
        print(f"  {bf*100:5.1f}%   "
              f"{mchi_exp_m1[i]:>8.1f}   {mchi_exp[i]:>9.1f}   {mchi_exp_p1[i]:>8.1f}")

    from scipy.interpolate import interp1d

    # interpolate all curves onto a dense common BF grid for smooth fills
    bf_dense = np.linspace(bf_pct.min(), 100., 500)

    def interp_crossing(bf_pct_arr, mchi_arr):
        ok = np.isfinite(mchi_arr)
        if ok.sum() < 2:
            return np.full_like(bf_dense, np.nan)
        f = interp1d(bf_pct_arr[ok], mchi_arr[ok],
                     kind='linear', bounds_error=False, fill_value=np.nan)
        return f(bf_dense)

    mchi_exp_dense    = interp_crossing(bf_pct, mchi_exp)
    mchi_exp_m1_dense = interp_crossing(bf_pct, mchi_exp_m1)
    mchi_exp_p1_dense = interp_crossing(bf_pct, mchi_exp_p1)

    fig, ax = plt.subplots(figsize=(8, 6))

    def valid(arr):
        return np.isfinite(arr)

    # ±1σ band (yellow) — always fill min to max to avoid crossing artifacts
    mchi_lo   = np.minimum(mchi_exp_m1_dense, mchi_exp_p1_dense)
    mchi_hi   = np.maximum(mchi_exp_m1_dense, mchi_exp_p1_dense)
    both_ok   = np.isfinite(mchi_lo) & np.isfinite(mchi_hi)
    if both_ok.any():
        ax.fill_betweenx(bf_dense[both_ok],
                         mchi_lo[both_ok],
                         mchi_hi[both_ok],
                         color="#FFD966", alpha=0.9, zorder=2,
                         label=r"$\pm 1\,\sigma_\mathrm{exp}$ band")

    # median expected line
    exp_ok = valid(mchi_exp_dense)
    if exp_ok.any():
        ax.plot(mchi_exp_dense[exp_ok], bf_dense[exp_ok],
                color="black", lw=2.0, ls="dashed", zorder=6,
                label="Expected limit")

    # ±1σ lines (dotted)
    m1_dense_ok = valid(mchi_exp_m1_dense)
    p1_dense_ok = valid(mchi_exp_p1_dense)
    if m1_dense_ok.any():
        ax.plot(mchi_exp_m1_dense[m1_dense_ok], bf_dense[m1_dense_ok],
                color="black", lw=1.0, ls="dotted", zorder=5)
    if p1_dense_ok.any():
        ax.plot(mchi_exp_p1_dense[p1_dense_ok], bf_dense[p1_dense_ok],
                color="black", lw=1.0, ls="dotted", zorder=5)

    all_mchi = np.concatenate([mchi_exp, mchi_exp_m1, mchi_exp_p1])
    all_mchi = all_mchi[np.isfinite(all_mchi)]
    xlo = np.nanmin(mchi_exp) if valid(mchi_exp).any() else 100
    xlo = max(0, xlo)
    xhi = all_mchi.max() + 50 if len(all_mchi) else 600

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(bf_pct.min(), 100.)
    ax.set_xlabel(r"$m(\tilde{\chi}_1^0)$  [GeV]", fontsize=13, labelpad=6)
    ax.set_ylabel(r"$\mathcal{B}(\tilde{\chi}_1^0 \to h\tilde{G})$  [%]",
                  fontsize=13, labelpad=6)
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, right=True, labelsize=11)

    proc = (r"$pp \to \tilde{\chi}_1^0\tilde{\chi}_1^0,\;"
            r"\tilde{\chi}_1^0 \to h\tilde{G}$ or $Z\tilde{G}$")
    ax.set_title(proc, fontsize=11, pad=8)

    ax.text(0.04, 0.97, "CMS", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")
    ax.text(0.04, 0.90, lumi_label, transform=ax.transAxes, fontsize=10, va="top")
    ax.text(0.04, 0.84, "All limits at 95% CL",
            transform=ax.transAxes, fontsize=10, va="top")

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"bf_exclusion_{era_tag}_{xsec_mode}xsec.png")
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved BF exclusion: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Plot limits")
    parser.add_argument("--csv",    default=None,
                        help="Limits CSV from run_combine.py (auto-found if --era given)")
    parser.add_argument("--era",    default=None,
                        help="Era label (used to auto-find CSV)")
    parser.add_argument("--m",      dest="model", choices=config.MODELS, required=True,
                        help="'gmsb': mlsp=1, sigma_1D; 'bino': sigma_2D, --mlsp selects the "
                             "limits CSV and the 1D Brazil band")
    parser.add_argument("--mlsp",   type=int, default=None,
                        help="mlsp value (gmsb: 1; bino: required for --plot 1d/both and for "
                             "auto-finding the CSV)")
    parser.add_argument("--plot",   choices=["1d", "2d", "both"], default="both")
    parser.add_argument("--outdir", default=WEB_DIR,
                        help=f"Output directory (default: {WEB_DIR})")
    args = parser.parse_args()
    if args.model == 'gmsb':
        try:
            args.mlsp = config.model_mlsp('gmsb', args.mlsp)
        except ValueError as e:
            parser.error(str(e))

    # ── auto-find CSV if not given ────────────────────────────────────────────
    if not args.csv:
        if not args.era:
            print("ERROR: --csv or --era required")
            sys.exit(1)
        args.csv = find_csv(args.era, args.model, args.mlsp)
        if not args.csv:
            print(f"ERROR: could not find limits_{args.era}_{args.model}_mlsp{args.mlsp}.csv "
                  f"in {config.LIMITS_DIR}")
            sys.exit(1)
        print(f"Auto-found CSV: {args.csv}")

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found"); sys.exit(1)

    df_raw = pd.read_csv(args.csv)
    df_raw = df_raw[df_raw['status'] == 'ok'].copy()
    if df_raw.empty:
        print("ERROR: no successful results in CSV"); sys.exit(1)

    era_tag    = args.era 
    xsec_mode  = args.model
    out_dir    = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    print("Loading cross sections from signal_config.json...", flush=True)
    df = add_xsec(df_raw, xsec_mode, era_tag)
    n_excl = (df['exp'] <= 1).sum()
    print(f"Loaded {len(df)} mass points: {n_excl} expected excluded (mu<1)")

    if args.plot in ("1d", "both"):
        if args.mlsp is not None:
            mlsp_val = args.mlsp
        else:
            counts   = df.groupby('mlsp').size()
            mlsp_val = int(counts[counts >= 3].index.min())
        print(f"\nMaking 1D Brazil band (mlsp={mlsp_val})...", flush=True)
        out_1d = os.path.join(out_dir,
                              f"asymptoticLimits_{era_tag}_{xsec_mode}_mlsp{mlsp_val}.png")
        plot_brazil(df, mlsp_val, xsec_mode, era_tag, out_1d)

    if args.plot in ("2d", "both"):
        print(f"\nMaking 2D exclusion map...", flush=True)
        out_2d = os.path.join(out_dir,
                              f"exclusion2d_{era_tag}_{xsec_mode}.png")
        plot_2d_exclusion(df, xsec_mode, era_tag, out_2d)

    print("\nDone.")


if __name__ == "__main__":
    main()