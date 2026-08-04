#!/usr/bin/env python3
"""
plot.py
Produces CMS-style limit plots:

  1D Brazil band (--plot 1d):
     x: m(χ̃₁⁰) [GeV]    y: cross section [pb]   (log scale)
     Green band ±1σ, yellow band ±2σ, blue dashed = expected limit median
     Theory cross section overlaid

  2D exclusion contour (--plot 2d):
     Color map: expected cross section upper limit [fb]  (log scale)
     Red contours: expected median, ±1σ, ±2σ exclusion boundaries

  BF exclusion (--plot bf):
     ATLAS-style B(χ̃₁⁰ → hG̃) vs m(χ̃₁⁰) contour.
     Reads all limits_{era}_bf*_{xsec}xsec.csv produced by run_bf_scan.py.
     Does NOT require --csv (scans the limits dir automatically).

Usage:
    python3 scripts/plot.py --era run2 --bf 1.0 --xsec 1d --plot 1d --mlsp 0
    python3 scripts/plot.py --era run2 --bf 0.5 --xsec 1d --plot 1d --mlsp 0
    python3 scripts/plot.py --era run2 --bf 1.0 --xsec 2d --plot 2d
    python3 scripts/plot.py --plot bf --era run2run3 --xsec 1d
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
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
import config

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

LUMI_LABELS = {
    'run2':     r'137.61 fb$^{-1}$ (13 TeV)',
    'run3':     r'282.44 fb$^{-1}$ (13.6 TeV)',
    'run2run3': r'420.05 fb$^{-1}$ (13+13.6 TeV)',
}


# ══════════════════════════════════════════════════════════════════════════════
#  Cross section helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_xsec(era_tag):
    if not os.path.exists(config.SIGNAL_CSV):
        raise FileNotFoundError(f"signal_master.csv not found: {config.SIGNAL_CSV}")
    df = pd.read_csv(config.SIGNAL_CSV)

    if era_tag == 'run2run3':
        # Combined-era reference xsec: previously this fell through to the
        # 'else' branch below and silently reused run3's (2022EE) xsec alone
        # for the whole combined result, which doesn't correctly cancel
        # against the combined datacard's mixed run2+run3 nominal rate.
        # Approximation used here: xsec_weighted = (xsec_run2*lumi_run2 +
        # xsec_run3*lumi_run3) / lumi_total, i.e. luminosity-weighted,
        # assuming eff_run2 ~ eff_run3 (checked to be reasonably close,
        # though not exactly equal -- see diag notes on run2 vs run3
        # signal efficiency). This is an approximation, not an exact
        # cancellation like the single-era case.
        df_run2 = df[df['year'] == '2018'][['mchi', 'xsec_2d', 'xsec_1d']].drop_duplicates(subset=['mchi'])
        df_run3 = df[df['year'] == '2022EE'][['mchi', 'xsec_2d', 'xsec_1d']].drop_duplicates(subset=['mchi'])
        if df_run2.empty or df_run3.empty:
            raise ValueError("No rows found for year=2018 or year=2022EE in signal_master.csv "
                              "(needed to build combined-era reference xsec)")
        merged = df_run2.merge(df_run3, on='mchi', suffixes=('_run2', '_run3'))

        lumi_run2 = config.LUMI_RUN2
        lumi_run3 = config.LUMI_RUN3
        lumi_total = lumi_run2 + lumi_run3

        for col in ['xsec_2d', 'xsec_1d']:
            merged[col] = (merged[f'{col}_run2'] * lumi_run2 +
                          merged[f'{col}_run3'] * lumi_run3) / lumi_total

        return merged[['mchi', 'xsec_2d', 'xsec_1d']]

    ref_year = '2018' if era_tag == 'run2' else '2022EE'
    df_year = df[df['year'] == ref_year][['mchi', 'xsec_2d', 'xsec_1d']].drop_duplicates(subset=['mchi'])
    if df_year.empty:
        raise ValueError(f"No rows found for year={ref_year} in signal_master.csv")
    return df_year


def add_xsec(df_lim, xsec_mode, era_tag):
    df_xsec = load_xsec(era_tag)
    df = df_lim.merge(df_xsec, on='mchi', how='left')
    xsec_col = 'xsec_1d' if xsec_mode == '1d' else 'xsec_2d'
    df['sigma_ref_pb'] = df[xsec_col]
    for band in ['exp', 'exp_m1', 'exp_p1', 'exp_m2', 'exp_p2']:
        df[f'sigma_{band}_pb'] = df[band] * df['sigma_ref_pb']
    return df


def find_csv(era_tag, bf, xsec_mode):
    """Auto-find limits CSV for a given era, bf, xsec."""
    bf_str = f"bf{bf:.2f}".replace('.', 'p')
    # search in era subdir first, then limits root
    for d in [os.path.join(config.LIMITS_DIR, era_tag), config.LIMITS_DIR]:
        pattern = os.path.join(d, f"limits_{era_tag}_{bf_str}_{xsec_mode}xsec.csv")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  1D Brazil band
# ══════════════════════════════════════════════════════════════════════════════

def plot_brazil(df, mlsp_val, xsec_mode, era_tag, lumi_label, out_path):
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

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(mchi, s_exp_m2, s_exp_p2,
                    color='#FFEE44', label=r'Expected $\pm2\sigma$', zorder=2)
    ax.fill_between(mchi, s_exp_m1, s_exp_p1,
                    color='#44BB44', label=r'Expected $\pm1\sigma$', zorder=3)
    ax.plot(mchi, s_exp,        'b--', lw=2, label='Expected limit', zorder=4)
    ax.plot(mchi, sigma_theory, 'r-',  lw=2, label='Theory',         zorder=5)

    ax.set_yscale('log')
    ax.set_xlabel(r'$m(\tilde{\chi}_1^0)$ [GeV]')
    ax.set_ylabel('Cross section [pb]')
    ax.set_xlim(mchi.min() - 25, mchi.max() + 25)
    all_vals = np.concatenate([s_exp_m2, s_exp_p2, sigma_theory])
    ax.set_ylim(max(1e-4, all_vals.min() / 5), all_vals.max() * 5)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', alpha=0.3, ls=':')

    bf_val   = float(df['bf'].iloc[0])
    xsec_lbl = r'$\sigma_{\rm 1D}$' if xsec_mode == '1d' else r'$\sigma_{\rm 2D}$'
    ax.set_title(f'Higgsino, $\\mathcal{{B}}(\\tilde{{\\chi}}\\to H)={bf_val:.2f}$, '
                 f'$m_{{\\rm LSP}}={mlsp_val}$ GeV, {xsec_lbl}', fontsize=11)

    ax.text(0.04, 0.97, "CMS", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")
    ax.text(0.04, 0.90, lumi_label, transform=ax.transAxes, fontsize=10, va="top")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved 1D: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  2D exclusion contour
# ══════════════════════════════════════════════════════════════════════════════

def plot_2d_exclusion(df, xsec_mode, era_tag, lumi_label, out_path):
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

    mchi_range = np.linspace(mchi_pts.min(), mchi_pts.max(), 400)
    mlsp_range = np.linspace(0, mchi_pts.max(), 400)
    MX, MY = np.meshgrid(mchi_range, mlsp_range)

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
    Z_pixel = np.full((len(mlsp_uniq), len(mchi_uniq)), np.nan)
    mchi_idx = {v: i for i, v in enumerate(mchi_uniq)}
    mlsp_idx = {v: i for i, v in enumerate(mlsp_uniq)}
    for _, row in df.iterrows():
        Z_pixel[mlsp_idx[row['mlsp']], mchi_idx[row['mchi']]] = \
            np.log10(row['sigma_exp_pb'] * 1000.0)

    step_mchi = np.diff(mchi_uniq).mean() if len(mchi_uniq) > 1 else 50.0
    step_mlsp = np.diff(mlsp_uniq).mean() if len(mlsp_uniq) > 1 else 50.0
    mchi_edges = np.append(mchi_uniq, mchi_uniq[-1] + step_mchi)
    mlsp_edges = np.append(mlsp_uniq, mlsp_uniq[-1] + step_mlsp)
    ME, MLE = np.meshgrid(mchi_edges, mlsp_edges)

    import colorsys
    def desaturate_cmap(cmap_name, sat=0.65):
        cmap = plt.get_cmap(cmap_name)
        cols = cmap(np.linspace(0, 1, 256))
        new_cols = []
        for r, g, b, a in cols:
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            s *= sat
            r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
            new_cols.append((r2, g2, b2, a))
        return mcolors.LinearSegmentedColormap.from_list('jet_muted', new_cols)

    norm = mcolors.LogNorm(vmin=vmin_fb, vmax=vmax_fb)
    cmap = desaturate_cmap('jet', sat=0.65)

    fig, ax = plt.subplots(figsize=(8, 7))
    pcm = ax.pcolormesh(ME, MLE, 10**Z_pixel,
                        cmap=cmap, norm=norm, shading='flat', zorder=1)

    contour_kw = dict(zorder=5)
    for Z, lw, ls in [(Z_excl, 2.0, 'solid'), (Z_excl_m1, 1.3, 'dashed'),
                      (Z_excl_p1, 1.3, 'dashed'), (Z_excl_m2, 0.8, 'dotted'),
                      (Z_excl_p2, 0.8, 'dotted')]:
        try:
            ax.contour(MX, MY, Z, levels=[0],
                       colors='red', linewidths=lw, linestyles=ls, **contour_kw)
        except Exception:
            pass

    cb = plt.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label('Cross section upper limit (95% CL) [fb]', fontsize=12)
    tick_vals = [10**p for p in range(int(np.floor(np.log10(vmin_fb))),
                                       int(np.ceil(np.log10(vmax_fb))) + 1)]
    cb.set_ticks(tick_vals)
    cb.set_ticklabels([f'$10^{{{int(np.log10(t))}}}$' if np.log10(t) != 0
                       else '1' for t in tick_vals])

    ax.set_xlim(mchi_pts.min(), mchi_pts.max() + step_mchi)
    ax.set_ylim(0, mchi_pts.max() * 0.8)
    ax.set_xlabel(r'$m(\tilde{\chi}_1^0)_{\rm NLSP}$ [GeV]', fontsize=14)
    ax.set_ylabel(r'$m_{\rm LSP}$ [GeV]', fontsize=14)

    bf_val   = float(df['bf'].iloc[0])
    xsec_lbl = r'$\sigma_{\rm 1D}$' if xsec_mode == '1d' else r'$\sigma_{\rm 2D}$'
    ax.text(0.03, 0.97,
            f'pp → higgsino → HH/ZH + LSP\n$\\mathcal{{B}}(\\tilde{{\\chi}}\\to H)={bf_val:.2f}$,  {xsec_lbl}',
            transform=ax.transAxes, fontsize=10, va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4'),
            zorder=10)

    legend_elements = [
        Line2D([0,1],[0,0], color='red', lw=2.0, ls='solid',  label='Exp. median'),
        Line2D([0,1],[0,0], color='red', lw=1.3, ls='dashed', label=r'Exp. $\pm1\sigma$'),
        Line2D([0,1],[0,0], color='red', lw=0.8, ls='dotted', label=r'Exp. $\pm2\sigma$'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved 2D: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  BF exclusion (ATLAS-style)
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
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved BF exclusion: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Plot limits")
    parser.add_argument("--csv",    default=None,
                        help="Limits CSV from run_combine.py (auto-found if --era and --bf given)")
    parser.add_argument("--era",    default=None,
                        help="Era label (required for --plot bf; used to auto-find CSV)")
    parser.add_argument("--bf",     type=float, default=1.0,
                        help="B(chi->H) for auto-finding CSV (default: 1.0)")
    parser.add_argument("--xsec",   choices=["1d", "2d"], default="1d",
                        help="Xsec mode (default: 1d)")
    parser.add_argument("--mlsp",   type=int, default=None,
                        help="mlsp value for 1D Brazil band (default: smallest)")
    parser.add_argument("--plot",   choices=["1d", "2d", "bf", "both"], default="both")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    # ── BF mode: no CSV needed ────────────────────────────────────────────────
    if args.plot == "bf":
        if not args.era:
            print("ERROR: --era required for --plot bf"); sys.exit(1)
        xsec_mode  = args.xsec
        era_tag    = args.era
        lumi_label = LUMI_LABELS.get(era_tag, era_tag)
        out_dir    = args.outdir or os.path.join(config.LIMITS_DIR, era_tag)
        if not glob.glob(os.path.join(out_dir, f"limits_{era_tag}_bf*_{xsec_mode}xsec.csv")):
            out_dir = config.LIMITS_DIR
        print(f"Making BF exclusion plot  era={era_tag}  xsec={xsec_mode}", flush=True)
        plot_bf_exclusion(era_tag, xsec_mode, lumi_label, out_dir)
        print("\nDone.")
        return

    # ── 1d / 2d / both: auto-find CSV if not given ───────────────────────────
    if not args.csv:
        if not args.era:
            print("ERROR: --csv or (--era + --bf) required for --plot 1d/2d/both")
            sys.exit(1)
        args.csv = find_csv(args.era, args.bf, args.xsec)
        if not args.csv:
            bf_str = f"bf{args.bf:.2f}".replace('.', 'p')
            print(f"ERROR: could not find limits_{args.era}_{bf_str}_{args.xsec}xsec.csv "
                  f"in {config.LIMITS_DIR}")
            sys.exit(1)
        print(f"Auto-found CSV: {args.csv}")

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found"); sys.exit(1)

    df_raw = pd.read_csv(args.csv)
    df_raw = df_raw[df_raw['status'] == 'ok'].copy()
    if df_raw.empty:
        print("ERROR: no successful results in CSV"); sys.exit(1)

    era_tag    = args.era  or str(df_raw['era'].iloc[0])
    xsec_mode  = args.xsec or str(df_raw['xsec'].iloc[0])
    bf_val     = float(df_raw['bf'].iloc[0])
    bf_str     = f"bf{bf_val:.2f}".replace('.', 'p')
    lumi_label = LUMI_LABELS.get(era_tag, era_tag)
    out_dir    = args.outdir or os.path.dirname(args.csv)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading cross sections from signal_master.csv...", flush=True)
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
                              f"brazil_{era_tag}_{bf_str}_{xsec_mode}xsec_mlsp{mlsp_val}.png")
        plot_brazil(df, mlsp_val, xsec_mode, era_tag, lumi_label, out_1d)

    if args.plot in ("2d", "both"):
        print(f"\nMaking 2D exclusion map...", flush=True)
        out_2d = os.path.join(out_dir,
                              f"exclusion2d_{era_tag}_{bf_str}_{xsec_mode}xsec.png")
        plot_2d_exclusion(df, xsec_mode, era_tag, lumi_label, out_2d)

    print("\nDone.")


if __name__ == "__main__":
    main()