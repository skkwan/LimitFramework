#!/usr/bin/env python3
"""
make_datacards.py
Writes Combine datacards for each (mchi, mlsp, era) mass point.

Single signal process (TChiZH), single signal region (SR) 

Model (--m):
  gmsb: implies mlsp = 1 GeV; sigma_1D (the signal samples were produced with it,
        the full higgsino); the datacard signal rate is BR(ZH) = 0.5 (config.GMSB_BR_ZH)
  bino: requires --mlsp; yields scaled by sigma_2D/sigma_1D per mass point
        (sigma_2D is the Higgsino-Bino cross section); --mlsp all writes every
        mlsp (the full 2D plane)
  both: gmsb (mlsp = 1 GeV) and bino (at --mlsp, required)

--mlsp all with gmsb is the same as mlsp = 1 GeV, its only mlsp.

Cards go to combine/datacards/<era>/<model>/.

Usage:
    python3 make_datacards.py --era run2 --m gmsb
    python3 make_datacards.py --era run2 --m bino --mlsp 100
    python3 make_datacards.py --era run2 --m bino --mlsp all
    python3 make_datacards.py --era run2 --m both --mlsp 100
    python3 make_datacards.py --era run2 --m gmsb --mchi 650 --force
"""

import os
import sys
import json
import argparse
import pandas as pd
import ROOT

ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))
import config


# ── Era settings ──────────────────────────────────────────────────────────────

LUMI_UNC  = {'run2': 1.02, 'run3': 1.02, 'run2run3': 1.02}


# ── Signal config helpers ─────────────────────────────────────────────────────

def load_signal_config():
    """Load signal_config.json as a DataFrame, one row per (mchi, mlsp)."""
    df = pd.DataFrame(config.load_signal_config())
    dup = df[df.duplicated(subset=['mchi', 'mlsp'], keep=False)]
    if not dup.empty:
        raise ValueError("duplicate (mchi, mlsp) entries in signal_config.json: "
                         f"{sorted(set(zip(dup['mchi'], dup['mlsp'])))}")
    return df


def get_yield(df, mchi, mlsp):
    """
    Raw signal yield from signal_config.json for one (mchi, mlsp), at
    config.PROJECTED_LUMI. 0.0 if the mass point or its yield is missing/null.
    """
    rows = df[(df['mchi'] == mchi) & (df['mlsp'] == mlsp)]
    if rows.empty or config.SIGNAL_YIELD_COL not in rows:
        return 0.0
    val = rows[config.SIGNAL_YIELD_COL].iloc[0]
    return 0.0 if pd.isna(val) else float(val)


def get_xsec_ratio(df, mchi, era_tag):
    """
    sigma_2d / sigma_1d for this mchi and era.
    For --m bino: multiply the yield by this to convert from
    sigma_1D-based (as produced) to sigma_2D-based expected events.
    Returns None if signal_config.json has no (positive) cross sections for this mchi and era.
    """
    for _, point in df[df['mchi'] == mchi].iterrows():
        xsec_2d = config.signal_xsec(point, era_tag, '2d')
        xsec_1d = config.signal_xsec(point, era_tag, '1d')
        if xsec_2d is not None and xsec_1d is not None:
            return xsec_2d / xsec_1d if xsec_2d > 0 and xsec_1d > 0 else None
    return None


# ── Path helpers ──────────────────────────────────────────────────────────────

def ws_path(mchi, mlsp, era_tag):
    return os.path.abspath(config.combined_ws(mchi, mlsp, era_tag))


# ── Workspace helpers ─────────────────────────────────────────────────────────

def fixed_shape_params(ws_file, era_tag, mchi, mlsp):
    """
    Shape parameters of the signal and background pdfs in the per-mass-point
    workspace (make_workspaces.py fixes them all), for stating in the
    datacard. ratio_peaking_{era} is excluded: it has its own param line.

    Returns [(label, [(name, value), ...]), ...], one entry per pdf.
    """
    groups = [
        (f"sig_pdf_mchi{mchi}_mlsp{mlsp}_{era_tag}", "sig",
         "m(ll) Crystal Ball; MET is a fixed spline (no parameters)"),
        (f"bkg_peak_{era_tag}", "bkg peaking",
         "m(ll) Crystal Ball (fit_zpeak.py) x MET Gumbel"),
        (f"bkg_nonpeak_{era_tag}", "bkg non-peaking",
         "m(ll) exponential x MET Gumbel"),
    ]
    f = ROOT.TFile(ws_file, 'READ')
    ws = f.Get(f"ws_{era_tag}")
    if not ws:
        f.Close()
        raise RuntimeError(f"Workspace ws_{era_tag} not found in {ws_file}")
    observables = ROOT.RooArgSet(ws.var("mll"), ws.var("met"))
    out, seen = [], {f"ratio_peaking_{era_tag}"}
    for pdf_name, short, desc in groups:
        pdf = ws.pdf(pdf_name)
        if not pdf:
            f.Close()
            raise RuntimeError(f"PDF {pdf_name} not found in {ws_file}")
        params = sorted((p.GetName(), p.getVal()) for p in pdf.getParameters(observables)
                        if p.GetName() not in seen)
        seen.update(name for name, _ in params)
        out.append((f"{short} ({pdf_name}): {desc}", params))
    f.Close()
    return out


# ── Datacard writer ───────────────────────────────────────────────────────────

def write_datacard(mchi, mlsp, era_tag, df, model, force=False):
    out_path = config.datacard_path(mchi, mlsp, era_tag, model)

    if not force and os.path.exists(out_path):
        print(f"  EXISTS (skip): {os.path.basename(out_path)}")
        return True

    r_raw = get_yield(df, mchi, mlsp)
    rate_factor = get_xsec_ratio(df, mchi, era_tag) if model == 'bino' else config.model_br(model)
    if rate_factor is None:
        print(f"  SKIP mchi={mchi} mlsp={mlsp}: no {era_tag} xsec_2d/xsec_1d in signal_config.json (needed for --m bino, with xsec_2d > 0)")
        return False
    r = r_raw * rate_factor

    if r <= 0:
        print(f"  SKIP mchi={mchi} mlsp={mlsp}: yield zero or missing in signal_config.json")
        return False

    # Signal pdf + yield (sig_pdf_..._norm) and background model live in one
    # workspace per mass point (make_workspaces.py)
    ws_file = ws_path(mchi, mlsp, era_tag)
    ws_name = f"ws_{era_tag}"
    sig_pdf = f"sig_pdf_mchi{mchi}_mlsp{mlsp}_{era_tag}"

    if not os.path.exists(ws_file):
        print(f"  SKIP mchi={mchi} mlsp={mlsp}: missing workspace {ws_file}")
        return False

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # CombineHarvester's datacard parser joins the shapes path onto the
    # datacard's own directory even when it's already absolute, so the
    # shapes lines must use paths relative to out_dir.
    ws_rel = os.path.relpath(ws_file, out_dir)

    bkg_rate = sum(config.BKG_YIELD.values())

    # Central value of the ratio of peaking to total background, from the MC fit
    json_path = config.bkg_fit_json(era_tag)
    if not os.path.exists(json_path):
        print(f"  SKIP mchi={mchi} mlsp={mlsp}: missing {json_path} (run fit_background.py)")
        return False
    with open(json_path) as fj:
        ratio = json.load(fj)['ratio_peaking_mc']
    ratio_unc = config.RATIO_PEAKING_REL_UNC * ratio

    shape_groups = fixed_shape_params(ws_file, era_tag, mchi, mlsp)
    name_w = max(len(name) for _, params in shape_groups for name, _ in params)

    with open(out_path, 'w') as f:
        f.write(f"# Higgsino Z(ll)H(bb) datacard\n")
        f.write(f"# mchi={mchi} mlsp={mlsp} era={era_tag} model={model}\n")
        f.write(f"# Yields at {config.PROJECTED_LUMI:g} fb-1: sig={r:.4f}  bkg={bkg_rate:.6g} "
                f"(peaking {config.BKG_YIELD['peaking']:g}, non-peaking {config.BKG_YIELD['nonpeak']:g})\n")
        f.write(f"# Run: combine -M AsymptoticLimits <card> -t -1 --expectSignal 0\n")

        f.write(f"\nimax 1\njmax *\nkmax *\n\n---\n\n")
        f.write(f"shapes sig      SR  {ws_rel}  {ws_name}:{sig_pdf}\n")
        f.write(f"shapes bkg      SR  {ws_rel}  {ws_name}:bkg_total_{era_tag}\n")
        f.write(f"shapes data_obs SR  {ws_rel}  {ws_name}:data_obs\n")
        f.write(f"\n---\n\n")
        f.write(f"bin         SR\n")
        f.write(f"observation -1\n")
        f.write(f"\n---\n\n")
        f.write(f"bin      SR   SR\n")
        f.write(f"process  sig  bkg\n")
        f.write(f"process  0    1\n")
        # sig: the yield is the workspace's sig_pdf_..._norm (from signal_config.json, set
        # in make_workspaces.py); the rate here multiplies it.
        # bkg: no _norm in the workspace, so the rate is the yield itself.
        f.write(f"rate     {rate_factor:.6f}  {bkg_rate:.6g}\n")
        f.write(f"\n---\n\n")
        # Luminosity: log-normal on both yields
        f.write(f"lumi_{era_tag}  lnN  {LUMI_UNC[era_tag]:g}  {LUMI_UNC[era_tag]:g}\n")
        # Total background yield: unconstrained, floats in the fit (multiplies the rate above)
        f.write(f"bkg_norm_{era_tag}  rateParam  SR  bkg  1  [0,10]\n")
        # Ratio of peaking to total background inside bkg_total: Gaussian
        # constraint with RATIO_PEAKING_REL_UNC relative width, range limited
        # to [0,1] (otherwise Combine uses mean +/- 4 sigma, which goes below 0)
        f.write(f"ratio_peaking_{era_tag}  param  {ratio:.3g}  {ratio_unc:.3g}  [0,1]\n")


    print(f"  Wrote: {os.path.basename(out_path)}  [rate_factor={rate_factor:.3g}  sig={r:.3f}  "
          f"bkg={bkg_rate:.6g} (floating)  lumi lnN {LUMI_UNC[era_tag]:g}  "
          f"ratio_peaking={ratio:.3g} +/- {ratio_unc:.3g}  "
          f"fixed shape params={sum(len(p) for _, p in shape_groups)}]")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def mlsp_arg(s):
    """--mlsp value: an int, or 'all' (every mlsp; gmsb: mlsp=1)."""
    return s if s == "all" else int(s)


def main():
    parser = argparse.ArgumentParser(description="Write Combine datacards")
    parser.add_argument("--era",   choices=["run2", "run3", "run2run3", "both"],
                        default="both")
    parser.add_argument("--m",     dest="model", choices=[*config.MODELS, "both"], required=True,
                        help="'gmsb': mlsp=1 GeV, sigma_1D, yields as-is; "
                             "'bino': requires --mlsp, yields scaled by sigma_2D/sigma_1D; "
                             "'both': gmsb (mlsp=1 GeV) and bino (requires --mlsp)")
    parser.add_argument("--mchi",  type=int, default=None)
    parser.add_argument("--mlsp",  type=mlsp_arg, default=None,
                        help="'all' (bino) writes every mlsp, the full 2D plane; "
                             "with model --m both, applies to bino only (gmsb always uses mlsp=1 GeV)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    models = config.MODELS if args.model == "both" else [args.model]
    # model -> mlsp ('all' = no mlsp selection); with --m both, --mlsp is
    # bino's (gmsb's is implied)
    model_mlsps = {}
    try:
        for model in models:
            if model == "gmsb" and (args.model == "both" or args.mlsp == "all"):
                model_mlsps[model] = config.GMSB_MLSP
            elif model == "bino" and args.mlsp == "all":
                model_mlsps[model] = "all"
            else:
                model_mlsps[model] = config.model_mlsp(model, args.mlsp)
    except ValueError as e:
        parser.error(str(e))

    eras = ["run2", "run3"] if args.era == "both" else [args.era]

    print("Loading signal_config.json...", flush=True)
    df = load_signal_config()
    print(f"  {len(df)} mass points", flush=True)

    for model, model_mlsp in model_mlsps.items():
        print(f"\n##### Settings: m={model} mlsp={model_mlsp} #####", flush=True)
        mass_points = sorted(config.SIGNAL_GRID)
        if args.mchi is not None:
            mass_points = [(mc, ml) for mc, ml in mass_points if mc == args.mchi]
        if model_mlsp != "all":
            mass_points = [(mc, ml) for mc, ml in mass_points if ml == model_mlsp]
        print(f"Mass points: {len(mass_points)}", flush=True)

        for era in eras:
            print(f"\n=== era={era} m={model} ===")
            n_ok = n_skip = n_fail = 0
            for mchi, mlsp in mass_points:
                try:
                    ok = write_datacard(mchi, mlsp, era, df, model=model, force=args.force)
                    if ok:  n_ok   += 1
                    else:   n_skip += 1
                except Exception as e:
                    print(f"  ERROR mchi={mchi} mlsp={mlsp}: {e}")
                    import traceback; traceback.print_exc()
                    n_fail += 1

            print(f"\n  Done: {n_ok} written  {n_skip} skipped  {n_fail} failed")
            if n_ok > 0 and mass_points:
                card_dir = config.datacard_dir(era, model)
                print(f"\n  Validate:")
                print(f"    cd {card_dir}")
                example = config.datacard_path(mass_points[0][0], mass_points[0][1], era, model)
                print(f"    ValidateDatacards.py {os.path.basename(example)}")
                print(f"    combine -M AsymptoticLimits {os.path.basename(example)} "
                      f"-t -1 --expectSignal 0")


if __name__ == "__main__":
    main()
