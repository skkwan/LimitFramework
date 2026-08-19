#!/usr/bin/env python3
"""
make_inputs.py
Step 1 of the pipeline.

Adapted from 2DFit_higgsinos/hbb_zll/individual_pdf_fits/reformat.py: raw
ntuples are split into separate mm and ee trees (weight_nominal_mm /
weight_nominal_ee), which this script harmonizes into a single weight_nominal branch
and hadds together. Raw branch names assumed (matching hbb_zll): "m_ll",
"met", "mbb", "weight_nominal_mm" / "weight_nominal_ee". Adjust
RAW_TREE_NAME below if your ntuples differ.

Background handling mirrors reformat.py's bkg mode: samples are
looked up in config.BKG_SAMPLES (one raw ntuple subdirectory per sample,
grouped by physics process) and each group is hadded separately depending
on whether it's in config.PEAKING_SAMPLES -- producing one "peaking"
(real-Z/ttZ-like) and one "non-peaking" (everything else) background-MC
n-tuple per year, feeding the 2-component background model in
fit_background.py / fit_zpeak.py.

The "zpeak" mode mirrors 2DFit_higgsinos/hbb_zll/zpeak_fit/reformat_zPeak.py:
it hadds the dedicated Z-peak control-region (CRZ) MC ntuples for the same
DYJets + TTZ_peak sample groups into one backgrounds_CRZ_Zpeak_{year}.root
per year, consumed by fit_zpeak.py in place of the SR-selected "peaking"
background-MC category.

Output ntuples use tree name "tree" (the name every other script in this
framework expects), with branches:
  signal:               mll, met, weight_nominal
  background (each cat): mll, met, weight_nominal
  zpeak CRZ:             mll, weight_nominal

Usage:
    python3 scripts/make_inputs.py --mode sig --years 2018
    python3 scripts/make_inputs.py --mode bkg --years 2018
    python3 scripts/make_inputs.py --mode zpeak --years 2018
    python3 scripts/make_inputs.py --mode all                 # all configured years
    python3 scripts/make_inputs.py --mode sig --mass-points 300,0 500,100
"""

import argparse
import glob
import os
import sys

import ROOT

sys.path.insert(0, os.path.dirname(__file__))
import config

ROOT.gROOT.SetBatch(True)

RAW_TREE_NAME = "event_tree"
OUT_TREE_NAME = "tree"
OUT_BRANCHES_SIG  = ROOT.std.vector('string')(["mll", "met", "weight_nominal"])
OUT_BRANCHES_DATA = ROOT.std.vector('string')(["mll", "met"])
OUT_BRANCHES_ZPEAK = ROOT.std.vector('string')(["mll", "weight_nominal"])


def sr_cut_expr():
    """Any remaining SR cuts not applied to the input n-tuples"""
    return (# f"mbb > {config.MBB_LO} && mbb < {config.MBB_HI} && "
            f"met > 0 ")


def harmonize_and_filter(files, weight_expr, out_path):
    """
    Build an RDataFrame from `files`, rename m_ll->mll, define weight_nominal (if
    weight_expr is given, else data with no weight branch), apply the SR
    cut, and snapshot the reduced branch list to out_path.

    Returns out_path, or None if no files were given.
    """
    if not files:
        return None
    df = ROOT.RDataFrame(RAW_TREE_NAME, files).Define("mll", "m_ll")
    branches = OUT_BRANCHES_DATA
    if weight_expr is not None:
        df = df.Define("weight_nominal", weight_expr)
        branches = OUT_BRANCHES_SIG
    df.Filter(sr_cut_expr()).Snapshot(OUT_TREE_NAME, out_path, branches)
    return out_path


def hadd(out_path, in_paths):
    in_paths = [p for p in in_paths if p]
    if not in_paths:
        return False
    cmd = f"hadd -f -j -k {out_path} " + " ".join(in_paths)
    print(f"  $ {cmd}")
    rc = os.system(cmd)
    return rc == 0


# ── Signal ────────────────────────────────────────────────────────────────────

def discover_signal_mass_points(year):
    base = config.RAW_SIGNAL_DIR.get(year)
    if not base:
        return []
    pattern = os.path.join(base, f"{config.SIGNAL_PROCESS}_*")
    pairs = []
    for d in sorted(glob.glob(pattern)):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)  # TChiZH_650_1
        parts = name.split("_")
        try:
            m1, m2 = int(parts[-2]), int(parts[-1])
        except (ValueError, IndexError):
            continue
        pairs.append((m1, m2))
    return pairs


def make_signal_ntuple(year, mchi, mlsp, force=False):
    out_path = config.ntuple_sig_path(year, mchi, mlsp)
    if not force and os.path.exists(out_path):
        print(f"  EXISTS (skip): {os.path.basename(out_path)}")
        return True

    base = config.RAW_SIGNAL_DIR[year]
    mp_name = f"{config.SIGNAL_PROCESS}_{mchi}_{mlsp}"
    mp_dir = os.path.join(base, mp_name)

    os.system(f"mkdir -p $(dirname {out_path})")

    mm_files = sorted(glob.glob(f"{mp_dir}/snapshot_{mp_name}_*mm_SR_mll_MET_fit_scheme.root"))
    ee_files = sorted(glob.glob(f"{mp_dir}/snapshot_{mp_name}_*ee_SR_mll_MET_fit_scheme.root"))
    if not mm_files and not ee_files:
        print(f"  SKIP: no raw ntuples found in {mp_dir}")
        return False

    tmp_mm = out_path.replace(".root", "_mm_tmp.root")
    tmp_ee = out_path.replace(".root", "_ee_tmp.root")
    fixed = []
    fixed.append(harmonize_and_filter(mm_files, "weight_nominal_mm", tmp_mm))
    fixed.append(harmonize_and_filter(ee_files, "weight_nominal_ee", tmp_ee))
    fixed = [f for f in fixed if f]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = hadd(out_path, fixed)
    for f in fixed:
        os.remove(f)

    if ok:
        print(f"  Saved {out_path}")
    return ok


# ── Background (peaking vs. non-peaking) ───────────────────────────────────────

def discover_bkg_files(year, category):
    """
    Glob raw background-MC ntuples for one year/category ('peaking' or
    'nonpeak'), grouping samples exactly like reformat.py: every group in
    config.BKG_SAMPLES is classified as peaking or non-peaking depending on
    membership in config.PEAKING_SAMPLES, and every sample in a matching
    group contributes its mm/ee files.
    """
    base = config.RAW_BKG_DIR.get(year)
    if not base:
        return [], []

    mm_files, ee_files = [], []
    for group, samples in config.BKG_SAMPLES.items():
        is_peaking = group in config.PEAKING_SAMPLES
        if (category == "peaking") != is_peaking:
            continue
        for s in samples:
            mm_files += sorted(glob.glob(f"{base}/{s}/snapshot*mm_SR_mll_MET_fit_scheme.root"))
            ee_files += sorted(glob.glob(f"{base}/{s}/snapshot*ee_SR_mll_MET_fit_scheme.root"))
    return mm_files, ee_files


def make_bkg_ntuple(year, category, force=False):
    """Hadd one background category ('peaking' or 'nonpeak') for one year."""
    out_path = config.bkg_category_ntuple_path(year, category)
    if not force and os.path.exists(out_path):
        print(f"  EXISTS (skip): {os.path.basename(out_path)}")
        return True

    if not config.RAW_BKG_DIR.get(year):
        print(f"  SKIP: no RAW_BKG_DIR configured for {year} (fill it in config.py)")
        return False

    mm_files, ee_files = discover_bkg_files(year, category)
    if not mm_files and not ee_files:
        print(f"  SKIP: no raw '{category}' background ntuples found for {year}")
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tmp_mm = out_path.replace(".root", "_mm_tmp.root")
    tmp_ee = out_path.replace(".root", "_ee_tmp.root")
    fixed = []
    fixed.append(harmonize_and_filter(mm_files, "weight_nominal_mm", tmp_mm))
    fixed.append(harmonize_and_filter(ee_files, "weight_nominal_ee", tmp_ee))
    fixed = [f for f in fixed if f]

    ok = hadd(out_path, fixed)
    for f in fixed:
        os.remove(f)

    if ok:
        print(f"  Saved {out_path}")
    return ok


# ── Z-peak control region (CRZ) ──────────────────────────────────────────────

def discover_crz_files(year):
    """
    Glob raw Z-peak control-region MC ntuples for one year: the same
    DYJets + TTZ_peak sample groups as the "peaking" background category,
    but selected with a CRZ (not SR) cut. Mirrors
    2DFit_higgsinos/hbb_zll/zpeak_fit/reformat_zPeak.py's `samples` dict.
    """
    base = config.RAW_BKG_CRZ_DIR.get(year)
    if not base:
        return [], []

    mm_files, ee_files = [], []
    for group in config.PEAKING_SAMPLES:
        for s in config.BKG_SAMPLES[group]:
            mm_files += sorted(glob.glob(f"{base}/{s}/snapshot*_mm_CRZ.root"))
            ee_files += sorted(glob.glob(f"{base}/{s}/snapshot*_ee_CRZ.root"))
    return mm_files, ee_files


def harmonize_crz(files, weight_expr, out_path):
    """
    Like harmonize_and_filter, but for the raw CRZ ntuples: no SR cut is
    applied (the CRZ selection is already baked into the input files by
    the ntuple production step), matching reformat_zPeak.py.
    """
    if not files:
        return None
    df = ROOT.RDataFrame(RAW_TREE_NAME, files).Define("mll", "m_ll") \
             .Define("weight_nominal", weight_expr)
    df.Snapshot(OUT_TREE_NAME, out_path, OUT_BRANCHES_ZPEAK)
    return out_path


def make_zpeak_crz_ntuple(year, force=False):
    """Hadd the Z-peak CRZ background ntuple (DY + ttZ_peak) for one year."""
    out_path = config.zpeak_crz_ntuple_path(year)
    if not force and os.path.exists(out_path):
        print(f"  EXISTS (skip): {os.path.basename(out_path)}")
        return True

    if not config.RAW_BKG_CRZ_DIR.get(year):
        print(f"  SKIP: no RAW_BKG_CRZ_DIR configured for {year} (fill it in config.py)")
        return False

    mm_files, ee_files = discover_crz_files(year)
    if not mm_files and not ee_files:
        print(f"  SKIP: no raw CRZ ntuples found for {year}")
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tmp_mm = out_path.replace(".root", "_mm_tmp.root")
    tmp_ee = out_path.replace(".root", "_ee_tmp.root")
    fixed = []
    fixed.append(harmonize_crz(mm_files, "weight_nominal_mm", tmp_mm))
    fixed.append(harmonize_crz(ee_files, "weight_nominal_ee", tmp_ee))
    fixed = [f for f in fixed if f]

    ok = hadd(out_path, fixed)
    for f in fixed:
        os.remove(f)

    if ok:
        print(f"  Saved {out_path}")
    return ok


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process raw ntuples for the limit framework")
    parser.add_argument("--mode", choices=["sig", "bkg", "zpeak", "all"], default="all")
    parser.add_argument("--years", nargs="+", default=None,
                        help="Years to process (default: all years with a configured raw dir)")
    parser.add_argument("--mass-points", nargs="+", default=None, metavar="MCHI,MLSP",
                        help="Restrict signal to these 'mchi,mlsp' pairs (default: discover all)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mass_points = None
    if args.mass_points:
        mass_points = [tuple(int(x) for x in mp.split(",")) for mp in args.mass_points]

    if args.mode in ("sig", "all"):
        years = args.years or sorted(config.RAW_SIGNAL_DIR.keys())
        print(f"=== Signal ntuples: years={years} ===")
        n_ok = n_fail = 0
        for year in years:
            points = mass_points or discover_signal_mass_points(year)
            print(f"-- {year}: {len(points)} mass point(s) --")
            for mchi, mlsp in points:
                ok = make_signal_ntuple(year, mchi, mlsp, force=args.force)
                n_ok += ok
                n_fail += not ok
        print(f"Signal: {n_ok} ok, {n_fail} failed/skipped\n")

    if args.mode in ("bkg", "all"):
        years = args.years or sorted(config.RAW_BKG_DIR.keys())
        print(f"=== Background ntuples (peaking / non-peaking): years={years} ===")
        if not years:
            print("  No years configured in config.RAW_BKG_DIR -- nothing to do.")
        n_ok = n_fail = 0
        for year in years:
            for category in ("peaking", "nonpeak"):
                ok = make_bkg_ntuple(year, category, force=args.force)
                n_ok += ok
                n_fail += not ok
        print(f"Background: {n_ok} ok, {n_fail} failed/skipped")

    if args.mode in ("zpeak", "all"):
        years = args.years or sorted(config.RAW_BKG_CRZ_DIR.keys())
        print(f"=== Z-peak CRZ background ntuple: years={years} ===")
        if not years:
            print("  No years configured in config.RAW_BKG_CRZ_DIR -- nothing to do.")
        n_ok = n_fail = 0
        for year in years:
            ok = make_zpeak_crz_ntuple(year, force=args.force)
            n_ok += ok
            n_fail += not ok
        print(f"Z-peak CRZ: {n_ok} ok, {n_fail} failed/skipped")


if __name__ == "__main__":
    main()
