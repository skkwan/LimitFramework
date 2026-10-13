#!/usr/bin/env python3
"""
prepare_signal_json.py
Fills the yields in signal_config.json for every mass point found in the signal
ntuples of each year in config.SIGNAL_MC_YEARS:

  yield_{year}     = sum of weight_nominal in the fit range (config.MLL_LO/HI, MET_LO/HI),
                     one field per year with an ntuple for that mass point
  yield_run2       = sum of yield_{year} over all of config.SIGNAL_MC_YEARS (2016APV,
                     2016, 2017, 2018); only written if every one of them is available
  yield_450fb      = yield_run2 * PROJECTED_LUMI / LUMI_RUN2; the datacard signal rate.
                     Only written if yield_run2 is (otherwise the mass point is skipped)
  yield_run2_from2018only = yield_2018 * LUMI_RUN2 / LUMI['2018']
  yield_450fb_from2018only = yield_run2_from2018only * PROJECTED_LUMI / LUMI_RUN2

Each year's ntuples are read from config.NTUPLE_SIG_DIR/{year}; --sigdir YEAR=PATH
overrides one year's directory, which can have either layout:
  raw (condorHistogramming output): one {SIGNAL_PROCESS}_{mchi}_{mlsp}/
      subdirectory per mass point, with
      snapshot_*_channel_{mm,ee}_SR_mll_MET_fit_scheme.root (tree event_tree,
      branches m_ll, met, weight_nominal_{mm,ee}); the mm and ee sums are added.
      A missing channel file means no event passed in that channel.
  hadded (make_inputs.py output):
      snapshot_{SIGNAL_PROCESS}_{mchi}_{mlsp}_SR_mll_MET_fit_scheme.root
      (tree tree, branches mll, met, weight_nominal).

Every mass point in --sigdir is filled. Points not yet in signal_config.json
are added, with xsec_1d looked up by mchi in the luna-zhmet xsec headers
(run2: xsec.h, run3: xsec_13p6TeV.h; null if mchi is not listed).
xsec_2d (Higgsino-Bino, the "higgsinoBino_<mchi>" entries) is refreshed for
every mass point in signal_config.json; xsec_1d of existing points is left alone.
Points in signal_config.json without an ntuple in --sigdir are otherwise unchanged.

Usage:
    python3 scripts/prepare_signal_json.py
    python3 scripts/prepare_signal_json.py --sigdir 2018=/eos/cms/store/group/phys_susy/skkwan/condorHistogramming/2026-09-29-20h09m_2018-samples-SR_mll_MET_fit_scheme-only
"""

import os
import re
import sys
import glob
import json
import argparse
import ROOT

ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.dirname(__file__))
import config


FIT_CUT = (f"mll > {config.MLL_LO} && mll < {config.MLL_HI} && "
           f"met > {config.MET_LO} && met < {config.MET_HI}")

YEAR_YIELD_FIELDS = {
    f"yield_{y}": f"sum of weight_nominal in the {y} signal ntuple, fit range ({FIT_CUT}); absent if the mass point has no {y} ntuple"
    for y in config.SIGNAL_MC_YEARS
}
YIELD_FIELDS = {
    **YEAR_YIELD_FIELDS,
    "yield_run2": ("sum of yield_{year} over " + ", ".join(config.SIGNAL_MC_YEARS)
                   + "; absent unless all of them are available"),
    "yield_450fb": (f"yield_run2 * PROJECTED_LUMI/LUMI_RUN2 ({config.PROJECTED_LUMI:g}/{config.LUMI_RUN2:g}); "
                    "the datacard signal rate. absent if yield_run2 is (the mass point is then skipped)"),
    "yield_run2_from2018only": f"yield_2018 * LUMI_RUN2/LUMI['2018'] ({config.LUMI_RUN2:g}/{config.LUMI['2018']:g})",
    "yield_450fb_from2018only": (f"yield_run2_from2018only * PROJECTED_LUMI/LUMI_RUN2 "
                                 f"({config.PROJECTED_LUMI:g}/{config.LUMI_RUN2:g})"),
}

XSEC_DIR = '/afs/cern.ch/work/s/skkwan/public/zhmet/CMSSW_14_0_21/src/luna-zhmet/histogramming/helpers'
XSEC_HEADERS = {'run2': 'xsec.h', 'run3': 'xsec_13p6TeV.h'}
HB_PREFIX = 'higgsinoBino_'


def sig6(x):
    return float(f"{x:.6g}")


def read_signal_xsecs(header, prefix=''):
    """{mchi: xsec [pb]} from the {"<prefix><mchi>", <xsec>} entries of a luna-zhmet xsec header."""
    xsecs = {}
    with open(header) as f:
        for line in f:
            m = re.search(r'\{\s*"' + re.escape(prefix) + r'(\d+)"\s*,\s*([-+.\deE]+)\s*\}', line.split('//')[0])
            if m:
                xsecs[int(m[1])] = sig6(float(m[2]))
    return xsecs


def find_signal_inputs(sigdir):
    """
    {(mchi, mlsp): [(tree, files, weight_branch), ...]} for every mass point in
    sigdir, one entry per ntuple set to sum (raw: mm and ee; hadded: one file).
    """
    proc = config.SIGNAL_PROCESS
    inputs = {}
    for d in sorted(glob.glob(os.path.join(sigdir, f'{proc}_*_*'))):
        m = re.fullmatch(rf'{proc}_(\d+)_(\d+)', os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        sources = []
        for ch in ('mm', 'ee'):
            files = sorted(glob.glob(os.path.join(d, f'snapshot_*_channel_{ch}_SR_mll_MET_fit_scheme.root')))
            if files:
                sources.append(('event_tree', files, f'weight_nominal_{ch}'))
        inputs[(int(m[1]), int(m[2]))] = sources
    for path in glob.glob(os.path.join(sigdir, f'snapshot_{proc}_*_*_SR_mll_MET_fit_scheme.root')):
        m = re.fullmatch(rf'snapshot_{proc}_(\d+)_(\d+)_SR_mll_MET_fit_scheme\.root', os.path.basename(path))
        if m and (int(m[1]), int(m[2])) not in inputs:
            inputs[(int(m[1]), int(m[2]))] = [('tree', [path], 'weight_nominal')]
    return inputs


def book_yield(tree, files, weight):
    """Lazy sum of `weight` over `files` in the fit range."""
    df = ROOT.RDataFrame(tree, files)
    if not df.HasColumn('mll'):
        df = df.Define('mll', 'm_ll')
    return df.Filter(FIT_CUT).Sum(weight)


def print_bkg_yields(years=config.RUN2_YEARS):
    """
    Print the peaking and non-peaking background yields (sum of weight_nominal
    in the fit range) of the hadded background-MC ntuples
    (config.bkg_category_ntuple_path), per year, plus each year's projection
    to PROJECTED_LUMI for comparison with config.BKG_YIELD.
    """
    cats = ('peaking', 'nonpeak')
    booked = {(y, c): book_yield('tree', [config.bkg_category_ntuple_path(y, c)], 'weight_nominal')
              for y in years for c in cats
              if os.path.exists(config.bkg_category_ntuple_path(y, c))}
    if not booked:
        print(f"Background yields: no hadded background ntuples for {years}")
        return
    ROOT.RDF.RunGraphs(list(booked.values()))
    print(f"Background yields in the fit range (config.BKG_YIELD at {config.PROJECTED_LUMI:g} fb^-1: "
          f"{config.BKG_YIELD}):")
    for y in years:
        found = [c for c in cats if (y, c) in booked]
        if not found:
            print(f"  {y}: no hadded background ntuples")
            continue
        scale = config.PROJECTED_LUMI / config.LUMI[y]
        msg = "  ".join(f"{c}={sig6(booked[(y, c)].GetValue())} "
                        f"({sig6(booked[(y, c)].GetValue() * scale)} at {config.PROJECTED_LUMI:g} fb^-1)"
                        for c in found)
        print(f"  {y}: {msg}")


def main():
    parser = argparse.ArgumentParser(description="Fill signal yields in signal_config.json")
    parser.add_argument("--sigdir", nargs="+", default=[], metavar="YEAR=PATH",
                        help="override the signal ntuple directory of a year (default: "
                             "config.NTUPLE_SIG_DIR/YEAR); raw (per-mass-point "
                             "subdirectories) or hadded layout")
    parser.add_argument("--xsec-dir", default=XSEC_DIR,
                        help="luna-zhmet helpers directory with the xsec headers used for "
                             "mass points new to signal_config.json (default: %(default)s)")
    args = parser.parse_args()

    sigdirs = {y: os.path.join(config.NTUPLE_SIG_DIR, y) for y in config.SIGNAL_MC_YEARS}
    for item in args.sigdir:
        year, _, path = item.partition('=')
        if year not in sigdirs or not path:
            print(f"ERROR: --sigdir expects YEAR=PATH with YEAR in {config.SIGNAL_MC_YEARS}, got {item}")
            sys.exit(1)
        sigdirs[year] = path

    inputs = {}  # year -> {(mchi, mlsp): [(tree, files, weight_branch), ...]}
    for year, sigdir in sigdirs.items():
        if not os.path.isdir(sigdir):
            print(f"  SKIP {year}: {sigdir} is not a directory")
            continue
        found = find_signal_inputs(sigdir)
        for key in [k for k, sources in found.items() if not sources]:
            print(f"  SKIP {year} mchi={key[0]} mlsp={key[1]}: no snapshot files in its directory")
            del found[key]
        print(f"{year}: {len(found)} mass points in {sigdir}")
        if found:
            inputs[year] = found
    if not inputs:
        print(f"ERROR: no {config.SIGNAL_PROCESS} mass points found")
        sys.exit(1)
    mass_points = sorted(set().union(*(set(v) for v in inputs.values())))

    # book everything first so RunGraphs runs all event loops in one go
    booked = {(year, key): [book_yield(*src) for src in sources]
              for year, found in inputs.items() for key, sources in found.items()}
    ROOT.RDF.RunGraphs([r for rs in booked.values() for r in rs])

    with open(config.SIGNAL_CONFIG) as f:
        d = json.load(f)
    points = {(pt['mchi'], pt['mlsp']): pt for pt in d['mass_points']}

    xsec_tables = {era: read_signal_xsecs(os.path.join(args.xsec_dir, h))
                   for era, h in XSEC_HEADERS.items()}
    # Higgsino-Bino (2D) xsecs; a header without them (run3 currently) gives null
    xsec_2d_tables = {era: read_signal_xsecs(os.path.join(args.xsec_dir, h), HB_PREFIX)
                      for era, h in XSEC_HEADERS.items()}
    n_new = 0
    for mchi, mlsp in mass_points:
        pt = points.get((mchi, mlsp))
        if pt is None:
            xsec_1d = {era: table.get(mchi) for era, table in xsec_tables.items()}
            pt = {'mchi': mchi, 'mlsp': mlsp, 'xsec_2d': {}, 'xsec_1d': xsec_1d}
            points[(mchi, mlsp)] = pt
            n_new += 1
        pt['xsec_2d'] = {era: table.get(mchi) for era, table in xsec_2d_tables.items()}
        yields = {y: sum(r.GetValue() for r in booked[(y, (mchi, mlsp))])
                  for y in config.SIGNAL_MC_YEARS if (y, (mchi, mlsp)) in booked}
        prev = (pt.get('yield_run2'), pt.get('yield_450fb'))
        # re-insert so the yields sit together after the xsecs
        for k in YIELD_FIELDS:
            pt.pop(k, None)
        pt.pop('yield_run2_total', None)  # earlier name of yield_run2
        for y, v in yields.items():
            pt[f'yield_{y}'] = sig6(v)
        msg = "  ".join(f"{y}={pt[f'yield_{y}']}" for y in yields)
        if len(yields) == len(config.SIGNAL_MC_YEARS):
            pt['yield_run2'] = sig6(sum(yields.values()))
            pt['yield_450fb'] = sig6(pt['yield_run2'] * config.PROJECTED_LUMI / config.LUMI_RUN2)
            msg += f"  run2={pt['yield_run2']}  450fb={pt['yield_450fb']}  previous (run2, 450fb)={prev}"
        else:
            msg += "  (not all years: yield_run2 and yield_450fb not filled)"
        if '2018' in yields:
            yrun2_18 = yields['2018'] * config.LUMI_RUN2 / config.LUMI['2018']
            pt.update(yield_run2_from2018only=sig6(yrun2_18),
                      yield_450fb_from2018only=sig6(yrun2_18 * config.PROJECTED_LUMI / config.LUMI_RUN2))
        print(f"  ({mchi:4d},{mlsp:4d})  {msg}")

    for mchi, mlsp in sorted(set(points) - set(mass_points)):
        print(f"  UNCHANGED mchi={mchi} mlsp={mlsp}: no ntuple in any year")
    print(f"{len(mass_points)} mass points filled ({n_new} new), "
          f"{len(points) - len(mass_points)} left unchanged")

    d['mass_points'] = [points[k] for k in sorted(points)]
    other_fields = {k: v for k, v in d.get('fields', {}).items() if k not in YIELD_FIELDS}
    d['fields'] = {**YIELD_FIELDS, **other_fields}

    with open(config.SIGNAL_CONFIG, 'w') as f:
        json.dump(d, f, indent=2)
        f.write('\n')
    print(f"Wrote {config.SIGNAL_CONFIG}")

    print_bkg_yields()


if __name__ == "__main__":
    main()
