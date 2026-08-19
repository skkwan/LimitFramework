"""
config.py — Central configuration for the limit framework.
All other scripts import from here. Change things in one place.

Final state: higgsino -> Z(ll) H(bb) + MET (TChiZH), single signal process,
single signal region. Observables are m(ll) and MET (see
2DFit_higgsinos/hbb_zll for the reference implementation this framework's
fit machinery is adapted from).
"""

import os
import glob

# ── Root directory (always resolves to limitframework/, wherever you run from) ──
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_WWW_DIR = "/eos/user/s/skkwan/www/higgsino/studies/mll-MET-fit-2D/combine_shapes"

# ===========================================================================
# Era definitions
# ===========================================================================

RUN2_YEARS = ['2016', '2016APV', '2017', '2018']
RUN3_YEARS = ['2022', '2022EE', '2023', '2023BPix', '2024', '2025']
ALL_YEARS  = RUN2_YEARS + RUN3_YEARS

# Years for which signal MC exists (TODO: fix 2025)
SIGNAL_YEAR_MAP = {y: y for y in ALL_YEARS}
SIGNAL_YEAR_MAP['2025'] = '2024'

def signal_year_for(year):
    """Return the year whose signal ntuples/shapes to use for a given data year."""
    return SIGNAL_YEAR_MAP[year]

def era_years(era):
    """Return the list of years for a given era string."""
    if era == 'run2':
        return RUN2_YEARS
    elif era == 'run3':
        return RUN3_YEARS
    elif era == 'run2run3':
        return ALL_YEARS
    else:
        raise ValueError(f"Unknown era '{era}'. Choose: run2, run3, run2run3")

def era_label(years):
    """Infer a label string from a list of years."""
    if set(years) == set(ALL_YEARS):
        return 'run2run3'
    if set(years) == set(RUN2_YEARS):
        return 'run2'
    if set(years) == set(RUN3_YEARS):
        return 'run3'
    return '_'.join(years)


LUMI = {
    '2016':     19.51,
    '2016APV':  16.80,
    '2017':     41.48,
    '2018':     59.83,
    '2022':      7.98,
    '2022EE':   26.67,
    '2023':     17.79,
    '2023BPix':  9.45,
    '2024':    109.95,
    '2025':    110.60,
}

LUMI_RUN2    = sum(LUMI[y] for y in RUN2_YEARS)           # 137.62 fb⁻¹
LUMI_RUN3    = sum(LUMI[y] for y in RUN3_YEARS)           # 282.44 fb⁻¹
LUMI_RUN2RUN3 = LUMI_RUN2 + LUMI_RUN3                     # 420 fb⁻¹

# ===========================================================================
# Signal process 
# ===========================================================================

SIGNAL_PROCESS = 'TChiZH'

# ===========================================================================
# Signal region: m(bb) window (only useful if samples are not already cut on m_bb)
# ===========================================================================
MBB_LO, MBB_HI = 100., 140.

# ===========================================================================
# Fit variables
# ===========================================================================
# m(ll) and MET

MLL_LO, MLL_HI = 60.,  120.
MET_LO, MET_HI = 200., 1200.

N_MET_BINS = 50
N_MLL_BINS = 60

# Minimum number of (unweighted) entries required to attempt a fit
MIN_EVENTS_FOR_FIT = 10

# Fixed background normalization (bkg_total_{era}_norm), per era.
# The background shape+norm are fully determined by fit_background.py's
# fit to background MC, so this is set constant rather than left floating.
BKG_NORM = {
    'run2': 24.0,
    'run3': 48.0,
}

# ===========================================================================
# Directory structure
# ===========================================================================

NTUPLE_SIG_DIR = '/eos/cms/store/group/phys_susy/skkwan/toys/inputs_to_fits'
NTUPLE_BKG_DIR = '/eos/cms/store/group/phys_susy/skkwan/toys/inputs_to_fits'
COMBINE_DIR   = os.path.join(ROOT_DIR, 'combine')
FITS_DIR      = os.path.join(COMBINE_DIR, 'fits')
WS_DIR        = os.path.join(COMBINE_DIR, 'workspaces')
CARDS_DIR     = os.path.join(COMBINE_DIR, 'datacards')
LIMITS_DIR    = os.path.join(COMBINE_DIR, 'limits')

# Signal master CSV (produced by make_signal_csv.py)
SIGNAL_CSV = os.path.join(ROOT_DIR, 'signal_master.csv')

# ===========================================================================
# Raw ntuple locations -- INPUT to make_inputs.py.
# Fill in per year as ntuples become available. Each directory is expected
# to contain, per mass point (signal) or per process (data), files matching
# snapshot*_{mm,ee}_SR_mll_MET_fit_scheme.root -- mirroring the
# 2DFit_higgsinos/hbb_zll ntuple-production convention (separate mm/ee trees
# with weight_nominal_mm / weight_nominal_ee branches, harmonized into a
# single weight_nominal / w_lumi branch by make_inputs.py).
# ===========================================================================

RAW_SIGNAL_DIR = {
    '2018': '/eos/cms/store/group/phys_susy/skkwan/condorHistogramming/'
            '2026-06-06-00h18m-2018-sample-signal-points',
    # TODO: add remaining years once signal ntuples exist for them.
}

RAW_BKG_DIR = {
    # TODO: add remaining years
    '2018': '/eos/cms/store/group/phys_susy/skkwan/condorHistogramming/'
            '2026-02-25-00h42m-2018-dataMC-with-SR-ntuples',
}

# Dedicated Z-peak control-region (CRZ) MC ntuples -- INPUT to make_inputs.py's
# zpeak mode. Same DYJets + TTZ_peak samples as PEAKING_SAMPLES, but selected
# with a CRZ (not SR) cut, matching 2DFit_higgsinos/hbb_zll/zpeak_fit's
# reformat_zPeak.py. Files match snapshot*_{mm,ee}_CRZ.root, raw branches
# m_ll / weight_nominal_mm / weight_nominal_ee in tree "event_tree".
RAW_BKG_CRZ_DIR = {
    '2018': '/eos/cms/store/group/phys_susy/skkwan/condorHistogramming/'
            '2026-04-09-01h05m-2018-CRZ-MC-only-for-fit',
}

# ===========================================================================
# Background MC sample classes -- INPUT to make_inputs.py's bkg mode.
# Mirrors 2DFit_higgsinos/hbb_zll/individual_pdf_fits/reformat.py: each
# sample lives in its own subdirectory under RAW_BKG_DIR[year], and every
# group is classified as either "peaking" (real Z / real ttZ -- peaks in
# m(ll) near the Z mass, same as signal) or left as "non-peaking"
# (everything else), feeding the 2-component background model.
# ===========================================================================

BKG_SAMPLES = {
    "DYJets": [
        "DYJetsToLL_M-50",
        "DYJetsToLL_M-50_HT-70to100",
        "DYJetsToLL_M-50_HT-100to200",
        "DYJetsToLL_M-50_HT-200to400",
        "DYJetsToLL_M-50_HT-400to600",
        "DYJetsToLL_M-50_HT-600to800",
        "DYJetsToLL_M-50_HT-800to1200",
        "DYJetsToLL_M-50_HT-1200to2500",
        "DYJetsToLL_M-50_HT-2500toInf",
    ],
    "WJets": [
        "WJetsToLNu",
    ],
    "ttbar": [
        "TTTo2L2Nu",
    ],
    "TTZ_peak": [
        "TTZToLLNuNu_M-10_peak_mll",
    ],
    "TTZ_nonpeak": [
        "TTZToLLNuNu_M-10_nonpeak_mll",
    ],
    "TTW": [
        "TTWJetsToLNu",
    ],
    "WH": [
        "WminusH_HToBB_WToLNu_M-125",
        "WplusH_HToBB_WToLNu_M-125",
    ],
    "ZH": [
        "ZH_HToBB_ZToLL_M-125",
    ],
    "WZ": [
        "WZTo3LNu",
        "WZTo2Q2L",
        "WZTo2Q2Nu",
    ],
    "WW": [
        "WWTo2L2Nu",
    ],
    "ZZ": [
        "ZZTo2L2Nu",
        "ZZTo2Q2L",
        "ZZTo2Q2Nu",
        "ZZTo4L",
    ],
}

PEAKING_SAMPLES = ["DYJets", "TTZ_peak"]

# ===========================================================================
# N-tuple file path helpers
# ===========================================================================

def ntuple_sig_path(year, mchi, mlsp):
    """Full path to one signal n-tuple"""
    return os.path.join(
        NTUPLE_SIG_DIR, str(year),
        f'snapshot_{SIGNAL_PROCESS}_{mchi}_{mlsp}_SR_mll_MET_fit_scheme.root'
    )

def ntuple_bkg_path(year):
    """Full path to one background n-tuple"""
    return os.path.join(NTUPLE_BKG_DIR, str(year), f'ntuple_data_{year}.root')

def bkg_category_ntuple_path(year, category):
    """
    Full path to one hadded, peaking/non-peaking-categorized background-MC
    n-tuple (produced by make_inputs.py's bkg mode). category: 'peaking' or
    'nonpeak'.
    """
    if category not in ('peaking', 'nonpeak'):
        raise ValueError("category must be 'peaking' or 'nonpeak'")
    return os.path.join(NTUPLE_BKG_DIR, str(year), f'background_{category}_{year}.root')

def bkg_category_files(category, years):
    """Return list of existing hadded background-MC n-tuples for one category."""
    paths = [bkg_category_ntuple_path(y, category) for y in years]
    return [p for p in paths if os.path.exists(p)]

def zpeak_crz_ntuple_path(year):
    """
    Full path to one hadded Z-peak control-region (CRZ) background-MC
    n-tuple (produced by make_inputs.py's zpeak mode), mirroring
    reformat_zPeak.py's backgrounds_CRZ_Zpeak_{year}.root.
    """
    return os.path.join(NTUPLE_BKG_DIR, str(year), f'backgrounds_CRZ_Zpeak_{year}.root')

def zpeak_crz_files(years):
    """Return list of existing hadded Z-peak CRZ background-MC n-tuples."""
    paths = [zpeak_crz_ntuple_path(y) for y in years]
    return [p for p in paths if os.path.exists(p)]

# ===========================================================================
# Signal mass grid — discovered from existing ntuple files
# ===========================================================================

def _discover_mass_points():
    """Scan the signal ntuple directory to find all (mchi, mlsp) pairs that
    exist, matching ntuple_sig_path's naming:
    snapshot_{SIGNAL_PROCESS}_{mchi}_{mlsp}_SR_mll_MET_fit_scheme.root"""
    pattern = os.path.join(NTUPLE_SIG_DIR, '*',
                           f'snapshot_{SIGNAL_PROCESS}_*_*_SR_mll_MET_fit_scheme.root')
    files = glob.glob(pattern)
    pairs = set()
    for f in files:
        base = os.path.basename(f).replace('.root', '')
        parts = base.split('_')  # ['snapshot', SIGNAL_PROCESS, mchi, mlsp, 'SR', ...]
        try:
            mchi = int(parts[2])
            mlsp = int(parts[3])
            pairs.add((mchi, mlsp))
        except (ValueError, IndexError):
            continue
    return sorted(pairs)

# Sorted list of (mchi, mlsp) pairs with ntuple signal files on disk.
SIGNAL_GRID = _discover_mass_points()

def ntuple_sig_files(mchi, mlsp, years):
    """Return list of existing signal n-tuples for the given years."""
    paths = [ntuple_sig_path(y, mchi, mlsp) for y in years]
    return [p for p in paths if os.path.exists(p)]

def ntuple_data_files(years):
    """Return list of existing background n-tuples for the given years."""
    paths = [ntuple_bkg_path(y) for y in years]
    return [p for p in paths if os.path.exists(p)]

# ===========================================================================
# Fit output path helpers
# ===========================================================================

def _sig_fit_base(mchi, mlsp, era_tag):
    """Base filename stem for signal fit outputs."""
    return f'fit_sig_mchi{mchi}_mlsp{mlsp}_{era_tag}'

def sig_fit_json(mchi, mlsp, era_tag):
    return os.path.join(FITS_DIR, 'signal',
                        _sig_fit_base(mchi, mlsp, era_tag) + '.json')

def sig_fit_pdf(mchi, mlsp, era_tag):
    return os.path.join(FITS_DIR, 'signal',
                        _sig_fit_base(mchi, mlsp, era_tag) + '.pdf')

def sig_fit_log(mchi, mlsp, era_tag):
    return os.path.join(FITS_DIR, 'signal',
                        _sig_fit_base(mchi, mlsp, era_tag) + '.log')

def sig_fit_ws(mchi, mlsp, era_tag):
    """Path to the per-mass-point fitted signal workspace ROOT file."""
    return os.path.join(WS_DIR, 'signal',
                        _sig_fit_base(mchi, mlsp, era_tag) + '_ws.root')

# Z-peak control-region fit outputs (one per era) — fixed m(ll) shape for
# the peaking background component, consumed by fit_background.py.
def zpeak_fit_json(era_tag):
    return os.path.join(FITS_DIR, 'zpeak', f'fit_zpeak_{era_tag}.json')

def zpeak_fit_pdf(era_tag):
    return os.path.join(FITS_DIR, 'zpeak', f'fit_zpeak_{era_tag}.pdf')

def zpeak_fit_log(era_tag):
    return os.path.join(FITS_DIR, 'zpeak', f'fit_zpeak_{era_tag}.log')

def zpeak_fit_ws(era_tag):
    return os.path.join(WS_DIR, 'zpeak', f'fit_zpeak_{era_tag}_ws.root')

# Background fit outputs (one per era)
def bkg_fit_json(era_tag):
    return os.path.join(FITS_DIR, 'background', f'fit_bkg_{era_tag}.json')

def bkg_fit_pdf(era_tag):
    return os.path.join(FITS_DIR, 'background', f'fit_bkg_{era_tag}.pdf')

def bkg_fit_log(era_tag):
    return os.path.join(FITS_DIR, 'background', f'fit_bkg_{era_tag}.log')

def bkg_fit_ws(era_tag):
    return os.path.join(WS_DIR, 'background', f'fit_bkg_{era_tag}_ws.root')

# Combined workspace (produced by make_workspaces.py)
def combined_bkg_ws(era_tag):
    return os.path.join(WS_DIR, 'combined', f'bkg_workspace_{era_tag}.root')

# Datacards (produced by make_datacards.py)
def datacard_path(mchi, mlsp, era_tag, xsec_mode='2d'):
    return os.path.join(CARDS_DIR, era_tag,
                        f'datacard_mchi{mchi}_mlsp{mlsp}_{era_tag}_{xsec_mode}xsec.txt')


BTAG_WP = {
    '2016':     {'branch': 'jet_deepflav',  'wps': [0.0480, 0.2489, 0.6377]},
    '2016APV':  {'branch': 'jet_deepflav',  'wps': [0.0508, 0.2598, 0.6502]},
    '2017':     {'branch': 'jet_deepflav',  'wps': [0.0532, 0.3040, 0.7476]},
    '2018':     {'branch': 'jet_deepflav',  'wps': [0.0490, 0.2783, 0.7100]},
    '2022':     {'branch': 'jet_btagpnetb', 'wps': [0.0470, 0.2450, 0.6734]},
    '2022EE':   {'branch': 'jet_btagpnetb', 'wps': [0.0499, 0.2605, 0.6915]},
    '2023':     {'branch': 'jet_btagpnetb', 'wps': [0.0358, 0.1917, 0.6172]},
    '2023BPix': {'branch': 'jet_btagpnetb', 'wps': [0.0359, 0.1919, 0.6133]},
    '2024':     {'branch': 'jet_btaguptb',  'wps': [0.0246, 0.1272, 0.4648]},
    '2025':     {'branch': 'jet_btaguptb',  'wps': [0.0246, 0.1272, 0.4648]},  # placeholder; update when 2025 WPs are available
}
