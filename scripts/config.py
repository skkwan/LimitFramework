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
import json

# ── Root directory (always resolves to limitframework/, wherever you run from) ──
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_WWW_DIR = "/eos/user/s/skkwan/www/higgsino/studies/mll-MET-fit-2D/combine_shapes"

# ===========================================================================
# Era definitions
# ===========================================================================

RUN2_YEARS = ['2016', '2016APV', '2017', '2018']
RUN3_YEARS = ['2022', '2022EE', '2023', '2023BPix', '2024', '2025']
ALL_YEARS  = RUN2_YEARS + RUN3_YEARS

# Years with signal MC used for the signal shapes. A mass point is only fit if
# its ntuple exists in every one of these years (see _discover_mass_points).
SIGNAL_MC_YEARS = ['2016APV', '2016', '2017', '2018']

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
    '2016':     16.80,
    '2016APV':  19.51,
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

# Projected yields for the expected limits, at PROJECTED_LUMI (scaled up
# from 2018 MC). The total background is the bkg rate in the datacards
# (make_datacards.py) and the data_obs event count (make_workspaces.py).
# Not per era: the projection is for the total dataset.
PROJECTED_LUMI = 450.0  # fb^-1
#   2018: peaking=0.20672 (1.55481 at 450 fb^-1)  nonpeak=1.17161 (8.81208 at 450 fb^-1)
BKG_YIELD = {'peaking': 1.5548, 'nonpeak': 8.812}
SIGNAL_YIELD_COL = 'yield_450fb'  # signal_config.json field at PROJECTED_LUMI

# Relative Gaussian uncertainty on ratio_peaking in the datacards
RATIO_PEAKING_REL_UNC = 0.30
# Relative (lnN) uncertainty on the total background yield in the datacards
BKG_YIELD_REL_UNC = 0.30

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

# Per-mass-point signal yields and cross sections
SIGNAL_CONFIG = os.path.join(ROOT_DIR, 'signal_config.json')

def load_signal_config():
    """List of per-mass-point dicts from SIGNAL_CONFIG's "mass_points"."""
    if not os.path.exists(SIGNAL_CONFIG):
        raise FileNotFoundError(f"signal config not found: {SIGNAL_CONFIG}")
    with open(SIGNAL_CONFIG) as f:
        return json.load(f)['mass_points']

def signal_xsec(point, era_tag, mode):
    """
    Cross section [pb] of one signal_config.json entry: xsec_{mode}[era_tag],
    mode '2d' or '1d', era_tag 'run2' or 'run3'. None if the entry doesn't
    list it.

    'run2run3': luminosity-weighted (xsec_run2*lumi_run2 +
    xsec_run3*lumi_run3) / lumi_total, so the combined-era reference xsec
    roughly cancels against the combined datacard's mixed run2+run3 nominal
    rate. Assumes eff_run2 ~ eff_run3 (checked to be reasonably close,
    though not exactly equal -- see diag notes on run2 vs run3 signal
    efficiency). This is an approximation, not an exact cancellation like
    the single-era case.
    """
    xsec = point.get(f'xsec_{mode}')
    if not isinstance(xsec, dict):
        return None
    if era_tag == 'run2run3':
        x2, x3 = xsec.get('run2'), xsec.get('run3')
        if x2 is None or x3 is None:
            return None
        return (x2 * LUMI_RUN2 + x3 * LUMI_RUN3) / LUMI_RUN2RUN3
    return xsec.get(era_tag)

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
    '2016APV': '/eos/cms/store/group/phys_susy/skkwan/condorHistogramming/2026-10-05-19h18m_2016-samples-SR_mll_MET_fit_scheme-only',
    '2018': '/eos/cms/store/group/phys_susy/skkwan/condorHistogramming/2026-09-29-20h09m_2018-samples-SR_mll_MET_fit_scheme-only',
    '2017': '/eos/cms/store/group/phys_susy/skkwan/condorHistogramming/2026-10-05-19h11m_2017-samples-SR_mll_MET_fit_scheme-only', 
    '2016': '/eos/cms/store/group/phys_susy/skkwan/condorHistogramming/2026-10-05-19h18m_2016-samples-SR_mll_MET_fit_scheme-only',
}

# The 2016 production holds both halves of the year in one directory, with
# sample (and file) names prefixed: preVFP_ = 2016APV, postVFP_ = 2016.
RAW_SIGNAL_PREFIX = {'2016APV': 'preVFP_', '2016': 'postVFP_'}

RAW_BKG_DIR = {
    # TODO: add remaining years
    '2018': '/eos/cms/store/group/phys_susy/skkwan/condorHistogramming/2026-09-29-20h09m_2018-samples-SR_mll_MET_fit_scheme-only',
    '2017': '/eos/cms/store/group/phys_susy/skkwan/condorHistogramming/2026-10-05-19h11m_2017-samples-SR_mll_MET_fit_scheme-only', 
    '2016': '/eos/cms/store/group/phys_susy/skkwan/condorHistogramming/2026-10-05-19h18m_2016-samples-SR_mll_MET_fit_scheme-only',
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
    """Scan the signal ntuple directory for (mchi, mlsp) pairs, matching
    ntuple_sig_path's naming:
    snapshot_{SIGNAL_PROCESS}_{mchi}_{mlsp}_SR_mll_MET_fit_scheme.root
    Only pairs present in every year of SIGNAL_MC_YEARS are kept."""
    per_year = []
    for year in SIGNAL_MC_YEARS:
        pattern = os.path.join(NTUPLE_SIG_DIR, year,
                               f'snapshot_{SIGNAL_PROCESS}_*_*_SR_mll_MET_fit_scheme.root')
        pairs = set()
        for f in glob.glob(pattern):
            base = os.path.basename(f).replace('.root', '')
            parts = base.split('_')  # ['snapshot', SIGNAL_PROCESS, mchi, mlsp, 'SR', ...]
            try:
                pairs.add((int(parts[2]), int(parts[3])))
            except (ValueError, IndexError):
                continue
        per_year.append(pairs)
    return sorted(set.intersection(*per_year))

# Sorted list of (mchi, mlsp) pairs with ntuple signal files on disk in all SIGNAL_MC_YEARS.
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

def sig_fit_all_pdf(era_tag):
    """Every mass point's signal fit PDF for one era, merged into one file."""
    return os.path.join(FITS_DIR, 'signal', f'fit_sig_all_{era_tag}.pdf')

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

def bkg_components_json(era_tag):
    """Per-component (MET peaking, MET non-peaking, m(ll) non-peaking) fit results."""
    return os.path.join(FITS_DIR, 'background', f'fit_bkg_components_{era_tag}.json')

def bkg_fit_pdf(era_tag):
    return os.path.join(FITS_DIR, 'background', f'fit_bkg_{era_tag}.pdf')

def bkg_fit_log(era_tag):
    return os.path.join(FITS_DIR, 'background', f'fit_bkg_{era_tag}.log')

def bkg_fit_ws(era_tag):
    return os.path.join(WS_DIR, 'background', f'fit_bkg_{era_tag}_ws.root')

# Combined signal+background workspace, one per mass point (produced by make_workspaces.py)
def combined_ws(mchi, mlsp, era_tag):
    return os.path.join(WS_DIR, 'combined',
                        f'workspace_mchi{mchi}_mlsp{mlsp}_{era_tag}.root')

# Signal models (--m): 'gmsb' (mlsp = 1 GeV, sigma_1D) and 'bino' (any mlsp,
# yields rescaled from sigma_1D to sigma_2D). The signal samples were produced
# with sigma_1D, so signal_config.json yields are sigma_1D-based.
MODELS = ('gmsb', 'bino')
GMSB_MLSP = 1
GMSB_BR_ZH = 0.5  # BR(chi chi -> ZH) in GMSB; applied as the datacard signal rate, not in the samples
BINO_BR_ZH = 1.0  # BR(chi chi -> ZH) allowed to be 100% in Higgsino-Bino models

def model_mlsp(model, mlsp):
    """The mlsp [GeV] to run for `model`: gmsb implies GMSB_MLSP, bino requires --mlsp."""
    if model == 'gmsb':
        if mlsp not in (None, GMSB_MLSP):
            raise ValueError(f"--m gmsb implies --mlsp {GMSB_MLSP}, got --mlsp {mlsp}")
        return GMSB_MLSP
    if mlsp is None:
        raise ValueError("--m bino requires --mlsp")
    return mlsp

def model_xsec_mode(model):
    """signal_config.json cross-section key suffix of the reference sigma: gmsb -> '1d', bino -> '2d'."""
    return {'gmsb': '1d', 'bino': '2d'}[model]

def model_br(model):
    """BR factor applied on top of the sigma_1D-based yields: 0.5 for gmsb; 1 for bino (BR not applied here)."""
    if model == 'gmsb':
        return GMSB_BR_ZH 
    elif model == 'bino':
        return BINO_BR_ZH
    else:
        return 1.0

# Datacards (produced by make_datacards.py), one folder per model
def datacard_dir(era_tag, model):
    return os.path.join(CARDS_DIR, era_tag, model)

def datacard_path(mchi, mlsp, era_tag, model):
    return os.path.join(datacard_dir(era_tag, model),
                        f'datacard_mchi{mchi}_mlsp{mlsp}_{era_tag}_{model}.txt')


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
