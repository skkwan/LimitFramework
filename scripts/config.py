"""
config.py — Central configuration for the limit framework.
All other scripts import from here. Change things in one place.
"""

import os
import glob

# ── Root directory (always resolves to limitframework/, wherever you run from) ──
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===========================================================================
# Era definitions
# ===========================================================================

RUN2_YEARS = ['2016', '2016APV', '2017', '2018']
RUN3_YEARS = ['2022', '2022EE', '2023', '2023BPix', '2024', '2025']
ALL_YEARS  = RUN2_YEARS + RUN3_YEARS

# Years for which signal MC exists. 2025 has no signal MC -- its signal slim
# files and shapes are borrowed from 2024 (same detector conditions assumed).
SIGNAL_YEAR_MAP = {y: y for y in ALL_YEARS}
SIGNAL_YEAR_MAP['2025'] = '2024'

def signal_year_for(year):
    """Return the year whose signal slims/shapes to use for a given data year."""
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
# Signal regions and mbb windows
# ===========================================================================
# To add a new region, just add an entry here.
# The mbb cut is applied downstream (in the fit/datacard), not in the slim.

SIGNAL_REGIONS = {
    'SRHH': {'mbb_lo': 100., 'mbb_hi': 140.},
    'SRZH': {'mbb_lo':  60., 'mbb_hi': 100.},
}

# ===========================================================================
# Fit variables
# ===========================================================================

MGG_LO, MGG_HI =  100.,  200.
MET_LO, MET_HI =    0., 1300.

# Sideband windows for background fit (mgg regions outside the signal peak)
SIDEBAND_WINDOWS = [(100., 120.), (130., 200.)]

# Blind window = the gap between the two sideband windows. Single source of
# truth for every stage that needs to exclude the signal-region mgg range
# (slim creation, background fit, workspace building).
BLIND_MGG_LO = SIDEBAND_WINDOWS[0][1]   # 120.
BLIND_MGG_HI = SIDEBAND_WINDOWS[1][0]   # 130.

def sideband_cut_expr(mgg_branch='mgg'):
    """TTree/RDataFrame-style boolean expression selecting the mgg sideband
    (i.e. excluding the blinded signal window)."""
    return f'({mgg_branch} < {BLIND_MGG_LO} || {mgg_branch} > {BLIND_MGG_HI})'

def is_blind_mgg(mgg):
    """True if mgg falls inside the blinded signal window. Works on scalars
    or numpy/array-like input."""
    return (mgg >= BLIND_MGG_LO) & (mgg <= BLIND_MGG_HI)

# Minimum number of (unweighted) entries required to attempt a fit
MIN_EVENTS_FOR_FIT = 50

# ===========================================================================
# Directory structure
# ===========================================================================

SLIM_SIG_DIR  = os.path.join(ROOT_DIR, 'slims', 'signal')
SLIM_DATA_DIR = os.path.join(ROOT_DIR, 'slims', 'data')
COMBINE_DIR   = os.path.join(ROOT_DIR, 'combine')
FITS_DIR      = os.path.join(COMBINE_DIR, 'fits')
WS_DIR        = os.path.join(COMBINE_DIR, 'workspaces')
CARDS_DIR     = os.path.join(COMBINE_DIR, 'datacards')
LIMITS_DIR    = os.path.join(COMBINE_DIR, 'limits')

# Signal master CSV (produced by make_signal_csv.py)
SIGNAL_CSV = os.path.join(ROOT_DIR, 'signal_master.csv')

# ===========================================================================
# Slim file path helpers
# ===========================================================================

def slim_sig_path(sigtype, year, mchi, mlsp):
    """Full path to one slim signal ROOT file."""
    return os.path.join(
        SLIM_SIG_DIR, sigtype, str(year),
        f'slim_{sigtype}_{year}_mchi{mchi}_mlsp{mlsp}.root'
    )

def slim_data_path(year):
    """Full path to one slim data ROOT file."""
    return os.path.join(SLIM_DATA_DIR, str(year), f'slim_data_{year}.root')

# ===========================================================================
# Signal mass grids — discovered from existing slim files
# ===========================================================================

def _discover_mass_points(sigtype):
    """Scan slim signal directory to find all (mchi, mlsp) pairs that exist."""
    pattern = os.path.join(SLIM_SIG_DIR, sigtype, '*',
                           f'slim_{sigtype}_*_mchi*_mlsp*.root')
    files = glob.glob(pattern)
    pairs = set()
    for f in files:
        base = os.path.basename(f)
        try:
            parts = base.replace('.root', '').split('_')
            mchi_idx = next(i for i, p in enumerate(parts) if p.startswith('mchi'))
            mlsp_idx = next(i for i, p in enumerate(parts) if p.startswith('mlsp'))
            mchi = int(parts[mchi_idx].replace('mchi', ''))
            mlsp = int(parts[mlsp_idx].replace('mlsp', ''))
            pairs.add((mchi, mlsp))
        except (ValueError, StopIteration):
            continue
    return sorted(pairs)

_signal_grid_cache = {}

def get_signal_grid(sigtype):
    """Return sorted list of (mchi, mlsp) pairs for this sigtype."""
    if sigtype not in _signal_grid_cache:
        _signal_grid_cache[sigtype] = _discover_mass_points(sigtype)
    return _signal_grid_cache[sigtype]

# Populated after SLIM_SIG_DIR is defined above
SIGNAL_GRID = {st: get_signal_grid(st) for st in ['HH', 'ZH']}

def slim_sig_files(sigtype, mchi, mlsp, years):
    """Return list of existing slim signal files for the given years."""
    paths = [slim_sig_path(sigtype, y, mchi, mlsp) for y in years]
    return [p for p in paths if os.path.exists(p)]

def slim_data_files(years):
    """Return list of existing slim data files for the given years."""
    paths = [slim_data_path(y) for y in years]
    return [p for p in paths if os.path.exists(p)]

# ===========================================================================
# Fit output path helpers
# ===========================================================================

def _fit_base(sigtype, mchi, mlsp, era_tag):
    """Base filename stem for fit outputs."""
    return f'fit_sig_{sigtype}_mchi{mchi}_mlsp{mlsp}_{era_tag}'

def sig_fit_json(sigtype, mchi, mlsp, era_tag):
    return os.path.join(FITS_DIR, 'signal', sigtype,
                        _fit_base(sigtype, mchi, mlsp, era_tag) + '.json')

def sig_fit_pdf(sigtype, mchi, mlsp, era_tag):
    return os.path.join(FITS_DIR, 'signal', sigtype,
                        _fit_base(sigtype, mchi, mlsp, era_tag) + '.pdf')

def sig_fit_log(sigtype, mchi, mlsp, era_tag):
    return os.path.join(FITS_DIR, 'signal', sigtype,
                        _fit_base(sigtype, mchi, mlsp, era_tag) + '.log')

def sig_fit_ws(sigtype, mchi, mlsp, era_tag):
    """Path to the per-mass-point fitted signal workspace ROOT file."""
    return os.path.join(WS_DIR, 'signal', sigtype,
                        _fit_base(sigtype, mchi, mlsp, era_tag) + '_ws.root')

# Background fit outputs (one per signal region per era)
def bkg_fit_json(region, era_tag):
    return os.path.join(FITS_DIR, 'background',
                        f'fit_bkg_{region}_{era_tag}.json')

def bkg_fit_pdf(region, era_tag):
    return os.path.join(FITS_DIR, 'background',
                        f'fit_bkg_{region}_{era_tag}.pdf')

def bkg_fit_log(region, era_tag):
    return os.path.join(FITS_DIR, 'background',
                        f'fit_bkg_{region}_{era_tag}.log')

def bkg_fit_ws(region, era_tag):
    return os.path.join(WS_DIR, 'background',
                        f'fit_bkg_{region}_{era_tag}_ws.root')

# Combined workspace (produced by make_workspaces.py)
def combined_bkg_ws(era_tag):
    return os.path.join(WS_DIR, 'combined', f'bkg_workspace_{era_tag}.root')

# Datacards (produced by make_datacards.py)
def datacard_path(mchi, mlsp, era_tag, bf=1.0, xsec_mode='2d'):
    bf_str = f"bf{bf:.2f}".replace('.', 'p')   # e.g. bf1p00
    return os.path.join(CARDS_DIR, era_tag,
                        f'datacard_mchi{mchi}_mlsp{mlsp}_{era_tag}_{bf_str}_{xsec_mode}xsec.txt')


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
