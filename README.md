# LimitFramework

Limit-setting pipeline for the higgsino Z(ll)H(bb) + MET search (CMS,
Run 2 + Run 3). Single signal process (TChiZH), single signal region.
Produces Combine datacards, expected limits, and the final Brazil-band /
2D exclusion plots.

All configuration lives in `scripts/config.py` — years, luminosities, the
m(bb) signal-region window, fit ranges, and every file-path helper. Every
other script imports from it rather than redefining constants.

The 2D (m(ll), MET) fit machinery (signal Crystal Ball + spline, background
peaking/non-peaking mixture) is adapted from the standalone study in
`2DFit_higgsinos/hbb_zll/`.

**No blind window is implemented yet** (pre-unblinding / early validation
stage) — `fit_background.py` fits the full declared (mll, met) range with
nothing excluded. Add real blinding logic before ever looking at unblinded
data.

## Layout

```
limitframework/
  scripts/        the main pipeline (see below)
  ntuples/        the inputs
  combine/        fits/, workspaces/, datacards/, limits/ — all pipeline outputs
  signal_master.csv   per-mass-point signal yields (input to make_datacards.py)
```

## Pipeline (run in this order)

| # | Script | What it does |
|---|--------|---------------|
| 1 | `make_inputs.py` | Create `mll, met, w_lumi` for signal MC and data, applying the m(bb) signal-region window once. |
| 2 | `fit_signal.py` | Fits a 2D signal shape (m(ll) Crystal Ball × MET spline) per mass point. Signal-only, never touches data. |
| 3 | `fit_zpeak.py` | Fits a fixed m(ll) Crystal Ball shape to a low-MET, Z+jets-dominated data control region — the "peaking" background m(ll) shape consumed by step 4. |
| 4 | `fit_background.py` | Fits the 2-component background model (peaking: fixed m(ll) shape × floating Gumbel(MET); non-peaking: floating Exp(m(ll)) × floating Gumbel(MET)) to data. |
| 5 | `make_workspaces.py` | Builds the combined background RooFit workspace per era: imports the background PDF, seeds its floating `_norm` from the observed data count, and builds `data_obs` from the data slims. |
| 6 | `make_datacards.py` | Writes one Combine datacard per (mchi, mlsp, era), with cross-section-mode scaling (`--xsec`). |
| 7 | `run_combine.py` | Runs `combine -M AsymptoticLimits ... -t -1 --noFitAsimov` on every datacard and collects expected limits into a CSV. |
| 8 | `plot.py` | Makes the 1D Brazil band and 2D exclusion contour plots from the limits CSV. |

`run_bf_scan.py` and `plot.py`'s BF-exclusion mode are **not applicable**
to this single-process channel (no branching-fraction mixing) — left in
place for reference only.

## Example

```bash
python3 scripts/make_inputs.py --mode all             
python3 scripts/fit_signal.py   --era run2 --force
python3 scripts/fit_zpeak.py    --era run2 --force
python3 scripts/fit_background.py --era run2 --force
python3 scripts/make_workspaces.py --era run2 --force
python3 scripts/make_datacards.py --era run2 --xsec 2d --force
python3 scripts/run_combine.py    --era run2 --xsec 2d --force
python3 scripts/plot.py --era run2 --xsec 1d --plot both
```

Most scripts skip work that already exists on disk unless `--force` is
given, so the pipeline is safe to re-run incrementally.

## Toy experiments
After `make_datacards.py`,
```bash
combine -M FitDiagnostics -d combine/datacards/run2/datacard_mchi650_mlsp1_run2_2dxsec.txt -t 1000 -s -1 --expectSignal 0 --rMin -5 --rMax 5 --toysNoSystematics
```