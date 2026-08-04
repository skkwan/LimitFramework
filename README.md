# LimitFramework

Limit-setting pipeline for the higgsino HH/ZH → bbγγ + MET search (CMS,
Run 2 + Run 3). Produces Combine datacards, expected limits, and the final
Brazil-band / 2D exclusion / BF-exclusion plots.

All configuration lives in `scripts/config.py` — years, luminosities, signal
regions, fit ranges, blind window, and every file-path helper. Every other
script imports from it rather than redefining constants.

## Layout

```
limitframework/
  scripts/        the main pipeline (see below)
  slims/          slimmed ROOT files (signal MC + data), one tree per file
  combine/        fits/, workspaces/, datacards/, limits/ — all pipeline outputs
  signal_master.csv   per-mass-point signal yields (input to make_datacards.py)
```

## Pipeline (run in this order)

| # | Script | What it does |
|---|--------|---------------|
| 1 | `make_slims.py` | Slims raw pico files down to `mgg, met, mbb, w_lumi` for signal MC and data. |
| 2 | `fit_signal.py` | Fits a 2D signal shape (mgg × MET) per mass point using RooKeysPdf (KDE). Signal-only, never touches data. |
| 3 | `fit_background.py` | Fits the 2-component background model (real-MET / fake-MET, each Gumbel(MET)×Exp(mgg)) to the data sideband (mgg outside the blind window). |
| 4 | `make_workspaces.py` | Builds the combined background RooFit workspace per era: imports the background PDF, seeds its floating `_norm` from `bkg_estimate_full`, and builds `data_obs` from the data slims. |
| 5 | `make_datacards.py` | Writes one Combine datacard per (mchi, mlsp, era), with BF and cross-section-mode scaling (`--bf`, `--xsec`). |
| 6 | `run_combine.py` | Runs `combine -M AsymptoticLimits ... -t -1 --noFitAsimov` on every datacard and collects expected limits into a CSV. |
| 7 | `plot.py` | Makes the 1D Brazil band, 2D exclusion contour, and BF-exclusion plots from the limits CSV. |

Supporting scripts: `run_bf_scan.py` (wraps steps 5–6 over a grid of
BF values for the BF-exclusion plot).

## Typical invocation

```bash
python3 scripts/make_slims.py --era run2          # or omit --era for all years
python3 scripts/fit_signal.py --era run2 --force
python3 scripts/fit_background.py --era run2 --force
python3 scripts/make_workspaces.py --era run2 --force
python3 scripts/make_datacards.py --era run2 --bf 1.0 --xsec 2d --force
python3 scripts/run_combine.py    --era run2 --bf 1.0 --xsec 2d --force
python3 scripts/plot.py --era run2 --bf 1.0 --xsec 1d --plot both
```

Most scripts skip work that already exists on disk unless `--force` is
given, so the pipeline is safe to re-run incrementally.
