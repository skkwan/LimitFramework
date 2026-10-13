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
| 2 | `fit_signal.py` | Fits a 2D signal shape ple

```bash
python3 scripts/make_inputs.py --mode all             
python3 scripts/fit_signal.py   --era run2 --force
python3 scripts/fit_zpeak.py    --era run2 --force
python3 scripts/fit_background.py --era run2 --force
python3 scripts/make_workspaces.py --era run2 --force
python3 scripts/make_datacards.py --era run2 --m gmsb --force
python3 scripts/run_combine.py combine/datacards/run2/gmsb --m gmsb --force
python3 scripts/plot.py --era run2 --m gmsb --plot 1d
```

Most scripts skip work that already exists on disk unless `--force` is
given, so the pipeline is safe to re-run incrementally.

## Signal models (`--m`)

`make_datacards.py`, `run_combine.py` and `plot.py` take `--m {gmsb,bino}`
(there is no `--xsec` option). The signal samples were produced with
sigma_1D (the full higgsino cross section), so the yields in
`signal_config.json` (`yield_450fb`, which sets the workspace `_norm`) are
sigma_1D-based. The model changes only the datacard `rate`, so workspaces
are shared between models.

| `--m` | mlsp | datacard signal rate | reference cross section (`plot.py`) |
|-------|------|----------------------|-------------------------------------|
| `gmsb` | implied: 1 GeV (`--mlsp` other than 1 is an error) | BR(ZH) = 0.5 (`config.GMSB_BR_ZH`) | 0.5 x sigma_1D |
| `bino` | required (`--mlsp N`) | sigma_2D / sigma_1D | sigma_2D |

- The expected signal is workspace `sig_pdf_..._norm` x datacard `rate` x `r`, so r is relative to
  the rate-scaled signal (r = 1 means BR(ZH) = 0.5 x sigma_1D for gmsb).
- The background yield is not a nuisance parameter: the datacard has `bkg_norm_<era> rateParam SR bkg 1 [0,10]`
  (floats freely, no lnN). The luminosity lnN (`lumi_<era>`) applies to both signal and background.
- sigma_1D and sigma_2D (Higgsino-Bino, `higgsinoBino_<mchi>` in luna-zhmet `xsec.h`) are filled in
  `signal_config.json` by `scripts/prepare_signal_json.py`. Run3 has no Higgsino-Bino cross
  sections yet, so `--m bino --era run3` skips every point.
- Output naming: cards in `combine/datacards/<era>/<model>/datacard_mchi*_mlsp*_<era>_<model>.txt`,
  limits in `combine/limits/limits_<era>_<model>_mlsp<N>.csv`, plots
  `asymptoticLimits_<era>_<model>_mlsp<N>.png` / `exclusion2d_<era>_<model>.png`.
- Pass the model's card folder to `run_combine.py`, e.g. `combine/datacards/run2/gmsb`.
- If `yield_450fb` changes, rerun `make_workspaces.py --force`; changing `xsec_2d` or the BR needs
  only `make_datacards.py --force`.

```bash
# GMSB (mlsp = 1)
python3 make_datacards.py --era run2 --m gmsb --force
python3 run_combine.py ../combine/datacards/run2/gmsb --m gmsb --jobs 4 --force
python3 plot.py --era run2 --m gmsb --plot 1d
# Higgsino-Bino at mlsp = 100
python3 make_datacards.py --era run2 --m bino --mlsp 100 --force
python3 run_combine.py ../combine/datacards/run2/bino --m bino --mlsp 100 --jobs 4 --force
python3 plot.py --era run2 --m bino --mlsp 100 --plot 1d
```

## Toy experiments
After `make_datacards.py`,
```bash
combine -M FitDiagnostics -d combine/datacards/run2/gmsb/datacard_mchi650_mlsp1_run2_gmsb.txt -t 1000 -s -1 --expectSignal 0 --rMin -5 --rMax 5 --toysNoSystematics
```
(m(ll) Crystal Ball × MET spline) per mass point. Signal-only, never touches data. |
| 3 | `fit_zpeak.py` | Fits a fixed m(ll) Crystal Ball shape to a low-MET, Z+jets-dominated data control region — the "peaking" background m(ll) shape consumed by step 4. |
| 4 | `fit_background.py` | Fits the 2-component background model (peaking: fixed m(ll) shape × floating Gumbel(MET); non-peaking: floating Exp(m(ll)) × floating Gumbel(MET)) to data. |
| 5 | `make_workspaces.py` | Builds the combined background RooFit workspace per era: imports the background PDF, seeds its floating `_norm` from the observed data count, and builds `data_obs` from the data slims. |
| 6 | `make_datacards.py` | Writes one Combine datacard per (mchi, mlsp, era), with model selection (`--m gmsb`: mlsp=1, sigma_1D, signal rate x BR(ZH)=0.5; `--m bino --mlsp N`: yields scaled by sigma_2D/sigma_1D). |
| 7 | `run_combine.py` | Runs `combine -M AsymptoticLimits ... -t -1 --noFitAsimov` on every datacard and collects expected limits into a CSV. |
| 8 | `plot.py` | Makes the 1D Brazil band and 2D exclusion contour plots from the limits CSV. |

`run_bf_scan.py` and `plot.py`'s BF-exclusion mode are **not applicable**
to this single-process channel (no branching-fraction mixing) — left in
place for reference only.

## H(bb)Z(ll) workflow

1. First skim and run histograms and snapshot n-tuples in the signal region for all our signal mass points and MC background. This should result in some kind of big output directory, e.g. `/eos/cms/store/group/phys_susy/skkwan/condorHistogramming/2026-09-29-20h09m_2018-samples-SR_mll_MET_fit_scheme-only`. 

2. We will also need the Z-peak m(ll) distribution from a control region.

3. Then edit [scripts/config.py](scripts/config.py): `RAW_SIGNAL_DIR`, `RAW_BKG_DIR`, and `RAW_BKG_CRZ_DIR` need to point to the above directories.

   Signal shapes use the signal MC of all years in `SIGNAL_MC_YEARS` (2016APV, 2016, 2017, 2018), chained into one dataset per mass point. Only mass points with ntuples (mm and ee) in every year of `SIGNAL_MC_YEARS` are made and fit (`make_inputs.py` prints the points missing per year; `config.SIGNAL_GRID` is the same intersection). The signal fit plots scale the MC from the summed luminosity of the loaded years to `PROJECTED_LUMI`, and the fit JSON records `mc_years` and `mc_lumi`. `prepare_signal_json.py` fills `yield_{year}`, `yield_run2` (sum over all four years, only if all are present) and `yield_450fb` (`yield_run2` scaled to `PROJECTED_LUMI`); the 2018-scaled values are kept as `yield_run2_from2018only` and `yield_450fb_from2018only`.

4. Then run these steps:
```bash
cd scripts
# This combines the mm and ee n-tuples 
python3 make_inputs.py --mode all
python3 fit_signal.py --era run2 --force  
# After this, check the PDFs of the fits: https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/combine_shapes/ 
# Fit the z-peak m(ll) only
python3 fit_zpeak.py --era run2 --force
# Declare the analytic forms for the remaining background components and fit the MC in two dimensions. r is fixed to the generated value
python3 fit_background.py --era run2 --force
# Load the background and signal PDFs. Uses the master_signal.csv file by default. Declares three RooRealVars (bkg_total_{era}_norm, ratio_peaking_{era}, and sig_pdf_...{era}_norm) and builds the total background PDF. Only the signal strength r floats in the fit
python3 make_workspaces.py --era run2 --force
# Write Combine datacards for each mass point and each era, for GMSB (--m gmsb: mlsp=1, rate x BR(ZH)=0.5) or Higgsino-Bino (--m bino --mlsp N: scaled by sigma_2D/sigma_1D); see "Signal models" above
python3 make_datacards.py --era run2 --m gmsb --force
# if we do the --both option, we need to specify a mLSP for the Higgsino-Bino model
python3 make_datacards.py --era run2 --m both --mlsp 1 --force  
python3 make_datacards.py --era run2 --m bino --mlsp all --force  


# Run Combine Asymptotic Limits
python3 run_combine.py ../combine/datacards/run2/gmsb --m gmsb --jobs 4 --force
python3 run_combine.py ../combine/datacards/run2/bino --m bino --mlsp 1 --jobs 4 --force
python3 run_combine.py ../combine/datacards/run2/bino --m bino --mlsp all --jobs 4 --force

# Plot the results
python3 plot.py --csv=/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/limits/limits_run2_gmsb_mlsp1.csv --era run2 --m gmsb --plot 1d
```
Locations of intermediate outputs:
- The signal shapes are saved to `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/fits/signal/*.json` and the output plots can be viewed at https://skkwan.web.cern.ch/higgsino/studies/mll-MET-fit-2D/combine_shapes/. 
- The Z-peak m(ll) shape is saved to `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/fits/zpeak/fit_zpeak_run2.json`
- The remaining three background shapes are saved to `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/fits/background/fit_bkg_components_run2.json`.
- The plots of all of the background shapes are in `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/fits/background/fit_bkg_run2.pdf`.
- The workspaces are in `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/workspaces/combined/`, e.g. `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/workspaces/combined/workspace_mchi650_mlsp1_run2.root`.
- The datacards are in `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/datacards/run2/gmsb/`, e.g. `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/datacards/run2/gmsb/datacard_mchi650_mlsp1_run2_gmsb.txt`.


5. To manually run Combine and get expected limits with an asymptotic approximation:
```bash
cd /afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/datacards/run2/gmsb
cmsenv
ValidateDatacards.py datacard_mchi650_mlsp1_run2_gmsb.txt
combine -M AsymptoticLimits datacard_mchi650_mlsp1_run2_gmsb.txt -t -1 --expectSignal 0
```
Output from before the GMSB BR(ZH) = 0.5 rate was applied (r limits are now 2x these):
```
 <<< Combine >>> 
 <<< v10.2.1 >>>
>>> Random number generator seed is 123456
>>> Method used is AsymptoticLimits

 -- AsymptoticLimits ( CLs ) --
Observed Limit: r < 0.3380
Expected  2.5%: r < 0.1439
Expected 16.0%: r < 0.2121
Expected 50.0%: r < 0.3379
Expected 84.0%: r < 0.5480
Expected 97.5%: r < 0.8439

Done in 0.00 min (cpu), 0.00 min (real)
```

6. To have Combine compute limits without the asymptotic approximation, so instead using toy datasets to determine the test statistic distributions under the signal + background and background-only hypothesis, we use the `HybridNew` method to calculate limits using toys. To calculate LHC-style profile likelihood limits we set the option `--LHCmode LHC-limits`. 
```bash
combine -M HybridNew --LHCmode LHC-limits datacard_mchi650_mlsp1_run2_gmsb.txt
```
which gives
```
 <<< Combine >>> 
 <<< v10.2.1 >>>
>>> Random number generator seed is 123456
>>> using the Profile Likelihood test statistic modified for upper limits (Q_LHC)
>>> Method used is HybridNew

 -- Hybrid New -- 
Limit: r < 0.336382 +/- 0.0130679 @ 95% CL
Done in 0.22 min (cpu), 0.22 min (real)
```