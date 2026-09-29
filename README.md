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

## H(bb)Z(ll) setup
```bash
cd scripts
python3 make_inputs.py --mode all
python3 fit_signal.py --era run2 --force  
# Fit the z-peak m(ll) only
python3 fit_zpeak.py --era run2 --force
# Declare the analytic forms for the remaining background components and fit the MC in two dimensions. r is fixed to the generated value
python3 fit_background.py --era run2 --force
# Load the background and signal PDFs. Uses the master_signal.csv file by default. Declares three RooRealVars (bkg_total_{era}_norm, ratio_peaking_{era}, and sig_pdf_...{era}_norm) and builds the total background PDF. Only the signal strength r floats in the fit
python3 make_workspaces.py --era run2 --force
# Write Combine datacards for each mass point and each era. 
python3 make_datacards.py --era run2 --mchi 650 --mlsp 1 --force
```
Locations of intermediate outputs:
- The signal shapes are saved to `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/fits/signal/*.json`
- The Z-peak m(ll) shape is saved to `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/fits/zpeak/fit_zpeak_run2.json`
- The remaining three background shapes are saved to `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/fits/background/fit_bkg_components_run2.json`.
- The plots of all of the background shapes are in `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/fits/background/fit_bkg_run2.pdf`.
- The workspaces are in `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/workspaces/combined/`, e.g. `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/workspaces/combined/workspace_mchi650_mlsp1_run2.root`.
- The datacards are in `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/datacards/run2/`, e.g. `/afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/datacards/run2/datacard_mchi650_mlsp1_run2_2dxsec.txt`.


Then to run Combine:
```bash
cd /afs/cern.ch/work/s/skkwan/public/combineArea/CMSSW_14_1_0_pre4/src/CombineHarvester/CombineTools/bin/LimitFramework/combine/datacards/run2
cmsenv
ValidateDatacards.py datacard_mchi650_mlsp1_run2_2dxsec.txt
combine -M AsymptoticLimits datacard_mchi650_mlsp1_run2_2dxsec.txt -t -1 --expectSignal 0
```
Currently getting this:
```
 <<< Combine >>> 
 <<< v10.2.1 >>>
>>> Random number generator seed is 123456
>>> Method used is AsymptoticLimits

 -- AsymptoticLimits ( CLs ) --
Observed Limit: r < 0.4094
Expected  2.5%: r < 0.1794
Expected 16.0%: r < 0.2624
Expected 50.0%: r < 0.4102
Expected 84.0%: r < 0.6554
Expected 97.5%: r < 0.9929

Done in 0.12 min (cpu), 0.12 min (real)
```