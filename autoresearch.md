# Autoresearch: GenomeSPOT composite_score

## Objective
Optimize the composite_score for GenomeSPOT, a tool that predicts microbial growth conditions (oxygen tolerance, temperature, salinity, pH) from genome amino acid composition. The composite_score is a weighted mean of per-condition CV scores: oxygen F1, temperature R2, salinity R2, pH R2 (2x weight). Higher is better.

The training data has 15,596 bacterial/archaeal genomes with features computed from proteome composition (amino acid frequencies, isoelectric point distributions, GRAVY, Zc, nH2O, thermostable residue freq) across 5 localization compartments (all, extracellular_soluble, intracellular_soluble, membrane, diff_extra_intra), plus DNA features (nucleotide freq, purine-pyrimidine transitions, coding density, mean protein length). Labels come from BacDive.

Evaluation uses phylogenetic holdouts at the family level (no data leakage).

## Metrics
- **Primary**: composite_score (unitless 0-1, higher is better)
  - = (oxygen_f1 + temperature_r2 + salinity_r2 + 2*ph_r2) / 5
- **Secondary**: oxygen_f1, temperature_r2, salinity_r2, ph_r2, temperature_rmse, salinity_rmse, ph_rmse, test_* variants

## How to Run
`./autoresearch.sh` — outputs `METRIC name=number` lines.

## Files in Scope
- `train.py` — Model definitions, feature selection, pipeline config. **ONLY file the agent modifies.**

## Off Limits
- `prepare.py` — Evaluation harness (data loading, splits, scoring)
- `data/` — Training data and holdout sets
- `autoresearch.sh` — Run script

## Constraints
- All evaluation uses phylogenetic holdouts at family level
- Features must be computable from genome sequence alone
- Prefer simpler models when performance is equivalent

## Baseline Performance (Run 1)
- composite_score: 0.628956
- oxygen_f1: 0.9438, temperature_r2: 0.7069, salinity_r2: 0.6042, ph_r2: 0.4449

## Current Best (Run 45, composite_score: 0.643527, +2.3%)
- oxygen_f1: 0.9501, temperature_r2: 0.7080, salinity_r2: 0.6143, ph_r2: 0.4726

### Current Best Configuration
- **Oxygen**: LogReg(C=10) on AA+pI+derived features (all compartment, 34 features)
- **Temperature**: BaggingRegressor(Lasso(0.01), n=20, 0.8) on 60 AA features (3 compartments)
- **Salinity**: LassoCV on 60 AA + 3 pI_4_5 features (63 features)
- **pH**: BaggingRegressor(Lasso(0.01), n=20, 0.8) on 60 AA + 7 derived features (67 features)
  - diff_extra_intra_aa_E, diff_extra_intra_aa_D, diff_extra_intra_mean_thermostable_freq
  - all_mean_pi, all_pis_basic, all_pis_acidic, membrane_pis_5_6

## What's Been Tried (46 runs)

### Key Wins
1. **Correlation-guided feature selection** (Run 9, +1.1%): 2-3 most correlated non-AA features per condition
2. **Oxygen pI+derived features + C=10** (Runs 13,17): Expanded feature set + less regularization for large N
3. **LassoCV for salinity** (Run 21): Auto alpha tuning finds better regularization
4. **pI features for pH** (Runs 29-33): all_mean_pi, pis_basic, pis_acidic, membrane_pis_5_6 — progressively improved pH
5. **BaggingRegressor(Lasso) for pH** (Run 38, biggest single gain): Bootstrap variance reduction on small N=603
6. **Remove noise features** (Run 24): Dropping thermostable_freq from temperature improved it

### Key Dead Ends
- **Ridge/ElasticNet**: L2 keeps noise features active → always hurts
- **GradientBoosting**: Bad CV scores — phylogenetic CV creates hard distribution shifts favoring linear models
- **Adding many features at once**: Small N causes overfitting (pH=603, sal=473)
- **Auto-tuning alpha for pH (LassoCV/ElasticNetCV)**: Inner CV doesn't match phylogenetic CV structure
- **QuantileTransformer**: Destroys linear relationships
- **BaggingRegressor for salinity**: N=473 too small, bootstrap reduces effective sample size too much
- **BaggingClassifier for oxygen**: Near F1 ceiling, bagging adds noise
- **max_features < 1.0 in BaggingRegressor**: Missing important features per bag

### Architectural Insights
1. **Linear models fundamentally better for phylogenetic CV** — distribution shifts across families favor simpler models
2. **Lasso's feature selection is essential** — any model keeping all features active hurts
3. **BaggingRegressor is the best variance reduction strategy** — works for pH and temperature but not salinity
4. **Per-condition model + feature optimization is critical** — no single approach works for all conditions
5. **pI distribution features are underexploited** — especially for pH and salinity
6. **Correlation analysis before feature engineering** is essential for small N problems
7. **pH is fundamentally hard**: 66% of samples at pH 7-8, only 4% at extremes
8. **Simpler can be better**: Removing noise features (thermostable_freq) improved temperature
