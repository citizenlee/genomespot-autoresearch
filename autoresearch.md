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
- Available features: ~174 columns across 5 localizations (AA freq, pI bins, derived protein metrics, DNA metrics)
- Can use any sklearn model or ensemble strategy
- Can add derived features computed from existing columns

## Feature Groups Available
- `aa_X` (20 AAs) — per localization
- `pis_N_M` (8 pI bins) — per localization
- `mean_gravy`, `mean_nh2o`, `mean_pi`, `mean_zc`, `proportion_R_RK`, `mean_thermostable_freq` — per localization
- `nt_C`, `pur_pyr_transition_freq`, `protein_coding_density`, `mean_protein_length` — all only
- Localizations: all, extracellular_soluble, intracellular_soluble, membrane, diff_extra_intra

## Baseline Performance (Run 1)
- composite_score: 0.628956
- oxygen_f1: 0.9438, temperature_r2: 0.7069, salinity_r2: 0.6042, ph_r2: 0.4449

## What's Been Tried
(Updated as experiments accumulate)

### Baseline (Run 1)
- Oxygen: LogisticRegression on 20 AA features (all compartment only)
- Temperature/Salinity/pH: Lasso(alpha=0.01) on 60 AA features (3 compartments)
- No pI features, no derived metrics, no diff_extra_intra features
