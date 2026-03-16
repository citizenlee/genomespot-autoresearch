# GenomeSPOT Autoresearch Worklog

## Session Info
- Goal: Optimize composite_score for GenomeSPOT growth condition prediction
- Started: 2026-03-15
- Branch: autoresearch/composite-score-2026-03-15

## Data Summary
- 15,596 genomes, 266 columns (174 expected features + extras)
- Oxygen: 7,300 labels (classification, F1)
- Temperature: 4,783 labels (regression, R2)
- Salinity: 2,376 labels (regression, R2)
- pH: 3,549 labels (regression, R2)
- Phylogenetic holdouts at family level

## Key Insights
- pH is the weakest condition (R2=0.44) — 2x weight in composite, so improving it has biggest ROI
- Protein localization features (compartment-aware) helped pH by +0.36 in the paper
- Only linear models tested so far — nonlinear models untested
- diff_extra_intra features capture export signal differences, important for pH
- pI distributions and derived metrics not yet used

## Next Ideas
- Add all feature groups (pI, derived metrics, diff_extra_intra) to all conditions
- Try Ridge instead of Lasso (less aggressive feature elimination)
- Try GradientBoosting or RandomForest for nonlinear relationships
- Try condition-specific feature sets (pH may benefit from different features than temperature)
- Feature engineering: interaction terms, ratios
- Ensemble/stacking across model types

---

### Run 1: baseline — composite_score=0.628956 (KEEP)
- Timestamp: 2026-03-15 16:40
- What changed: Initial baseline with default train.py
- Result: composite=0.6290, oxygen_f1=0.9438, temp_r2=0.7069, sal_r2=0.6042, ph_r2=0.4449
- Insight: pH is clearly the bottleneck. Oxygen is already strong. All models are basic linear.
- Next: Add pI distributions and derived protein metrics to all conditions
