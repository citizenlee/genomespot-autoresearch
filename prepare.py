"""FIXED evaluation harness for GenomeSPOT autoresearch.

DO NOT MODIFY THIS FILE. The autoresearch agent modifies only train.py.

This script:
1. Loads the pre-computed feature matrix and labels
2. Loads pre-computed train/test splits and CV folds
3. Imports the model pipeline from train.py
4. Runs 5-fold phylogenetic CV for all 4 conditions
5. Reports metrics in METRIC format for autoresearch
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

# Paths
DATA_DIR = Path(__file__).parent / "data"
HOLDOUTS_DIR = DATA_DIR / "holdouts"
TRAINING_DATA_FILE = DATA_DIR / "training_data.tsv"

CONDITIONS = ["oxygen", "temperature", "salinity", "ph"]

# Condition -> target column mapping
CONDITION_TO_TARGET = {
    "oxygen": "oxygen",
    "temperature": "temperature_optimum",
    "salinity": "salinity_optimum",
    "ph": "ph_optimum",
}

# Which conditions are classification vs regression
CLASSIFICATION_CONDITIONS = {"oxygen"}
REGRESSION_CONDITIONS = {"temperature", "salinity", "ph"}


def load_training_data() -> pd.DataFrame:
    """Load the pre-computed feature matrix with labels."""
    if not TRAINING_DATA_FILE.exists():
        print(f"ERROR: Training data not found at {TRAINING_DATA_FILE}", file=sys.stderr)
        print("Run: python prepare_data.py", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(TRAINING_DATA_FILE, sep="\t", index_col=0)


def load_cv_sets(condition: str, taxlevel: str = "family"):
    """Load pre-computed phylogenetic CV fold assignments."""
    cv_file = HOLDOUTS_DIR / f"{condition}_cv_sets.json"
    with open(cv_file) as f:
        cv_dict = json.load(f)
    cv_sets = []
    for fold in cv_dict[taxlevel]:
        cv_sets.append((np.array(fold[0]), np.array(fold[1])))
    return cv_sets


def load_train_test_sets(condition: str):
    """Load pre-computed train/test genome accession lists."""
    train_file = HOLDOUTS_DIR / f"train_set_{condition}.txt"
    test_file = HOLDOUTS_DIR / f"test_set_{condition}.txt"
    with open(train_file) as f:
        train_set = [line.strip() for line in f if line.strip()]
    with open(test_file) as f:
        test_set = [line.strip() for line in f if line.strip()]
    return train_set, test_set


def evaluate_condition(df, condition, build_pipeline_fn):
    """Run 5-fold phylogenetic CV and test evaluation for one condition.

    Args:
        df: Full training dataframe (features + labels)
        condition: One of 'oxygen', 'temperature', 'salinity', 'ph'
        build_pipeline_fn: Function from train.py that returns (pipeline, feature_list)

    Returns:
        dict with cv_score, test_score, cv_rmse (for regression), etc.
    """
    target_col = CONDITION_TO_TARGET[condition]
    is_classification = condition in CLASSIFICATION_CONDITIONS

    # Get pipeline and features for this condition
    pipeline, feature_list = build_pipeline_fn(condition)

    # Load splits
    train_accessions, test_accessions = load_train_test_sets(condition)
    cv_sets = load_cv_sets(condition)

    # Filter to genomes present in our data with valid targets
    available = set(df.index)
    train_accessions = [a for a in train_accessions if a in available]
    test_accessions = [a for a in test_accessions if a in available]

    df_train = df.loc[train_accessions].dropna(subset=[target_col])
    df_test = df.loc[test_accessions].dropna(subset=[target_col])

    # Check features exist
    missing_features = [f for f in feature_list if f not in df.columns]
    if missing_features:
        print(f"WARNING [{condition}]: Missing features: {missing_features[:5]}...", file=sys.stderr)
        # Fall back to available features
        feature_list = [f for f in feature_list if f in df.columns]
        if not feature_list:
            return {"cv_score": 0.0, "test_score": 0.0, "error": "no valid features"}

    X_train = df_train[feature_list].values
    y_train = df_train[target_col].values
    X_test = df_test[feature_list].values
    y_test = df_test[target_col].values

    # 5-fold phylogenetic CV
    # CV sets use integer indices into the training set, not accession strings
    cv_scores = []
    cv_rmses = []
    train_index_list = list(df_train.index)
    n_train = len(train_index_list)

    for fold_train_idx, fold_val_idx in cv_sets:
        # Filter indices to valid range for our training set
        fold_train_idx = [int(i) for i in fold_train_idx if int(i) < n_train]
        fold_val_idx = [int(i) for i in fold_val_idx if int(i) < n_train]

        if len(fold_train_idx) < 10 or len(fold_val_idx) < 5:
            continue

        X_fold_train = df_train.iloc[fold_train_idx][feature_list].values
        y_fold_train = df_train.iloc[fold_train_idx][target_col].values
        X_fold_val = df_train.iloc[fold_val_idx][feature_list].values
        y_fold_val = df_train.iloc[fold_val_idx][target_col].values

        try:
            from sklearn.base import clone
            fold_pipeline = clone(pipeline)
            fold_pipeline.fit(X_fold_train, y_fold_train)
            y_pred = fold_pipeline.predict(X_fold_val)

            if is_classification:
                if hasattr(fold_pipeline, "predict_proba"):
                    y_pred_binary = (fold_pipeline.predict_proba(X_fold_val)[:, 1] >= 0.5).astype(int)
                else:
                    y_pred_binary = y_pred
                score = f1_score(y_fold_val, y_pred_binary, zero_division=0)
            else:
                score = r2_score(y_fold_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                cv_rmses.append(rmse)

            cv_scores.append(score)
        except Exception as e:
            print(f"WARNING [{condition}] fold failed: {e}", file=sys.stderr)
            continue

    if not cv_scores:
        return {"cv_score": 0.0, "test_score": 0.0, "error": "all folds failed"}

    # Test set evaluation
    try:
        pipeline.fit(X_train, y_train)
        y_test_pred = pipeline.predict(X_test)

        if is_classification:
            if hasattr(pipeline, "predict_proba"):
                y_test_binary = (pipeline.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
            else:
                y_test_binary = y_test_pred
            test_score = f1_score(y_test, y_test_binary, zero_division=0)
        else:
            test_score = r2_score(y_test, y_test_pred)
    except Exception as e:
        print(f"WARNING [{condition}] test eval failed: {e}", file=sys.stderr)
        test_score = 0.0

    result = {
        "cv_score": float(np.mean(cv_scores)),
        "cv_score_std": float(np.std(cv_scores)),
        "test_score": float(test_score),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_list),
    }
    if cv_rmses:
        result["cv_rmse"] = float(np.mean(cv_rmses))

    return result


def main():
    """Run evaluation and output METRIC lines."""
    # Import train.py's pipeline builder
    try:
        from train import build_pipeline
    except ImportError:
        print("ERROR: Cannot import build_pipeline from train.py", file=sys.stderr)
        sys.exit(1)

    # Load data
    df = load_training_data()
    print(f"Loaded {len(df)} genomes with {len(df.columns)} features", file=sys.stderr)

    # Evaluate each condition
    results = {}
    for condition in CONDITIONS:
        print(f"Evaluating {condition}...", file=sys.stderr)
        result = evaluate_condition(df, condition, build_pipeline)
        results[condition] = result

        metric_name = "F1" if condition in CLASSIFICATION_CONDITIONS else "R2"
        print(f"  {condition}: CV {metric_name}={result['cv_score']:.4f} "
              f"(+/-{result.get('cv_score_std', 0):.4f}), "
              f"Test={result['test_score']:.4f}", file=sys.stderr)

    # Compute composite score (pH weighted 2x since it has most room to improve)
    weights = {"oxygen": 1.0, "temperature": 1.0, "salinity": 1.0, "ph": 2.0}
    composite = sum(results[c]["cv_score"] * weights[c] for c in CONDITIONS) / sum(weights.values())

    # Output METRIC lines for autoresearch
    print(f"METRIC composite_score={composite:.6f}")
    print(f"METRIC oxygen_f1={results['oxygen']['cv_score']:.6f}")
    print(f"METRIC temperature_r2={results['temperature']['cv_score']:.6f}")
    print(f"METRIC salinity_r2={results['salinity']['cv_score']:.6f}")
    print(f"METRIC ph_r2={results['ph']['cv_score']:.6f}")

    if "cv_rmse" in results.get("temperature", {}):
        print(f"METRIC temperature_rmse={results['temperature']['cv_rmse']:.4f}")
    if "cv_rmse" in results.get("salinity", {}):
        print(f"METRIC salinity_rmse={results['salinity']['cv_rmse']:.4f}")
    if "cv_rmse" in results.get("ph", {}):
        print(f"METRIC ph_rmse={results['ph']['cv_rmse']:.4f}")

    # Test scores as secondary metrics
    print(f"METRIC test_oxygen_f1={results['oxygen']['test_score']:.6f}")
    print(f"METRIC test_temperature_r2={results['temperature']['test_score']:.6f}")
    print(f"METRIC test_salinity_r2={results['salinity']['test_score']:.6f}")
    print(f"METRIC test_ph_r2={results['ph']['test_score']:.6f}")

    return composite


if __name__ == "__main__":
    score = main()
    print(f"\nComposite score: {score:.6f}", file=sys.stderr)
