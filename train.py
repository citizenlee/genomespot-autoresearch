"""Model pipeline definitions for GenomeSPOT autoresearch.

THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.

The agent can change:
- Feature sets and localizations
- Model types and hyperparameters
- Feature selection methods
- Preprocessing steps
- Derived feature computations
- Ensemble strategies
"""

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# Base amino acid features (20 standard)
BASE_AAS = [f"aa_{aa}" for aa in "A C D E F G H I K L M N P Q R S T V W Y".split()]

# Isoelectric point distribution bins
BASE_PIS = [f"pis_{lo}_{hi}" for lo, hi in
            [(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12)]]

# Derived protein metrics
BASE_DERIVED_PROTEIN = [
    "mean_gravy", "mean_nh2o", "mean_pi", "mean_zc",
    "proportion_R_RK", "mean_thermostable_freq",
]

# Derived genome metrics
BASE_DERIVED_GENOME = [
    "nt_C", "pur_pyr_transition_freq",
    "protein_coding_density", "mean_protein_length",
]

# Compartments for localization-aware features
COMPARTMENTS = ["extracellular_soluble", "intracellular_soluble", "membrane"]


def _prepend(features, prefixes):
    """Add localization prefix to feature names."""
    return [f"{prefix}_{feat}" for prefix in prefixes for feat in features]


# ============================================================
# PIPELINE BUILDER — this is what the agent modifies
# ============================================================

def build_pipeline(condition: str):
    """Build a model pipeline and feature list for the given condition.

    This function is called by prepare.py for each condition.

    Args:
        condition: One of 'oxygen', 'temperature', 'salinity', 'ph'

    Returns:
        pipeline: sklearn Pipeline ready to fit
        feature_list: list of feature column names to use from the dataframe
    """
    if condition == "oxygen":
        # --- OXYGEN (classification) ---
        # Baseline: LogisticRegression on 20 AA features, no localization
        features = _prepend(BASE_AAS, ["all"])
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=50000, solver="lbfgs")),
        ])

    elif condition == "temperature":
        # --- TEMPERATURE (regression) ---
        # Baseline: Lasso on 60 AA features (3 compartments)
        features = _prepend(BASE_AAS, COMPARTMENTS)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=50000)),
        ])

    elif condition == "salinity":
        # --- SALINITY (regression) ---
        # Baseline: Lasso on 60 AA features (3 compartments)
        features = _prepend(BASE_AAS, COMPARTMENTS)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=50000)),
        ])

    elif condition == "ph":
        # --- PH (regression) ---
        # Baseline: Lasso on 60 AA features (3 compartments)
        features = _prepend(BASE_AAS, COMPARTMENTS)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=50000)),
        ])

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return pipeline, features
