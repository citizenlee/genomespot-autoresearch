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
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
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
        # AA + pI + derived from "all" compartment, higher C (less reg, N=7300)
        features = (
            _prepend(BASE_AAS, ["all"])
            + _prepend(BASE_PIS, ["all"])
            + _prepend(BASE_DERIVED_PROTEIN, ["all"])
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=10.0, max_iter=50000, solver="lbfgs")),
        ])

    elif condition == "temperature":
        # --- TEMPERATURE (regression) ---
        # LassoCV auto alpha; 60 AA + thermostable_freq
        features = (
            _prepend(BASE_AAS, COMPARTMENTS)
            + ["all_mean_thermostable_freq", "intracellular_soluble_mean_thermostable_freq"]
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LassoCV(cv=5, max_iter=50000)),
        ])

    elif condition == "salinity":
        # --- SALINITY (regression) ---
        # LassoCV for auto alpha; 60 AA + pI features
        features = (
            _prepend(BASE_AAS, COMPARTMENTS)
            + ["intracellular_soluble_pis_4_5", "membrane_pis_4_5", "all_pis_4_5"]
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LassoCV(cv=5, max_iter=50000)),
        ])

    elif condition == "ph":
        # --- PH (regression) ---
        # Baseline 60 AA + 3 high-corr diff_extra_intra features (r=0.38-0.46)
        features = (
            _prepend(BASE_AAS, COMPARTMENTS)
            + ["diff_extra_intra_aa_E", "diff_extra_intra_aa_D",
               "diff_extra_intra_mean_thermostable_freq"]
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=50000)),
        ])

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return pipeline, features
