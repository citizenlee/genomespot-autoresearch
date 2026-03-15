"""Data preparation for GenomeSPOT autoresearch.

This script generates the training_data.tsv feature matrix needed for the
autoresearch loop. It supports multiple data sources:

1. Pre-computed: If data/training_data.tsv exists, skip
2. From GenomeSPOT repo: If a path to a GenomeSPOT training run's output is provided
3. From scratch: Download BacDive data + compute features from genome FASTA files

Usage:
    # Check if data is ready
    python prepare_data.py --check

    # Use pre-computed feature matrix from a GenomeSPOT training run
    python prepare_data.py --from-tsv /path/to/training_data.tsv

    # Compute from scratch (requires BacDive credentials + genome files)
    python prepare_data.py --bacdive-user USER --bacdive-pass PASS --genomes-dir /path/to/genomes/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
HOLDOUTS_DIR = DATA_DIR / "holdouts"
TRAINING_DATA_FILE = DATA_DIR / "training_data.tsv"

# All features that should be in the training data
EXPECTED_FEATURES = {
    # Amino acid frequencies per localization
    **{f"{loc}_aa_{aa}": True
       for loc in ["all", "extracellular_soluble", "intracellular_soluble", "membrane", "diff_extra_intra"]
       for aa in "A C D E F G H I K L M N P Q R S T V W Y".split()},
    # Isoelectric point distributions per localization
    **{f"{loc}_pis_{lo}_{hi}": True
       for loc in ["all", "extracellular_soluble", "intracellular_soluble", "membrane", "diff_extra_intra"]
       for lo, hi in [(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12)]},
    # Derived protein metrics per localization
    **{f"{loc}_{feat}": True
       for loc in ["all", "extracellular_soluble", "intracellular_soluble", "membrane", "diff_extra_intra"]
       for feat in ["mean_gravy", "mean_nh2o", "mean_pi", "mean_zc", "proportion_R_RK", "mean_thermostable_freq"]},
    # Derived genome metrics (all prefix only)
    **{f"all_{feat}": True
       for feat in ["nt_C", "pur_pyr_transition_freq", "protein_coding_density", "mean_protein_length"]},
    # Target variables
    "oxygen": True,
    "temperature_optimum": True,
    "temperature_min": True,
    "temperature_max": True,
    "salinity_optimum": True,
    "salinity_min": True,
    "salinity_max": True,
    "ph_optimum": True,
    "ph_min": True,
    "ph_max": True,
}


def check_data():
    """Check if training data is ready."""
    if not TRAINING_DATA_FILE.exists():
        print(f"MISSING: {TRAINING_DATA_FILE}")
        print("Run: python prepare_data.py --from-tsv /path/to/training_data.tsv")
        print("Or:  python prepare_data.py --from-scratch ...")
        return False

    df = pd.read_csv(TRAINING_DATA_FILE, sep="\t", index_col=0, nrows=5)
    print(f"OK: {TRAINING_DATA_FILE}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Sample rows: {len(df)}")

    # Check holdout files
    for condition in ["oxygen", "temperature", "salinity", "ph"]:
        for split in ["train_set", "test_set"]:
            f = HOLDOUTS_DIR / f"{split}_{condition}.txt"
            if f.exists():
                with open(f) as fh:
                    n = sum(1 for line in fh if line.strip())
                print(f"  OK: {f.name} ({n} genomes)")
            else:
                print(f"  MISSING: {f}")
                return False
        cv_f = HOLDOUTS_DIR / f"{condition}_cv_sets.json"
        if cv_f.exists():
            print(f"  OK: {cv_f.name}")
        else:
            print(f"  MISSING: {cv_f}")
            return False

    # Full row count
    df_full = pd.read_csv(TRAINING_DATA_FILE, sep="\t", index_col=0)
    print(f"\nFull dataset: {len(df_full)} genomes x {len(df_full.columns)} columns")

    # Check coverage per condition
    for condition, target in [("oxygen", "oxygen"), ("temperature", "temperature_optimum"),
                               ("salinity", "salinity_optimum"), ("ph", "ph_optimum")]:
        if target in df_full.columns:
            n_valid = df_full[target].notna().sum()
            print(f"  {condition}: {n_valid} genomes with valid labels")

    return True


def from_tsv(tsv_path: str):
    """Copy a pre-computed training data TSV."""
    src = Path(tsv_path)
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, TRAINING_DATA_FILE)
    print(f"Copied {src} -> {TRAINING_DATA_FILE}")

    # Verify
    df = pd.read_csv(TRAINING_DATA_FILE, sep="\t", index_col=0)
    print(f"Loaded {len(df)} genomes x {len(df.columns)} columns")
    check_data()


def generate_synthetic_data(n_genomes: int = 2000):
    """Generate synthetic training data for testing the pipeline.

    This creates FAKE data with realistic feature distributions so the
    autoresearch loop can be tested before real data is available.
    Results from synthetic data are NOT scientifically meaningful.
    """
    print(f"Generating {n_genomes} synthetic genomes for pipeline testing...")
    rng = np.random.RandomState(42)

    # Load real genome accessions from holdout files to use as indices
    all_accessions = set()
    for condition in ["oxygen", "temperature", "salinity", "ph"]:
        for split in ["train_set", "test_set"]:
            f = HOLDOUTS_DIR / f"{split}_{condition}.txt"
            if f.exists():
                with open(f) as fh:
                    all_accessions.update(line.strip() for line in fh if line.strip())
    accessions = sorted(all_accessions)[:n_genomes]
    if len(accessions) < n_genomes:
        # Pad with synthetic accessions
        for i in range(n_genomes - len(accessions)):
            accessions.append(f"GCA_SYNTH_{i:06d}")

    data = {}

    # Amino acid frequencies (should sum to ~1.0 per localization)
    aas = "A C D E F G H I K L M N P Q R S T V W Y".split()
    for loc in ["all", "extracellular_soluble", "intracellular_soluble", "membrane", "diff_extra_intra"]:
        if loc == "diff_extra_intra":
            for aa in aas:
                data[f"{loc}_aa_{aa}"] = rng.normal(0, 0.01, n_genomes)
        else:
            freqs = rng.dirichlet(np.ones(20) * 10, n_genomes)
            for i, aa in enumerate(aas):
                data[f"{loc}_aa_{aa}"] = freqs[:, i]

    # pI distributions
    for loc in ["all", "extracellular_soluble", "intracellular_soluble", "membrane", "diff_extra_intra"]:
        for lo, hi in [(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12)]:
            if loc == "diff_extra_intra":
                data[f"{loc}_pis_{lo}_{hi}"] = rng.normal(0, 0.02, n_genomes)
            else:
                data[f"{loc}_pis_{lo}_{hi}"] = rng.beta(2, 5, n_genomes) * 0.3

    # Derived protein metrics
    for loc in ["all", "extracellular_soluble", "intracellular_soluble", "membrane", "diff_extra_intra"]:
        data[f"{loc}_mean_gravy"] = rng.normal(-0.3, 0.15, n_genomes)
        data[f"{loc}_mean_nh2o"] = rng.normal(0.35, 0.05, n_genomes)
        data[f"{loc}_mean_pi"] = rng.normal(7.0, 1.5, n_genomes)
        data[f"{loc}_mean_zc"] = rng.normal(-0.15, 0.05, n_genomes)
        data[f"{loc}_proportion_R_RK"] = rng.beta(5, 5, n_genomes)
        data[f"{loc}_mean_thermostable_freq"] = rng.normal(0.45, 0.05, n_genomes)

    # Derived genome metrics
    data["all_nt_C"] = rng.normal(0.25, 0.08, n_genomes)
    data["all_pur_pyr_transition_freq"] = rng.normal(0.5, 0.02, n_genomes)
    data["all_protein_coding_density"] = rng.normal(0.87, 0.05, n_genomes)
    data["all_mean_protein_length"] = rng.normal(300, 50, n_genomes)

    # Target variables with realistic correlations to features
    # Oxygen: correlated with cysteine frequency
    cys_freq = np.array(data["all_aa_C"])
    oxygen_prob = 1 / (1 + np.exp(-(cys_freq - 0.012) * 500 + rng.normal(0, 1, n_genomes)))
    data["oxygen"] = (oxygen_prob > 0.5).astype(float)

    # Temperature: correlated with hydrophobicity and thermostable residue freq
    gravy = np.array(data["all_mean_gravy"])
    thermo = np.array(data["all_mean_thermostable_freq"])
    data["temperature_optimum"] = 30 + gravy * 40 + thermo * 30 + rng.normal(0, 5, n_genomes)
    data["temperature_optimum"] = np.clip(data["temperature_optimum"], 0, 105)
    data["temperature_min"] = data["temperature_optimum"] - rng.uniform(5, 20, n_genomes)
    data["temperature_max"] = data["temperature_optimum"] + rng.uniform(5, 20, n_genomes)

    # Salinity: correlated with aspartic acid frequency in extracellular proteins
    asp_ext = np.array(data["extracellular_soluble_aa_D"])
    data["salinity_optimum"] = asp_ext * 100 + rng.normal(0, 2, n_genomes)
    data["salinity_optimum"] = np.clip(data["salinity_optimum"], 0, 37)
    data["salinity_min"] = np.clip(data["salinity_optimum"] - rng.uniform(0, 3, n_genomes), 0, 37)
    data["salinity_max"] = np.clip(data["salinity_optimum"] + rng.uniform(0, 5, n_genomes), 0, 37)

    # pH: weakly correlated with glutamic acid extra-intra difference
    glu_diff = np.array(data["diff_extra_intra_aa_E"])
    data["ph_optimum"] = 7.0 + glu_diff * 50 + rng.normal(0, 1.5, n_genomes)
    data["ph_optimum"] = np.clip(data["ph_optimum"], 0.5, 14)
    data["ph_min"] = np.clip(data["ph_optimum"] - rng.uniform(0.5, 2, n_genomes), 0.5, 14)
    data["ph_max"] = np.clip(data["ph_optimum"] + rng.uniform(0.5, 2, n_genomes), 0.5, 14)

    # Add NaN for some targets (mimicking real data sparsity)
    for col in ["temperature_optimum", "temperature_min", "temperature_max"]:
        mask = rng.random(n_genomes) > 0.6
        data[col] = np.where(mask, data[col], np.nan)
    for col in ["salinity_optimum", "salinity_min", "salinity_max"]:
        mask = rng.random(n_genomes) > 0.7
        data[col] = np.where(mask, data[col], np.nan)
    for col in ["ph_optimum", "ph_min", "ph_max"]:
        mask = rng.random(n_genomes) > 0.75
        data[col] = np.where(mask, data[col], np.nan)
    mask_o2 = rng.random(n_genomes) > 0.5
    data["oxygen"] = np.where(mask_o2, data["oxygen"], np.nan)

    df = pd.DataFrame(data, index=accessions)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_DATA_FILE, sep="\t")
    print(f"Wrote synthetic data: {len(df)} genomes x {len(df.columns)} columns -> {TRAINING_DATA_FILE}")
    print("\nWARNING: This is SYNTHETIC data for pipeline testing only.")
    print("Replace with real data before drawing scientific conclusions.\n")
    check_data()


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for GenomeSPOT autoresearch")
    parser.add_argument("--check", action="store_true", help="Check if data is ready")
    parser.add_argument("--from-tsv", type=str, help="Path to pre-computed training_data.tsv")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data for pipeline testing")
    parser.add_argument("--n-genomes", type=int, default=2000,
                        help="Number of synthetic genomes (default: 2000)")
    args = parser.parse_args()

    if args.check:
        ok = check_data()
        sys.exit(0 if ok else 1)
    elif args.from_tsv:
        from_tsv(args.from_tsv)
    elif args.synthetic:
        generate_synthetic_data(args.n_genomes)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python prepare_data.py --synthetic     # Test with fake data")
        print("  python prepare_data.py --from-tsv PATH # Use real pre-computed data")
        print("  python prepare_data.py --check         # Verify data is ready")


if __name__ == "__main__":
    main()
