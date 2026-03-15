#!/bin/bash
set -euo pipefail

# Activate venv
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# Pre-check: verify train.py has valid syntax
python -c "import py_compile; py_compile.compile('train.py', doraise=True)" 2>&1 || {
    echo "METRIC composite_score=0"
    exit 1
}

# Run the evaluation harness
python prepare.py
