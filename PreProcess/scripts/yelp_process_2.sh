#!/usr/bin/env bash
# -------------------------------------------------------------
# One-click script to execute the Yelp prompt-construction notebook.
# Prerequisite: yelp_process_1.sh has already produced files in
#   PreProcess/output/yelp/
# Usage (after cd PreProcess):
#   bash scripts/yelp_process_2.sh
# -------------------------------------------------------------
set -e

# Resolve script directory and switch to PreProcess root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."   # now in PreProcess/

NOTEBOOK_PATH="filter/yelp/prompt_process.ipynb"

if ! command -v jupyter &> /dev/null; then
  echo "Error: jupyter command not found. Please install JupyterLab/Notebook."
  exit 1
fi

# Execute the notebook in-place, keeping outputs; no timeout restriction
jupyter nbconvert --to notebook --execute "$NOTEBOOK_PATH" \
  --inplace --ExecutePreprocessor.timeout=-1

echo "\n✅ prompt_process_yelp.ipynb executed successfully."
