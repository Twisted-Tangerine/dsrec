#!/usr/bin/env bash
# -------------------------------------------------------------
# One-click preprocessing script for the Yelp dataset.
# Usage:
#   1) cd PreProcess           # project root requirement
#   2) bash scripts/yelp_process_1.sh [options]
# Any command-line options will be forwarded to data_process.py, e.g.:
#   bash scripts/yelp_process_1.sh --user_core 5 --item_core 5
# -------------------------------------------------------------
set -e

# Resolve the directory this script lives in → PreProcess/scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Move to PreProcess root to ensure relative imports work
cd "$SCRIPT_DIR/.."

# Launch the Yelp preprocessing module (relative path, no hard-coded dirs)
python -m filter.yelp.data_process "$@"
