#!/usr/bin/env bash
# -------------------------------------------------------------
# One-click preprocessing script for the Grocery dataset.
# Usage:
#   1) cd PreProcess           # reviewer requirement
#   2) bash scripts/grocery_process_1.sh [options]
# Any command-line options will be forwarded to data_process.py, e.g.:
#   bash scripts/grocery_process_1.sh --user_core 8 --item_core 8
# -------------------------------------------------------------
set -e

# Resolve the directory this script lives in → PreProcess/scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Move to PreProcess root to ensure relative imports work
cd "$SCRIPT_DIR/.."

# Launch the Grocery preprocessing module (relative path, no hard-coded dirs)
python -m filter.grocery.data_process "$@"
