#!/usr/bin/env bash
# -------------------------------------------------------------
# Combined cleaning & reindexing script (missing + bad ids)
# Usage: bash scripts/grocery_process_1.5.sh [--bad_json path/to/bad_ids.json]
# Must be executed after grocery_process_1.sh and before prompt_process
# -------------------------------------------------------------
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python -m filter.grocery.clean_reindex "$@"
