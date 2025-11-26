#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/generate_uv_lock.sh [output-file]
# Default output file: requirements-lock-uv.txt

OUT=${1:-requirements-lock-uv.txt}

# Ensure we run pip from the currently active environment
PIP_CMD=${PIP_CMD:-pip}

# List installed packages - simple freeze without problematic sed transforms
$PIP_CMD list --format=freeze > "$OUT"

# Inform the user
printf "Generated lock file: %s (%d packages)\n" "$OUT" "$(wc -l < "$OUT")"

# Note: If you prefer uv to resolve dependencies from `pyproject.toml`,
# use `uv lock` (requires a valid [project] table). The generated ~requirements
# file above captures the installed environment and may be used with `uv pip install -r`.
