#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

MODE="one-step"
if [[ "${1:-}" == "two-step" || "${1:-}" == "--two-step" ]]; then
  MODE="two-step"
  shift
elif [[ "${1:-}" == "one-step" || "${1:-}" == "--one-step" ]]; then
  shift
fi

exec "$PYTHON" scripts/run_gui.py --mode "$MODE" --source 0 "$@"
