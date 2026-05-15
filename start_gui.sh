#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="one-step"
if [[ "${1:-}" == "two-step" || "${1:-}" == "--two-step" ]]; then
  MODE="two-step"
  shift
elif [[ "${1:-}" == "one-step" || "${1:-}" == "--one-step" ]]; then
  shift
fi

exec .venv/bin/python scripts/live_fall_gui.py --mode "$MODE" --source 0 "$@"
