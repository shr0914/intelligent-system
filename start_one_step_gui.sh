#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec .venv/bin/python scripts/live_fall_gui.py --mode one-step --source 0 "$@"
