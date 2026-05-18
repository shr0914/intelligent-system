#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec scripts/generate_report_assets.sh "$@"
