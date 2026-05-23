#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec scripts/generate_low_light_extension.sh "$@"
