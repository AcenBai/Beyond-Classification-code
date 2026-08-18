#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m co_pretraining.run "$@"
