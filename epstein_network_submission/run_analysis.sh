#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 src/analyze_epstein_graph.py --data_dir data --output_dir outputs
