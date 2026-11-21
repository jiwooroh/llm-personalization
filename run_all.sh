#!/usr/bin/env bash
# Full Study 1 pipeline on one GPU. Usage: bash run_all.sh [MODEL] [RUN_DIR]
set -euo pipefail
MODEL=${1:-Qwen/Qwen2.5-7B-Instruct}
RUN=${2:-runs/qwen7b}
cd src
python extract_acts.py     --model "$MODEL" --out "../$RUN"
python probe.py            --run "../$RUN"          # also writes probe_oof.npz (per-user predictions)
python verbal_extract.py   --model "$MODEL" --run "../$RUN"
python minimal_pairs.py    --model "$MODEL" --run "../$RUN"
python verbal_calibrated.py --run "../$RUN"
python user_bootstrap.py   --run "../$RUN"          # user-level paired bootstrap (CPU, ~1 min)
python make_figures.py     --run "../$RUN"
echo "done -> $RUN (probe_summary.json, verbal_summary.json, minimal_summary.json, user_bootstrap.json, figures/)"
