#!/usr/bin/env bash
# Full pipeline on one GPU. Usage: bash run_all.sh [MODEL] [RUN_DIR]
set -euo pipefail
MODEL=${1:-Qwen/Qwen2.5-7B-Instruct}
RUN=${2:-runs/qwen7b}
cd src
python extract_acts.py  --model "$MODEL" --out "../$RUN"
python probe.py         --run "../$RUN"
python verbal_extract.py --model "$MODEL" --run "../$RUN"
python minimal_pairs.py --model "$MODEL" --run "../$RUN"
python make_figures.py  --run "../$RUN"
echo "done -> $RUN (summaries: probe_summary.json, verbal_summary.json, minimal_summary.json, figures/)"
