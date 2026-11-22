#!/usr/bin/env bash
# Study 4b: does implicit-preference use grow with model size, or is it a model-family difference?
# Same 800 items / 5 conditions / multiple choice as gpu_session_study4.sh; Qwen-7B and Llama-8B are already done.
# GPU: 48GB (A6000 / L40S / A100) because Qwen2.5-14B in bf16 needs ~30GB. Disk 80GB. ~1.5h.
# Llama-3.2 needs its own license approval on Hugging Face (separate from 3.1).
# Usage on the instance, from the repo root, inside tmux:
#   read -s HF_TOKEN && export HF_TOKEN
#   PYTHONUNBUFFERED=1 bash gpu_session_study4_scale.sh 2>&1 | tee study4_scale.log
# Pick models with MODELS="Qwen/Qwen2.5-3B-Instruct:qwen3b,..."
set -euo pipefail
BATCH=${BATCH:-8}
MODELS=${MODELS:-"Qwen/Qwen2.5-1.5B-Instruct:qwen1_5b,Qwen/Qwen2.5-3B-Instruct:qwen3b,meta-llama/Llama-3.2-1B-Instruct:llama1b,meta-llama/Llama-3.2-3B-Instruct:llama3b,Qwen/Qwen2.5-14B-Instruct:qwen14b"}

[ -f /etc/ssl/certs/ca-certificates.crt ] && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
pip install -q -r requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h . | tail -1

cd src/implicit
[ -f ../../runs/implicit/items.jsonl ] || python build_items.py --n_items 800 --out ../../runs/implicit/items.jsonl
[ -f ../../runs/implicit/embed_baseline.json ] || python embed_baseline.py --items ../../runs/implicit/items.jsonl

sanity() {
  python - "$1" <<'PY'
import sys
sys.path.insert(0, "..")
from common import load_model
tok, m = load_model(sys.argv[1])
x = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France? One word."}],
                            add_generation_prompt=True, return_tensors="pt", return_dict=True).to(m.device)
out = tok.decode(m.generate(**x, max_new_tokens=8, do_sample=False)[0, x["input_ids"].shape[1]:])
print("sanity:", repr(out)); sys.exit(0 if "Paris" in out else 1)
PY
}

IFS=',' read -ra LIST <<< "$MODELS"
for entry in "${LIST[@]}"; do
  model=${entry%%:*}; name=${entry##*:}
  echo "=== $model -> $name"
  if [ -f "../../runs/implicit/$name/results_mc.json" ]; then echo "already done"; continue; fi
  sanity "$model" || { echo "!! $model failed to load or the sanity check (license? weights?) -> skipping"; continue; }
  python extract.py --model "$model" --items ../../runs/implicit/items.jsonl --out "../../runs/implicit/$name" \
    --batch_size "$BATCH" --mc_only --conds none,explicit,implicit,explicit_far,implicit_far
  python analyze_mc.py --run "../../runs/implicit/$name" > /dev/null
  python -c "import json; print(json.load(open('../../runs/implicit/$name/results_mc.json'))['accuracy'])"
  rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub"   # free disk for the next model
done

cd ../..
tar czf study4_scale_results.tar.gz --exclude='items.jsonl' --exclude='personamem_v2_benchmark.csv' --exclude='*_lens.npy' \
  runs/implicit/*/meta.json runs/implicit/*/MC.npy runs/implicit/*/results_mc.json $( [ -f study4_scale.log ] && echo study4_scale.log )
ls -lh study4_scale_results.tar.gz
echo "STUDY4_SCALE_DONE"
