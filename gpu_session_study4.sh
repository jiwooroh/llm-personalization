#!/usr/bin/env bash
# Study 4 (final scope): do LLMs pick up implied preferences in multi-turn chat, beyond surface similarity?
# Multiple-choice only, 5 conditions, one GPU (24GB+, disk 60GB+), ~1h for both models.
# Usage on the instance, from the repo root:
#   read -s HF_TOKEN && export HF_TOKEN            # only needed for Llama (gated)
#   PYTHONUNBUFFERED=1 bash gpu_session_study4.sh 2>&1 | tee study4.log
# Skip a model with SKIP=qwen or SKIP=llama. Smaller test first: N_ITEMS=100 bash gpu_session_study4.sh
set -euo pipefail
SKIP=${SKIP:-}
N_ITEMS=${N_ITEMS:-800}
BATCH=${BATCH:-8}
skip() { [[ ",$SKIP," == *",$1,"* ]]; }

# certificates: some hosts break Python's bundle while curl works
[ -f /etc/ssl/certs/ca-certificates.crt ] && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
pip install -q -r requirements.txt
df -h . | tail -1

cd src/implicit
python build_items.py --n_items "$N_ITEMS" --out ../../runs/implicit/items.jsonl
python embed_baseline.py --items ../../runs/implicit/items.jsonl

# sanity: the model must answer a trivial question (a corrupt download loads fine but babbles)
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

run_model() {   # $1 model id, $2 run name
  sanity "$1" || { echo "!! $1 failed the sanity check (corrupt weights?) -> skipping"; return 0; }
  python extract.py --model "$1" --items ../../runs/implicit/items.jsonl --out "../../runs/implicit/$2" --batch_size "$BATCH" \
    --mc_only --conds none,explicit,implicit,explicit_far,implicit_far
  python analyze_mc.py --run "../../runs/implicit/$2"
}

skip qwen || run_model Qwen/Qwen2.5-7B-Instruct qwen7b

if ! skip llama; then
  LLAMA=meta-llama/Llama-3.1-8B-Instruct
  if python -c "from huggingface_hub import hf_hub_download as d; d('$LLAMA', 'config.json')" 2>/dev/null; then
    # free disk before the second model if space is short
    AVAIL=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
    if [ "${AVAIL:-0}" -lt 25 ]; then echo "low disk (${AVAIL}G) -> clearing HF cache"; rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub"; fi
    run_model "$LLAMA" llama8b
  else
    echo "!! no access to $LLAMA (HF_TOKEN / license) -> skipping Llama"
  fi
fi

cd ../..
tar czf study4_results.tar.gz --exclude='H.npy' --exclude='personamem_v2_benchmark.csv' runs/implicit $( [ -f study4.log ] && echo study4.log )
ls -lh study4_results.tar.gz
echo "STUDY4_DONE"
