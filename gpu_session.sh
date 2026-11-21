#!/usr/bin/env bash
# One vast.ai GPU session (24GB+ GPU, 80GB disk, ~3h total):
#   B1  Qwen2.5-7B Study 1 rerun  -> user-level bootstrap (user_bootstrap.json)
#   B2  Llama-3.1-8B Study 1 replication (checked first; skipped if there is no gated access)
#   A   Study 3: stated taste profile x text pair -> do hidden states / A-B logits predict real choices?
# Before running, copy LiteraryTaste/ from the Mac (it is not in git):
#   scp -P <port> -r LiteraryTaste root@<host>:~/llm-personalization/
# Usage on the instance:
#   git clone https://github.com/jiwooroh/llm-personalization.git && cd llm-personalization
#   read -s HF_TOKEN && export HF_TOKEN   # only for Llama; paste the token (hidden). See GPU_GUIDE.md
#   bash gpu_session.sh 2>&1 | tee session.log
# Then copy results_small.tar.gz to the Mac and DESTROY the instance.
# Skip parts with SKIP=B1,B2 bash gpu_session.sh
set -euo pipefail
SKIP=${SKIP:-}
skip() { [[ ",$SKIP," == *",$1,"* ]]; }

pip install -q -r requirements.txt
mkdir -p data
[ -f data/survey.jsonl ] || curl -L -o data/survey.jsonl \
  https://raw.githubusercontent.com/HannahKirk/prism-alignment/main/data/survey.jsonl

# Llama preflight: check gated access now, not after B1 has run for an hour.
# Token comes from HF_TOKEN or from `hf auth login` (~/.cache/huggingface/token).
LLAMA=meta-llama/Llama-3.1-8B-Instruct
RUN_LLAMA=0
if ! skip B2; then
  if python -c "from huggingface_hub import hf_hub_download as d; d('$LLAMA', 'config.json')" 2>/dev/null; then
    RUN_LLAMA=1
    echo "Llama access OK -> B2 will run"
  else
    echo "!! No access to $LLAMA (token missing, or license not approved yet) -> skipping B2"
    echo "   Check: https://huggingface.co/settings/gated-repos"
  fi
fi

skip B1 || bash run_all.sh Qwen/Qwen2.5-7B-Instruct runs/qwen7b
if [ "$RUN_LLAMA" = 1 ]; then bash run_all.sh "$LLAMA" runs/llama8b; fi

if ! skip A; then
  if [ -f LiteraryTaste/annotated_instances.csv ]; then
    (cd src/literary &&
      python make_profiles.py --data ../../LiteraryTaste --out ../../runs/literary_taste &&
      python extract_joint.py --model Qwen/Qwen2.5-7B-Instruct --batch_size 16 \
             --out ../../runs/literary_taste/joint_qwen7b &&
      python probe_joint.py --run ../../runs/literary_taste/joint_qwen7b)
  else
    echo "LiteraryTaste/ missing -> skipping Study 3 (scp it from the Mac first)"
  fi
fi

# small results only (no H_*.npy, which are several GB)
tar czf results_small.tar.gz --exclude='H_*.npy' \
  $(for d in runs/qwen7b runs/llama8b runs/literary_taste/joint_qwen7b; do [ -d "$d" ] && echo "$d"; done) \
  $( [ -f session.log ] && echo session.log )
ls -lh results_small.tar.gz
echo "Copy to Mac:  scp -P <port> root@<host>:~/llm-personalization/results_small.tar.gz ."
