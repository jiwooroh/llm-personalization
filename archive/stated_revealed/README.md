# Stated vs Revealed Preferences in an LLM's Residual Stream (PRISM)

Solo research project (Jiwoo Roh). Working title:
*Stated Priorities, Revealed Representations: What an LLM's Residual Stream Encodes About What People Reward.*

Design notes and literature: Obsidian `03_Projects/PRISM 개인화 이득/1113_Solo 전환 - 리서치 브리프`.

## Pipeline

| step | script | GPU | output |
|---|---|---|---|
| 0 | download PRISM jsonl (below) | no | `data/prism/*.jsonl` |
| 1 | `src/prepare_data.py` | no | `data/processed/responses.parquet`, `pairs.parquet` (user-level 70/15/15 split) |
| 2 | `src/extract_acts.py` | **yes** (48GB class) | `data/acts/<model>/mean.npy` (+`last.npy`) — per-layer response-token means |
| 3 | `src/probes.py` | optional (fast on GPU) | `results/<model>/probe_results.csv`, `dirs_LXX.npz` |
| 4 | steering + evaluation | yes | *(next step — not written yet)* |

Status (2026-11-13): steps 0–3 implemented and smoke-tested on CPU with Qwen2.5-0.5B-Instruct (400 responses).
Smoke-test numbers are meaningless (75 eval pairs); they only show the code runs end-to-end.

## Data facts (computed 2026-11-13)

- 30,049 opening-turn responses, 1,396 users, 8,011 interactions
- 31,421 within-prompt pairs with score gap ≥ 10 (train 22,137 / dev 4,837 / test 4,447)
- Winner has the larger surface feature: longer 62.8%, list format 73.2%, headers/bold 58.0%, refusal/disclaimer 45.0%, hedging 61.9%

## Runbook on vast.ai

1. **Before renting**: request access to `meta-llama/Llama-3.1-8B-Instruct` on HuggingFace (accept the license on the model page) and create a read token.
2. **Rent**: 1× 48GB GPU (RTX A6000 / L40S / A40), on-demand, disk 120 GB, PyTorch template image.
3. On the instance:
```bash
git clone <this repo> && cd stated_revealed      # or scp the folder up
pip install -r requirements.txt
export HF_TOKEN=hf_...                            # never commit this
mkdir -p data/prism && for f in utterances conversations survey; do
  curl -L -o data/prism/$f.jsonl https://raw.githubusercontent.com/HannahKirk/prism-alignment/main/data/$f.jsonl; done
python src/prepare_data.py
python src/extract_acts.py --model meta-llama/Llama-3.1-8B-Instruct --out data/acts/llama31_8b --batch_size 16
python src/probes.py --acts data/acts/llama31_8b --out results/llama31_8b
```
4. Download `results/` (small). `mean.npy` is ~8 GB (30k × 33 layers × 4096 × fp16) — keep it on the instance disk only if you will run more analyses; otherwise **delete the instance** (storage is billed even when stopped).

Expected time on a 48GB card: extraction ~1 h, probes ~30–60 min for all 33 layers.

## Analysis plan (RQs)

- **RQ1** Can a linear probe on (h_win − h_lose) predict which response a person rated higher (held-out users)? Compare with the surface-only baseline and published PRISM numbers (PersonalRewardBench: LLM judge 52.8%, fine-tuned BT RM 64.3% — different task setup, cite with care).
- **RQ2** Cosine between the preference direction and surface directions (length, list, header/bold, refusal, hedging) vs perceived-attribute directions (ridge on PRISM `performance_attributes` of chosen responses). Probe accuracy after erasing the surface subspace, vs erasing a random subspace of equal size.
- **RQ3** Steering with the preference direction vs its surface-erased version vs the length direction vs random; measure length/format change and quality (reward model + length-controlled analysis).
- **RQ4 (exploratory)** user subgroups by stated priorities / `system_string`.

## Known caveats to state in the paper

- `performance_attributes` are conversation-level ratings attached here to the chosen opening response — confirm against the PRISM codebook.
- Surface features are regex-based approximations.
- Pairs share a user and a prompt; the split is by user so no user appears in both train and eval.
