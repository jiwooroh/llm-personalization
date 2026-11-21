"""Extract residual-stream activations for every PRISM opening-turn response, read in context
(chat template: user prompt -> assistant response). Saves, for every layer, the mean over the
response tokens (and optionally the last response token) as float16 memmaps.

Usage (GPU):
    python src/extract_acts.py --model meta-llama/Llama-3.1-8B-Instruct --out data/acts/llama31_8b
Quick CPU smoke test:
    python src/extract_acts.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 64 --out data/acts/smoke
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def build(tok, prompt, response, max_len):
    msgs = [{"role": "user", "content": prompt}]
    p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": response}], tokenize=True)
    if hasattr(p_ids, "input_ids"):
        p_ids, full = p_ids["input_ids"], full["input_ids"]
    start = len(p_ids)
    assert full[:start] == list(p_ids), "chat template prefix mismatch"
    # response span = tokens after the generation prompt, minus trailing special tokens (eot etc.)
    end = len(full)
    while end > start and full[end - 1] in tok.all_special_ids:
        end -= 1
    full = full[:max_len]
    end = min(end, max_len)
    return full, start, max(end, start + 1)


@torch.no_grad()
def main(a):
    os.makedirs(a.out, exist_ok=True)
    resp = pd.read_parquet(a.responses)
    if a.limit:
        resp = resp.iloc[: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1  # embeddings + each block output
    d = model.config.hidden_size
    N = len(resp)
    mean_mm = np.lib.format.open_memmap(os.path.join(a.out, "mean.npy"), mode="w+", dtype=np.float16, shape=(N, n_layers, d))
    last_mm = np.lib.format.open_memmap(os.path.join(a.out, "last.npy"), mode="w+", dtype=np.float16, shape=(N, n_layers, d)) if a.save_last else None

    items = [build(tok, p, r, a.max_len) for p, r in zip(resp.user_prompt, resp.model_response)]
    order = np.argsort([len(x[0]) for x in items])  # length-sorted batching
    dev = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    n_tok = []
    for b0 in tqdm(range(0, N, a.batch_size)):
        idx = order[b0: b0 + a.batch_size]
        L = max(len(items[i][0]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.float32)
        for k, i in enumerate(idx):
            seq, s, e = items[i]
            ids[k, : len(seq)] = torch.tensor(seq)  # right padding
            att[k, : len(seq)] = 1
            mask[k, s:e] = 1
            n_tok.append(e - s)
        out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True)
        m = mask.to(out.hidden_states[0].device)[:, :, None]          # [B, L, 1]
        denom = m.sum(1).clamp(min=1)
        last_pos = torch.tensor([items[i][2] - 1 for i in idx], device=m.device)
        mean = np.empty((len(idx), n_layers, d), dtype=np.float16)
        last = np.empty((len(idx), n_layers, d), dtype=np.float16) if last_mm is not None else None
        for li, h in enumerate(out.hidden_states):                     # layer by layer: low memory
            hf = h.float()
            mean[:, li] = ((hf * m).sum(1) / denom).cpu().numpy().astype(np.float16)
            if last is not None:
                last[:, li] = hf[torch.arange(len(idx), device=m.device), last_pos].cpu().numpy().astype(np.float16)
        del out
        mean_mm[idx] = mean
        if last_mm is not None:
            last_mm[idx] = last
    mean_mm.flush()
    json.dump({"model": a.model, "n": N, "n_layers": n_layers, "d": d, "max_len": a.max_len,
               "mean_response_tokens": float(np.mean(n_tok))}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("saved", a.out, "| mean response tokens:", round(float(np.mean(n_tok)), 1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--responses", default="data/processed/responses.parquet")
    p.add_argument("--out", default="data/acts/llama31_8b")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=768)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save_last", action="store_true")
    main(p.parse_args())
