"""Shared helpers: data loading, prompt framing, activation extraction."""
import json
import os
import urllib.request

import numpy as np
import pandas as pd
import torch

ATTRS = ["values", "fluency", "factuality", "safety", "diversity", "creativity", "helpfulness", "personalisation"]
PRISM_RAW = "https://raw.githubusercontent.com/HannahKirk/prism-alignment/main/data/survey.jsonl"

# The user's own words are wrapped in a fixed frame so every text is read in the same role:
# a user telling an assistant what they want. Only the user's tokens are pooled.
FRAME_PREFIX = "Here is what I want from an AI assistant: "


def download_survey(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(PRISM_RAW, path)
    return path


def load_users(survey_path, field="system_string", seed=0, n_folds=5):
    rows = []
    for line in open(survey_path):
        s = json.loads(line)
        sp = s.get("stated_prefs") or {}
        text = (s.get(field) or "").strip()
        if not text or any(sp.get(a) is None for a in ATTRS):
            continue
        rows.append({"user_id": s["user_id"], "text": text, **{a: float(sp[a]) for a in ATTRS}})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    df["fold"] = rng.permutation(np.arange(len(df)) % n_folds)
    Y = df[ATTRS].to_numpy()
    rel = Y - Y.mean(1, keepdims=True)  # within-person relative priority (removes "rates everything high")
    for i, a in enumerate(ATTRS):
        df[f"rel_{a}"] = rel[:, i]
    return df


def load_model(name, device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16 if cuda else torch.float32, device_map="auto" if cuda else None)
    model.eval()
    return tok, model


def encode(tok, user_text):
    """Chat-formatted ids and the [start, end) span of the user's own words."""
    prefix_msgs = [{"role": "user", "content": FRAME_PREFIX}]
    full_msgs = [{"role": "user", "content": FRAME_PREFIX + user_text}]
    full = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=True)
    # locate the user's words by character offset, then map to tokens
    start_char = full.index(FRAME_PREFIX) + len(FRAME_PREFIX)
    end_char = start_char + len(user_text)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    span = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end_char and b > start_char]
    return enc["input_ids"], span[0], span[-1] + 1


@torch.no_grad()
def extract(tok, model, texts, batch_size=16, max_len=384):
    """Returns (mean over user-text tokens, last prompt token) per layer: [N, n_layers+1, d] each, float16."""
    items = [encode(tok, t) for t in texts]
    N = len(items)
    n_layers = model.config.num_hidden_layers + 1
    d = model.config.hidden_size
    mean_out = np.zeros((N, n_layers, d), dtype=np.float16)
    last_out = np.zeros((N, n_layers, d), dtype=np.float16)
    dev = next(model.parameters()).device
    order = np.argsort([len(x[0]) for x in items])
    for b0 in range(0, N, batch_size):
        idx = order[b0:b0 + batch_size]
        L = min(max(len(items[i][0]) for i in idx), max_len)
        ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        span = torch.zeros((len(idx), L), dtype=torch.float32)
        last = []
        for k, i in enumerate(idx):
            seq, s, e = items[i]
            seq = seq[:L]
            ids[k, :len(seq)] = torch.tensor(seq)
            att[k, :len(seq)] = 1
            span[k, s:min(e, L)] = 1
            last.append(len(seq) - 1)
        out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True)
        m = span.to(out.hidden_states[0].device)[:, :, None]
        denom = m.sum(1).clamp(min=1)
        last_t = torch.tensor(last, device=m.device)
        for li, h in enumerate(out.hidden_states):
            hf = h.float()
            mean_out[idx, li] = ((hf * m).sum(1) / denom).cpu().numpy().astype(np.float16)
            last_out[idx, li] = hf[torch.arange(len(idx), device=m.device), last_t].cpu().numpy().astype(np.float16)
        del out
    return mean_out, last_out
