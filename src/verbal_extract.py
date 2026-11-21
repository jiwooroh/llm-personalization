"""Step 3: what the model SAYS. Ask the same model to read each person's statement and rate how important
each of the 8 attributes is to that person (0-100, JSON). Compare with (a) the person's own survey ratings and
(b) what the probe reads from the hidden state (step 2). Gap between (a)-(b) agreement = 'knows vs says'.

python src/verbal_extract.py --model Qwen/Qwen2.5-7B-Instruct --run runs/qwen7b
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from common import ATTRS, load_model

DEFS = {
    "values": "reflects the person's values or cultural perspectives",
    "fluency": "well-written and coherent",
    "factuality": "factual and informative",
    "safety": "safe, does not risk harm",
    "diversity": "summarises multiple viewpoints or worldviews",
    "creativity": "creative and inspiring",
    "helpfulness": "helpful and relevant to their requests",
    "personalisation": "tailored to the person specifically",
}

PROMPT = (
    "A person wrote the following about what they want from an AI assistant:\n\n\"{text}\"\n\n"
    "Based only on what they wrote, estimate how important each property is to this person, "
    "from 0 (not at all important) to 100 (extremely important). Properties:\n{defs}\n\n"
    "Answer with JSON only, using exactly these keys: {keys}"
)


def parse(s):
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return [float(d[k]) for k in ATTRS]
    except Exception:
        return None


@torch.no_grad()
def main(a):
    users = pd.read_parquet(os.path.join(a.run, "users.parquet"))
    tok, model = load_model(a.model)
    tok.padding_side = "left"
    defs = "\n".join(f"- {k}: {v}" for k, v in DEFS.items())
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(text=t, defs=defs, keys=", ".join(ATTRS))}],
                                       tokenize=False, add_generation_prompt=True) for t in users.text]
    dev = next(model.parameters()).device
    outs = []
    for i in range(0, len(prompts), a.batch_size):
        b = tok(prompts[i:i + a.batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        g = model.generate(**b, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, b["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"{i + len(prompts[i:i + a.batch_size])}/{len(prompts)}", flush=True)
    parsed = [parse(o) for o in outs]
    ok = np.array([p is not None for p in parsed])
    V = np.array([p if p is not None else [np.nan] * len(ATTRS) for p in parsed])
    pd.DataFrame(V, columns=[f"verbal_{k}" for k in ATTRS]).assign(user_id=users.user_id.values, raw=outs) \
        .to_parquet(os.path.join(a.run, "verbal.parquet"))
    Y = users[ATTRS].to_numpy()
    Vr, Yr = V - np.nanmean(V, 1, keepdims=True), Y - Y.mean(1, keepdims=True)
    res = {"parse_rate": float(ok.mean())}
    for name, (P, T) in {"abs": (V, Y), "rel": (Vr, Yr)}.items():
        r = [spearmanr(P[ok, j], T[ok, j])[0] for j in range(len(ATTRS))]
        res[name] = {**dict(zip(ATTRS, np.round(r, 3))), "mean": round(float(np.mean(r)), 3)}
    json.dump(res, open(os.path.join(a.run, "verbal_summary.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--run", default="runs/qwen7b")
    p.add_argument("--batch_size", type=int, default=32)
    main(p.parse_args())
