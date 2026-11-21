"""Step 1: read every PRISM user's free-text statement of what they want from an AI (system_string)
and save the model's residual stream (mean over the user's words, and the last prompt token) at every layer.

python src/extract_acts.py --model Qwen/Qwen2.5-7B-Instruct --out runs/qwen7b
"""
import argparse
import json
import os

import numpy as np

from common import download_survey, extract, load_model, load_users


def main(a):
    os.makedirs(a.out, exist_ok=True)
    users = load_users(download_survey(a.survey), field=a.field)
    if a.limit:
        users = users.iloc[:a.limit].copy()
    users.to_parquet(os.path.join(a.out, "users.parquet"))
    tok, model = load_model(a.model)
    H_mean, H_last = extract(tok, model, users.text.tolist(), batch_size=a.batch_size)
    np.save(os.path.join(a.out, "H_mean.npy"), H_mean)
    np.save(os.path.join(a.out, "H_last.npy"), H_last)
    json.dump({"model": a.model, "field": a.field, "n": len(users), "n_layers": H_mean.shape[1],
               "d": H_mean.shape[2]}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"saved {len(users)} users x {H_mean.shape[1]} layers -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--survey", default="data/survey.jsonl")
    p.add_argument("--field", default="system_string", choices=["system_string", "self_description"])
    p.add_argument("--out", default="runs/qwen7b")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
