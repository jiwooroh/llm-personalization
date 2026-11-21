"""Build the response table, surface features, within-prompt preference pairs and a user-level split
from the public PRISM release (utterances / conversations / survey jsonl).

Usage:
    python src/prepare_data.py --prism_dir data/prism --out_dir data/processed
"""
import argparse
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

ATTRS = ["values", "fluency", "factuality", "safety", "diversity", "creativity", "helpfulness"]

# --- surface features -------------------------------------------------------------------------
LIST_RE = re.compile(r"(^|\n)\s*([-*•]|\d+[.)])\s+\S")
HEADER_RE = re.compile(r"(^|\n)\s*#{1,6}\s+\S|\*\*[^*\n]+\*\*")
REFUSAL_RE = re.compile(
    r"\b(I'?m sorry|I am sorry|I apologi[sz]e|I cannot|I can't|I can not|I'm not able to|I am not able to|"
    r"I'm unable to|I am unable to|as an AI|As an AI|as a language model|As a language model|"
    r"I don't have (personal )?(opinions|beliefs|feelings)|I do not have (personal )?(opinions|beliefs|feelings))",
    re.IGNORECASE,
)
HEDGE_RE = re.compile(r"\b(it depends|however|on the other hand|some people|others (argue|believe)|"
                      r"it is important to (note|consider)|it's important to (note|consider))\b", re.IGNORECASE)


def surface_features(text: str) -> dict:
    words = text.split()
    return {
        "n_words": len(words),
        "log_words": float(np.log1p(len(words))),
        "n_paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
        "has_list": int(bool(LIST_RE.search(text))),
        "has_header_or_bold": int(bool(HEADER_RE.search(text))),
        "has_refusal": int(bool(REFUSAL_RE.search(text))),
        "n_hedges": len(HEDGE_RE.findall(text)),
        "ends_with_question": int(text.strip().endswith("?")),
    }


def user_split(user_id: str, seed: str = "prism-sr") -> str:
    h = int(hashlib.md5((seed + user_id).encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("dev" if h < 85 else "test")


def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(a.prism_dir, "utterances.jsonl"))]
    t0 = pd.DataFrame([r for r in rows if r["turn"] == 0])
    t0 = t0[t0["model_response"].str.strip().str.len() > 0].copy()
    feats = pd.DataFrame([surface_features(t) for t in t0["model_response"]], index=t0.index)
    t0 = pd.concat([t0, feats], axis=1)
    t0["split"] = t0["user_id"].map(user_split)

    # perceived attribute ratings (conversation level; attached to the chosen opening response)
    conv = {}
    for l in open(os.path.join(a.prism_dir, "conversations.jsonl")):
        c = json.loads(l)
        conv[c["conversation_id"]] = c
    for k in ATTRS:
        t0[f"perf_{k}"] = [
            (conv[cid]["performance_attributes"] or {}).get(k) if ch else None
            for cid, ch in zip(t0["conversation_id"], t0["if_chosen"])
        ]
    # stated priorities (user level)
    surv = {json.loads(l)["user_id"]: json.loads(l) for l in open(os.path.join(a.prism_dir, "survey.jsonl"))}
    for k in ATTRS:
        t0[f"stated_{k}"] = [(surv.get(u, {}).get("stated_prefs") or {}).get(k) for u in t0["user_id"]]
    t0["system_string"] = [surv.get(u, {}).get("system_string") for u in t0["user_id"]]

    keep = ["utterance_id", "interaction_id", "conversation_id", "user_id", "conversation_type", "user_prompt",
            "model_response", "model_name", "score", "if_chosen", "split"] + list(feats.columns) + \
           [f"perf_{k}" for k in ATTRS] + [f"stated_{k}" for k in ATTRS] + ["system_string"]
    resp = t0[keep].reset_index(drop=True)
    resp["row"] = np.arange(len(resp))
    resp.to_parquet(os.path.join(a.out_dir, "responses.parquet"))

    # within-prompt pairs (same interaction = same opening prompt, same user)
    pairs = []
    for _, g in resp.groupby("interaction_id"):
        g = g.sort_values("row")
        r = g.to_dict("records")
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[i]["score"] - r[j]["score"]
                if abs(d) < a.min_gap:
                    continue
                w, l = (r[i], r[j]) if d > 0 else (r[j], r[i])
                pairs.append({"interaction_id": w["interaction_id"], "user_id": w["user_id"], "split": w["split"],
                              "win_row": w["row"], "lose_row": l["row"], "gap": abs(d)})
    pairs = pd.DataFrame(pairs)
    pairs.to_parquet(os.path.join(a.out_dir, "pairs.parquet"))

    print(f"responses: {len(resp):,}  users: {resp.user_id.nunique():,}  interactions: {resp.interaction_id.nunique():,}")
    print(f"pairs (gap >= {a.min_gap}): {len(pairs):,}")
    print(pairs.groupby("split").size().to_string())
    # descriptive check: how often the surface-feature-larger response wins
    R = resp.set_index("row")
    for f in ["log_words", "has_list", "has_header_or_bold", "has_refusal", "n_hedges"]:
        dw = R.loc[pairs.win_row, f].values - R.loc[pairs.lose_row, f].values
        m = dw != 0
        print(f"  {f:20s} winner has more: {np.mean(dw[m] > 0):.3f}  (n={m.sum():,})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prism_dir", default="data/prism")
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--min_gap", type=int, default=10)
    main(p.parse_args())
