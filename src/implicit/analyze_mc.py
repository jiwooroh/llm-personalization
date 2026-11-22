"""Study 4 (final scope): which preferences does the LLM pick up when they are only implied, and is it more than surface similarity?

Multiple-choice accuracy only (extract.py --mc_only). Three results:
  1. explicit vs implicit gap (and near vs far)
  2. LLM vs sentence-embedding baseline, overall and on items the embedding gets wrong
  3. which implicit preferences are lost: by distance, preference type, lexical overlap
All differences: persona-level paired bootstrap.

python analyze_mc.py --run ../../runs/implicit/qwen7b
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import acc, boot, mc_scores, words


def main(a):
    run = Path(a.run)
    meta = json.load(open(run / "meta.json"))
    items_path = Path(a.items) if a.items else run.parent / "items.jsonl"
    by_id = {it["item"]: it for it in map(json.loads, open(items_path))}
    items = [by_id[i] for i in meta["item_ids"]]
    ci = {c: k for k, c in enumerate(meta["conds"])}
    A = acc(mc_scores(np.load(run / "MC.npy")))                     # (N, C) 1 = correct
    g = np.array([it["persona"] for it in items])
    col = lambda c: A[:, ci[c]]
    out = {"model": meta["model"], "n_items": len(items), "n_personas": int(len(set(g))),
           "accuracy": {c: round(float(col(c).mean()), 4) for c in ci}}

    # 1. gaps
    pairs = [("explicit", "implicit"), ("explicit", "none"), ("implicit", "none"),
             ("explicit", "explicit_far"), ("implicit", "implicit_far")]
    out["gaps"] = {f"{x}-{y}": boot(col(x), col(y), g) for x, y in pairs if x in ci and y in ci}

    # 2. vs embedding baseline
    eb = json.load(open(items_path.parent / "embed_baseline.json"))["per_item"]
    pos = {iid: k for k, iid in enumerate(eb["item"])}
    out["vs_embedding"] = {}
    for c in ["explicit", "implicit", "explicit_far", "implicit_far"]:
        if c not in ci:
            continue
        e = np.array([eb[c][pos[it["item"]]] for it in items], float)
        hard = e == 0
        out["vs_embedding"][c] = {"llm": round(float(col(c).mean()), 4), "embed": round(float(e.mean()), 4),
                                  "llm_minus_embed": boot(col(c), e, g),
                                  "n_embed_wrong": int(hard.sum()),
                                  "llm_on_embed_wrong": round(float(col(c)[hard].mean()), 4),
                                  "llm_on_embed_right": round(float(col(c)[~hard].mean()), 4)}

    # 3. which implicit preferences are lost
    ov = np.array([len(words(it["options"][0]) & words(" ".join(t["content"] for t in it["contexts"]["implicit"]
                                                                  if t["role"] == "user"))) for it in items])
    hi = ov >= np.median(ov)
    out["overlap_split"] = {c: {"hi": round(float(col(c)[hi].mean()), 4), "lo": round(float(col(c)[~hi].mean()), 4)}
                            for c in ["explicit", "implicit"]}
    types = np.array([it["pref_type"] for it in items])
    out["by_pref_type"] = {t: {c: round(float(col(c)[types == t].mean()), 4) for c in ["none", "explicit", "implicit"]}
                           | {"n": int((types == t).sum())} for t in sorted(set(types))}

    json.dump(out, open(run / "results_mc.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--items", default="")
    main(p.parse_args())
