"""Study 4 baseline (CPU, no LLM): pick the reply most similar to the preceding user turns (sentence embeddings).

If this already solves the explicit condition but not the implicit one, explicit vs implicit is partly a surface-similarity
effect; the LLM's knows/says should be compared against it.

python embed_baseline.py --items ../../runs/implicit/items.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np

CONDS = ["explicit", "implicit", "explicit_far", "implicit_far"]


def main(a):
    from sentence_transformers import SentenceTransformer
    items = [json.loads(x) for x in open(a.items)]
    st = SentenceTransformer(a.model)
    opts = st.encode([o for it in items for o in it["options"]], batch_size=64, normalize_embeddings=True)
    opts = opts.reshape(len(items), 4, -1)
    q = st.encode([it["query"] for it in items], batch_size=64, normalize_embeddings=True)
    res = {"query_only": float(((opts @ q[:, :, None])[..., 0].argmax(1) == 0).mean())}
    per_item = {"item": [it["item"] for it in items]}
    for c in CONDS:
        ctx = [" ".join(t["content"] for t in it["contexts"][c] if t["role"] == "user") for it in items]
        e = st.encode(ctx, batch_size=32, normalize_embeddings=True)
        s = (opts @ e[:, :, None])[..., 0]
        res[c] = float((s.argmax(1) == 0).mean())
        per_item[c] = (s.argmax(1) == 0).astype(int).tolist()
    res = {k: round(v, 4) for k, v in res.items()}
    out = Path(a.items).parent / "embed_baseline.json"
    json.dump({"model": a.model, "n_items": len(items), "accuracy": res, "chance": 0.25, "per_item": per_item},
              open(out, "w"), indent=1)
    print(res, "->", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--items", default="../../runs/implicit/items.jsonl")
    p.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    main(p.parse_args())
