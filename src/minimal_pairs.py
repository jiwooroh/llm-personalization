"""Day 2: does the internal reading follow the LINGUISTIC FORM of a stated preference?
Known-answer perturbations (no human labels needed):

 A. Template ladder: "<attribute phrase>" stated at 6 strength levels, including negation
    (e.g. "It is essential that ..." > "I'd like ..." > "It would be nice if ..." > "I don't really care whether ..."
     > "It does not need to ..." > "I do not want ..."). Expected order is known.
 B. In-context edit: a real PRISM statement + one appended sentence that raises or lowers ONE attribute.
    Expected: that attribute's probe score moves in the stated direction; the other 7 move less.

Measured with the probe saved by probe.py (internal reading).

python src/minimal_pairs.py --model Qwen/Qwen2.5-7B-Instruct --run runs/qwen7b
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import ATTRS, extract, load_model

PHRASE = {  # (verb-phrase form used after "the AI ...", noun form)
    "values": ("reflects my values and cultural perspective", "respect for my values"),
    "fluency": ("writes clearly and coherently", "clear, well-written answers"),
    "factuality": ("gives accurate, factual information", "factual accuracy"),
    "safety": ("avoids anything harmful or dangerous", "safety"),
    "diversity": ("presents multiple viewpoints", "multiple viewpoints"),
    "creativity": ("is creative and imaginative", "creativity"),
    "helpfulness": ("is genuinely helpful and relevant to what I ask", "helpfulness"),
    "personalisation": ("adapts its answers to me personally", "personalised answers"),
}
LADDER = [  # level, template ; higher level = more important
    (5, "It is absolutely essential to me that the AI {vp}."),
    (4, "I really want the AI to be one that {vp}."),
    (3, "It would be nice if the AI {vp}."),
    (2, "I don't really care whether the AI {vp}."),
    (1, "The AI does not need to be one that {vp}."),
    (0, "I do not want an AI that {vp}."),
]
UP = "Above all, {np} matters most to me."
DOWN = "Honestly, {np} does not matter to me at all."


def load_probe(run):
    z = np.load(os.path.join(run, "probe_best.npz"), allow_pickle=True)
    return z, int(z["layer"]), str(z["pool"])


def probe_scores(H, z):
    return ((H - z["mu"]) / z["sd"]) @ z["coef"].T + z["intercept"]


def main(a):
    z, layer, pool = load_probe(a.run)
    users = pd.read_parquet(os.path.join(a.run, "users.parquet"))
    tok, model = load_model(a.model)
    pi = 0 if pool == "mean" else 1
    out = {"layer": layer, "pool": pool}

    # --- A. ladder
    rows = [{"attr": k, "level": lv, "text": t.format(vp=PHRASE[k][0])} for k in ATTRS for lv, t in LADDER]
    A = pd.DataFrame(rows)
    H = extract(tok, model, A.text.tolist(), batch_size=a.batch_size)[pi][:, layer].astype(np.float32)
    S = probe_scores(H, z)
    A["target_score"] = [S[i, ATTRS.index(k)] for i, k in enumerate(A.attr)]
    lad = {}
    for k in ATTRS:
        sub = A[A.attr == k]
        lad[k] = {"spearman_level": round(float(spearmanr(sub.level, sub.target_score)[0]), 3),
                  "neg_minus_pos": round(float(sub[sub.level == 0].target_score.mean() - sub[sub.level == 5].target_score.mean()), 2)}
    out["ladder"] = lad
    out["ladder_mean_spearman"] = round(float(np.mean([v["spearman_level"] for v in lad.values()])), 3)
    A.to_csv(os.path.join(a.run, "minimal_ladder.csv"), index=False)

    # --- B. in-context edits on real statements
    rng = np.random.default_rng(0)
    base = users.sample(n=min(a.n_context, len(users)), random_state=0)
    recs = [{"user_id": u, "attr": None, "dir": 0, "text": t} for u, t in zip(base.user_id, base.text)]
    for u, t in zip(base.user_id, base.text):
        for k in ATTRS:
            recs.append({"user_id": u, "attr": k, "dir": +1, "text": t.rstrip() + " " + UP.format(np=PHRASE[k][1])})
            recs.append({"user_id": u, "attr": k, "dir": -1, "text": t.rstrip() + " " + DOWN.format(np=PHRASE[k][1])})
    B = pd.DataFrame(recs)
    H = extract(tok, model, B.text.tolist(), batch_size=a.batch_size)[pi][:, layer].astype(np.float32)
    S = probe_scores(H, z)
    base_score = {u: S[i] for i, (u, d) in enumerate(zip(B.user_id, B.dir)) if d == 0}
    res = []
    for i, r in B.iterrows():
        if r.dir == 0:
            continue
        delta = S[i] - base_score[r.user_id]
        j = ATTRS.index(r.attr)
        res.append({"attr": r.attr, "dir": r.dir, "target_delta": delta[j],
                    "offtarget_abs_delta": np.abs(np.delete(delta, j)).mean(),
                    "correct_direction": float(np.sign(delta[j]) == r.dir)})
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(a.run, "minimal_context.csv"), index=False)
    ctx = R.groupby(["attr", "dir"]).agg(correct=("correct_direction", "mean"), target=("target_delta", "mean"),
                                         offtarget=("offtarget_abs_delta", "mean")).round(3)
    out["context"] = {f"{k}_{'up' if d > 0 else 'down'}": v for (k, d), v in ctx.to_dict("index").items()}
    out["context_correct_up"] = round(float(R[R.dir > 0].correct_direction.mean()), 3)
    out["context_correct_down"] = round(float(R[R.dir < 0].correct_direction.mean()), 3)
    json.dump(out, open(os.path.join(a.run, "minimal_summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("context",)}, indent=1, default=float))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--run", default="runs/qwen7b")
    p.add_argument("--n_context", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    main(p.parse_args())
