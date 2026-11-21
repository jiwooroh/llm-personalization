"""Fair 'knows vs says' comparison (no GPU).
The probe is a supervised ridge on hidden states; raw verbal ratings are zero-shot.
Here the 8 verbal ratings get the SAME treatment as the probe: ridge, same 5 user folds, Spearman per attribute.
If the probe still beats calibrated verbal, the model holds information it does not put into words.

python src/verbal_calibrated.py --run runs/qwen7b
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

# same as common.ATTRS / probe.cv_spearman, copied so this runs without torch
ATTRS = ["values", "fluency", "factuality", "safety", "diversity", "creativity", "helpfulness", "personalisation"]
ALPHAS = np.logspace(-1, 5, 13)


def cv_spearman(X, Y, folds):
    P = np.zeros_like(Y)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        r = RidgeCV(alphas=ALPHAS).fit((X[tr] - mu) / sd, Y[tr])
        P[te] = r.predict((X[te] - mu) / sd)
    return np.array([spearmanr(P[:, j], Y[:, j])[0] for j in range(Y.shape[1])])


def main(a):
    u = pd.read_parquet(os.path.join(a.run, "users.parquet"))
    v = pd.read_parquet(os.path.join(a.run, "verbal.parquet"))
    assert (u.user_id.values == v.user_id.values).all()
    V = v[[f"verbal_{k}" for k in ATTRS]].to_numpy(float)
    Vr = V - V.mean(1, keepdims=True)
    folds = u.fold.to_numpy()
    Y = {"abs": u[ATTRS].to_numpy(float), "rel": u[[f"rel_{k}" for k in ATTRS]].to_numpy(float)}
    probe = json.load(open(os.path.join(a.run, "probe_summary.json")))
    raw = json.load(open(os.path.join(a.run, "verbal_summary.json")))
    out, rows = {}, []
    for t in ["abs", "rel"]:
        r = cv_spearman(np.hstack([V, Vr]), Y[t], folds)
        out[f"verbal_calibrated_{t}"] = {**dict(zip(ATTRS, np.round(r, 3))), "mean": round(float(r.mean()), 3)}
        for name, d in [("TF-IDF", probe[f"tfidf_{t}"]), ("sentence emb.", probe[f"sbert_{t}"]),
                        ("verbal raw", raw[t]), ("verbal calibrated", out[f"verbal_calibrated_{t}"]),
                        ("LLM probe", probe[f"llm_{t}"])]:
            rows.append({"target": t, "method": name, **{k: d[k] for k in ATTRS}, "mean": d["mean"]})
    json.dump(out, open(os.path.join(a.run, "verbal_calibrated_summary.json"), "w"), indent=1, default=float)
    df = pd.DataFrame(rows).round(3)
    df.to_csv(os.path.join(a.run, "knows_vs_says.csv"), index=False)
    print(df[["target", "method", "mean"] + ATTRS].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/qwen7b")
    main(p.parse_args())
