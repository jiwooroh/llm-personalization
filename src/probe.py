"""Step 2: does the model's internal reading of a person's free-text wish encode the importance ratings
the same person gave on the survey?

For each layer: ridge regression from hidden state -> 8 stated_prefs, 5-fold CV by user, Spearman per attribute.
Targets: absolute ratings and within-person relative priority (rating minus the person's mean).
Baselines: TF-IDF ridge, sentence-embedding ridge (if sentence-transformers is installed), shuffled-label control.
Also saves a probe fit on all users at the best layer, used by the minimal-pair test.

python src/probe.py --run runs/qwen7b
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV

from common import ATTRS

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
    users = pd.read_parquet(os.path.join(a.run, "users.parquet"))
    folds = users.fold.to_numpy()
    targets = {"abs": users[ATTRS].to_numpy(), "rel": users[[f"rel_{x}" for x in ATTRS]].to_numpy()}
    rng = np.random.default_rng(0)
    rows = []

    # --- text baselines
    tfidf = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True).fit_transform(users.text).toarray()
    feats = {"tfidf": tfidf}
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        feats["sbert"] = st.encode(users.text.tolist(), batch_size=64, show_progress_bar=False)
    except Exception as e:  # optional
        print("sentence-transformers baseline skipped:", e)
    for name, X in feats.items():
        for t, Y in targets.items():
            r = cv_spearman(X, Y, folds)
            rows.append({"source": name, "layer": -1, "pool": "-", "target": t, **dict(zip(ATTRS, r)), "mean": r.mean()})

    # --- LLM residual stream, every layer
    for pool in a.pools.split(","):
        H = np.load(os.path.join(a.run, f"H_{pool}.npy"), mmap_mode="r")
        for L in range(H.shape[1]):
            X = np.asarray(H[:, L], dtype=np.float32)
            for t, Y in targets.items():
                r = cv_spearman(X, Y, folds)
                rows.append({"source": "llm", "layer": L, "pool": pool, "target": t, **dict(zip(ATTRS, r)), "mean": r.mean()})
            print(pool, L, {t: round(rows[-2 + i]["mean"], 3) for i, t in enumerate(targets)}, flush=True)

    df = pd.DataFrame(rows)
    # shuffled-label control at the best layer
    best = df[(df.source == "llm") & (df.target == "abs")].sort_values("mean").iloc[-1]
    H = np.load(os.path.join(a.run, f"H_{best.pool}.npy"), mmap_mode="r")
    Xb = np.asarray(H[:, int(best.layer)], dtype=np.float32)
    for t, Y in targets.items():
        r = cv_spearman(Xb, Y[rng.permutation(len(Y))], folds)
        df.loc[len(df)] = {"source": "shuffled_control", "layer": int(best.layer), "pool": best.pool, "target": t,
                           **dict(zip(ATTRS, r)), "mean": r.mean()}
    df.to_csv(os.path.join(a.run, "probe_results.csv"), index=False)

    # probe fit on all users at the best layer (for the minimal-pair test)
    mu, sd = Xb.mean(0), Xb.std(0) + 1e-6
    r = RidgeCV(alphas=ALPHAS).fit((Xb - mu) / sd, targets["abs"])
    np.savez(os.path.join(a.run, "probe_best.npz"), coef=r.coef_, intercept=r.intercept_, mu=mu, sd=sd,
             layer=int(best.layer), pool=best.pool, alpha=r.alpha_)
    summary = {"best_layer": int(best.layer), "best_pool": best.pool, "n_users": len(users)}
    for src in df.source.unique():
        sub = df[df.source == src]
        for t in targets:
            s = sub[sub.target == t].sort_values("mean").iloc[-1]
            summary[f"{src}_{t}"] = {"layer": int(s.layer), **{k: round(float(s[k]), 3) for k in ATTRS + ["mean"]}}
    json.dump(summary, open(os.path.join(a.run, "probe_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/qwen7b")
    p.add_argument("--pools", default="mean,last")
    main(p.parse_args())
