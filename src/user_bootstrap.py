"""User-level paired bootstrap for Study 1.

Needs only small files: users.parquet, probe_oof.npz (written by probe.py) and, if present, verbal.parquet.
Each bootstrap draw resamples the 1,500 users with replacement, recomputes every method's Spearman rho
per attribute on the same draw, averages over the 8 attributes, and records the differences.

python src/user_bootstrap.py --run runs/qwen7b
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV

from common import ATTRS

ALPHAS = np.logspace(-1, 5, 13)
NAMES = {"llm": "LLM probe", "sbert": "sentence emb.", "tfidf": "TF-IDF",
         "verbal_raw": "verbal raw", "verbal_cal": "verbal calibrated"}


def cv_predict(X, Y, folds):
    P = np.zeros_like(Y, dtype=float)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        P[te] = RidgeCV(alphas=ALPHAS).fit((X[tr] - mu) / sd, Y[tr]).predict((X[te] - mu) / sd)
    return P


def boot_rho(P, Y, idx):
    """Mean-over-attributes Spearman rho for every bootstrap draw. idx: (B, N)."""
    out = np.zeros((idx.shape[0], P.shape[1]))
    for j in range(P.shape[1]):
        rp = rankdata(P[idx, j], axis=1)
        ry = rankdata(Y[idx, j], axis=1)
        rp -= rp.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
        den = np.sqrt((rp ** 2).sum(1) * (ry ** 2).sum(1))
        out[:, j] = np.where(den > 0, (rp * ry).sum(1) / np.maximum(den, 1e-12), 0.0)
    return out.mean(1), out


def main(a):
    u = pd.read_parquet(os.path.join(a.run, "users.parquet"))
    z = np.load(os.path.join(a.run, "probe_oof.npz"), allow_pickle=True)
    assert (z["user_id"] == u.user_id.to_numpy()).all(), "probe_oof.npz rows do not match users.parquet"
    folds = u.fold.to_numpy()
    Y = {"abs": u[ATTRS].to_numpy(float), "rel": u[[f"rel_{k}" for k in ATTRS]].to_numpy(float)}

    preds = {t: {} for t in Y}
    for t in Y:
        for m in ["llm", "sbert", "tfidf"]:
            if f"{m}_{t}" in z.files:
                preds[t][m] = z[f"{m}_{t}"].astype(float)
    vpath = os.path.join(a.run, "verbal.parquet")
    if os.path.exists(vpath):
        v = pd.read_parquet(vpath)
        assert (v.user_id.values == u.user_id.values).all()
        V = v[[f"verbal_{k}" for k in ATTRS]].to_numpy(float)
        Vr = V - V.mean(1, keepdims=True)
        for t in Y:
            preds[t]["verbal_raw"] = V if t == "abs" else Vr
            preds[t]["verbal_cal"] = cv_predict(np.hstack([V, Vr]), Y[t], folds)

    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, len(u), size=(a.n_boot, len(u)))
    full = np.arange(len(u))[None, :]
    result = {"run": a.run, "n_users": len(u), "n_boot": a.n_boot, "layer": int(z["layer"]),
              "pool": str(z["pool"]), "rho": {}, "pairs": []}
    for t in Y:
        draws, point = {}, {}
        for m, P in preds[t].items():
            draws[m], _ = boot_rho(P, Y[t], idx)
            point[m] = float(boot_rho(P, Y[t], full)[0][0])
            lo, hi = np.percentile(draws[m], [2.5, 97.5])
            result["rho"][f"{m}_{t}"] = dict(rho=round(point[m], 4), ci=[round(float(lo), 4), round(float(hi), 4)])
        for m1, m2 in itertools.combinations(preds[t], 2):
            d = draws[m1] - draws[m2]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            result["pairs"].append(dict(target=t, a=NAMES[m1], b=NAMES[m2],
                                        delta=round(point[m1] - point[m2], 4),
                                        ci=[round(float(lo), 4), round(float(hi), 4)], p=round(max(p, 1 / a.n_boot), 4),
                                        sig=bool(lo > 0 or hi < 0)))
    json.dump(result, open(os.path.join(a.run, "user_bootstrap.json"), "w"), indent=1)
    pd.DataFrame(result["pairs"]).to_csv(os.path.join(a.run, "user_bootstrap.csv"), index=False)
    for k, v in result["rho"].items():
        print(f"{k:16s} rho={v['rho']:.3f}  CI {v['ci']}")
    for r in result["pairs"]:
        print(f"[{r['target']}] {r['a']:18s} - {r['b']:18s} {r['delta']:+.3f}  CI {r['ci']}  p={r['p']}"
              f"{'  *' if r['sig'] else ''}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/qwen7b")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
