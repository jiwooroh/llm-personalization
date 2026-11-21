"""Study 3, step 3: does the model's reading of a reader's stated taste predict that reader's real A/B choices?

Inputs (from extract_joint.py): judgments.csv, rows_<cond>.csv, H_<cond>.npy (rows, 2 orders, layers, d),
LP_<cond>.npy (rows, 2 orders) for conditions none / own / other / own_pos.

Position bias cancels by construction: order 0 shows text1 as A, order 1 shows text1 as B.
  knows  feature = h(order 0) - h(order 1); per-layer logistic on PCA of that difference,
         5-fold GroupKFold over pairs (same split as stated_vs_revealed.py, so numbers line up with Study 2)
  says   score   = [logit(A)-logit(B)](order 0) - [same](order 1); > 0 means "text1". No training.

Key test: own - other (did the reader's own profile help, beyond having *a* profile?).
Paired bootstrap over readers (each draw resamples readers, keeps all their judgments).

python probe_joint.py --run ../../runs/literary_taste/joint_qwen7b
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def features(run, cond, J):
    """Map a condition's rows onto the judgment table. Returns D (N, layers, d) and says-score (N,)."""
    rows = pd.read_csv(run / f"rows_{cond}.csv")
    H = np.load(run / f"H_{cond}.npy", mmap_mode="r")
    LP = np.load(run / f"LP_{cond}.npy")
    key = ["iid"] if cond == "none" else ["user", "iid"]
    pos = rows.reset_index().merge(J[key], on=key, how="right")["index"].to_numpy()
    assert not np.isnan(pos.astype(float)).any(), f"{cond}: some judgments have no row"
    D = np.asarray(H[pos, 0], dtype=np.float32) - np.asarray(H[pos, 1], dtype=np.float32)
    return D, LP[pos, 0] - LP[pos, 1]


def fit(X, y, C):
    # symmetric augmentation: swapping the texts flips the label, so no intercept is needed
    return LogisticRegression(C=C, max_iter=3000, fit_intercept=False).fit(np.r_[X, -X], np.r_[y, 1 - y])


def probe_layer(D, y, groups, n_pca):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(D, y, groups):
        k = min(n_pca, len(tr) - 1, D.shape[1])
        pca = PCA(k, random_state=0).fit(np.r_[D[tr], -D[tr]])
        Z = pca.transform(D); Z /= Z[tr].std(0) + 1e-9
        best, bC = -1, C_GRID[0]
        for C in C_GRID:   # inner CV over pairs, training folds only
            acc = [(fit(Z[tr][a], y[tr][a], C).predict(Z[tr][b]) == y[tr][b]).mean()
                   for a, b in GroupKFold(3).split(Z[tr], y[tr], groups[tr])]
            if np.mean(acc) > best:
                best, bC = np.mean(acc), C
        pred[te] = fit(Z[tr], y[tr], bC).predict(Z[te])
    return pred


def user_boot(correct_a, correct_b, users, n_boot, rng):
    """Paired bootstrap over readers of mean accuracy difference a - b."""
    ids = np.unique(users)
    by = {u: np.where(users == u)[0] for u in ids}
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([by[u] for u in rng.choice(ids, len(ids))])
        d[i] = correct_a[idx].mean() - correct_b[idx].mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(delta=round(float(correct_a.mean() - correct_b.mean()), 4), ci=[round(float(lo), 4), round(float(hi), 4)],
                p=round(max(p, 1 / n_boot), 4), sig=bool(lo > 0 or hi < 0))


def main(a):
    run = Path(a.run)
    meta = json.load(open(run / "meta.json"))
    J = pd.read_csv(run / "judgments.csv")
    y, groups, users = J.y.to_numpy(), J.iid.to_numpy(), J.user.to_numpy()
    conds = [c for c in ["none", "own", "other", "own_pos"] if (run / f"H_{c}.npy").exists()]
    rng = np.random.default_rng(a.seed)

    knows, says, layer_rows, per_layer = {}, {}, [], {}
    for c in conds:
        D, S = features(run, c, J)
        says[c] = ((S > 0).astype(int) == y).astype(float)
        best = None
        for k, L in enumerate(meta["layers"]):
            corr = (probe_layer(D[:, k], y, groups, a.pca) == y).astype(float)
            per_layer[(c, L)] = corr
            layer_rows.append({"cond": c, "layer": L, "acc": corr.mean()})
            print(f"[{c}] layer {L:2d}  knows acc={corr.mean():.3f}", flush=True)
            if best is None or corr.mean() > best[1].mean():
                best = (L, corr)
        knows[c] = best
        print(f"[{c}] says acc={says[c].mean():.3f}   best knows layer={best[0]} acc={best[1].mean():.3f}")

    # compare conditions at one shared layer (best layer for "own") so the comparison is not cherry-picked
    L_ref = knows["own"][0] if "own" in knows else meta["layers"][len(meta["layers"]) // 2]
    lr = pd.DataFrame(layer_rows)
    at_ref = {c: per_layer[(c, L_ref)] for c in conds}

    tests = []
    for x, z in [("own", "other"), ("own", "none"), ("other", "none"), ("own", "own_pos")]:
        if x in conds and z in conds:
            tests.append({"what": "knows", "a": x, "b": z, **user_boot(at_ref[x], at_ref[z], users, a.n_boot, rng)})
            tests.append({"what": "says", "a": x, "b": z, **user_boot(says[x], says[z], users, a.n_boot, rng)})
    for c in conds:  # knows vs says within a condition
        tests.append({"what": "knows-says", "a": c, "b": c, **user_boot(at_ref[c], says[c], users, a.n_boot, rng)})

    study2 = None
    s2 = run.parent / "svr_minilm_len" / "summary.json"
    if s2.exists():
        s = json.load(open(s2))
        study2 = {"population": s["acc"]["population"], "+revealed": s["acc"]["+revealed"],
                  "revealed_gain": s["vs_population"]["+revealed"]["delta"]}

    out = {"model": meta["model"], "n_judgments": len(J), "n_users": int(len(np.unique(users))),
           "ref_layer": L_ref,
           "acc_knows_at_ref": {c: round(float(v.mean()), 4) for c, v in at_ref.items()},
           "acc_knows_best": {c: {"layer": v[0], "acc": round(float(v[1].mean()), 4)} for c, v in knows.items()},
           "acc_says": {c: round(float(v.mean()), 4) for c, v in says.items()},
           "tests": tests, "study2_reference": study2}
    json.dump(out, open(run / "joint_summary.json", "w"), indent=1)
    lr.to_csv(run / "joint_layers.csv", index=False)
    pd.DataFrame(tests).to_csv(run / "joint_tests.csv", index=False)

    print(f"\n=== {meta['model']}  n={len(J)} judgments, {out['n_users']} readers, ref layer {L_ref} ===")
    print(f"{'cond':8s} {'knows':>7s} {'says':>7s}")
    for c in conds:
        print(f"{c:8s} {at_ref[c].mean():7.3f} {says[c].mean():7.3f}")
    for t in tests:
        print(f"{t['what']:10s} {t['a']:>7s} - {t['b']:<7s} {t['delta']:+.4f}  CI {t['ci']}  p={t['p']}"
              f"{'  *' if t['sig'] else ''}")
    if study2:
        print(f"Study 2 reference: population {study2['population']:.3f}, +revealed {study2['+revealed']:.3f} "
              f"(gain {study2['revealed_gain']:+.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="../../runs/literary_taste/joint_qwen7b")
    p.add_argument("--pca", type=int, default=32)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
