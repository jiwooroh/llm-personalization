"""Study 4, step 3 (CPU): explicit vs implicit preference — knows vs says, near vs far.

Per item and condition, each of the 4 replies gets three scores; accuracy = the correct reply scores highest (chance 0.25).
  says_mc   mean over rotations of the letter log-prob the model gives that reply
  says_gen  log-prob lift of the reply given the context: LP[cond] - LP[none]  (none itself: raw LP, a text-only baseline)
  knows     linear probe on the reply's last-token hidden state, using H[cond] - H[none] (what the context changed in how
            the model represents the reply). PCA + logistic, 5-fold GroupKFold over personas, inner CV picks C.
            "transfer": probe trained on explicit items, tested on implicit items (same direction or not?).

Paired bootstrap over personas for every comparison.

python analyze.py --run ../../runs/implicit/qwen7b
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

C_GRID = [0.001, 0.01, 0.1, 1.0]


def mc_scores(MC):
    """MC (N, C, rot, letter) -> (N, C, option). In rotation r, letter j shows option (j + r) % 4."""
    N, C = MC.shape[:2]
    S = np.zeros((N, C, 4))
    for r in range(4):
        for j in range(4):
            S[:, :, (j + r) % 4] += MC[:, :, r, j] / 4
    return S


def acc(S):  # correct reply is option 0; ties (e.g. flat early-layer logits) count as wrong
    return (S[..., 0] > S[..., 1:].max(-1)).astype(float)


def fit_probe(Xtr, ytr, C):
    return LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr)


def probe_scores(X, groups, n_pca, train_mask=None, test_X=None):
    """X (N, 4, d). Returns out-of-fold scores (N, 4) [or scores for test_X when given]."""
    N, O, d = X.shape
    y = np.tile(np.r_[1, 0, 0, 0], N)
    out = np.zeros((N, O)) if test_X is None else np.zeros(test_X.shape[:2])
    for tr, te in GroupKFold(5).split(np.zeros(N), groups=groups):
        if train_mask is not None:
            tr = tr[train_mask[tr]]
        k = min(n_pca, len(tr) * O - 1, d)
        pca = PCA(k, random_state=0).fit(X[tr].reshape(-1, d))
        Z = lambda A: pca.transform(A.reshape(-1, d))
        Ztr = Z(X[tr]); mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
        Ztr = (Ztr - mu) / sd
        ytr = np.tile(np.r_[1, 0, 0, 0], len(tr))
        best, bC = -1, C_GRID[0]
        for C in C_GRID:   # inner split over personas, training folds only
            a_ = []
            for itr, iva in GroupKFold(3).split(np.zeros(len(tr)), groups=groups[tr]):
                m = fit_probe(Ztr.reshape(len(tr), O, -1)[itr].reshape(-1, k), np.tile(np.r_[1, 0, 0, 0], len(itr)), C)
                s = m.decision_function(Ztr.reshape(len(tr), O, -1)[iva].reshape(-1, k)).reshape(len(iva), O)
                a_.append((s.argmax(1) == 0).mean())
            if np.mean(a_) > best:
                best, bC = np.mean(a_), C
        m = fit_probe(Ztr, ytr, bC)
        src = X if test_X is None else test_X
        out[te] = m.decision_function((Z(src[te]) - mu) / sd).reshape(len(te), O)
    return out


def boot(a, b, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ids}
    d = np.empty(n)
    for i in range(n):
        idx = np.concatenate([by[g] for g in rng.choice(ids, len(ids))])
        d[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(delta=round(float(a.mean() - b.mean()), 4), ci=[round(float(lo), 4), round(float(hi), 4)],
                p=round(max(p, 1 / n), 4), sig=bool(lo > 0 or hi < 0))


def words(t):
    return set(re.findall(r"[a-z]{4,}", t.lower()))


def main(a):
    run = Path(a.run)
    meta = json.load(open(run / "meta.json"))
    items = [json.loads(x) for x in open(Path(a.items) if a.items else run.parent / "items.jsonl")]
    items = [it for it in items if it["item"] in set(meta["item_ids"])]
    conds = meta["conds"]
    ci = {c: k for k, c in enumerate(conds)}
    MC, LP, H = np.load(run / "MC.npy"), np.load(run / "LP.npy"), np.load(run / "H.npy", mmap_mode="r")
    groups = np.array([it["persona"] for it in items])
    N = len(items)

    S_mc = mc_scores(MC)
    res, correct = {}, {}
    for c in conds:
        k = ci[c]
        correct[("says_mc", c)] = acc(S_mc[:, k])
        correct[("says_gen", c)] = acc(LP[:, k] if c == "none" else LP[:, k] - LP[:, ci["none"]])

    # knows: per layer, on context-induced change of the reply representation
    layer_rows = []
    for li, L in enumerate(meta["layers"]):
        base = np.asarray(H[:, ci["none"], :, li], np.float32)
        for c in conds:
            X = np.asarray(H[:, ci[c], :, li], np.float32)
            X = X if c == "none" else X - base
            s = probe_scores(X, groups, a.pca)
            layer_rows.append({"cond": c, "layer": L, "acc": acc(s).mean()})
            correct[("knows", c, L)] = acc(s)
        print(f"layer {L:2d} " + " ".join(f"{c}={r['acc']:.3f}" for c, r in
                                          zip(conds, layer_rows[-len(conds):])), flush=True)
    lr = pd.DataFrame(layer_rows)
    # one shared layer for all comparisons: best layer for the explicit condition (pre-registered rule)
    L_ref = int(lr[lr.cond == "explicit"].sort_values("acc").iloc[-1].layer)
    li_ref = meta["layers"].index(L_ref)
    for c in conds:
        correct[("knows", c)] = correct[("knows", c, L_ref)]

    has = lambda *cs: all(c in ci for c in cs)
    # transfer: explicit-trained probe applied to implicit (and near->far)
    base = np.asarray(H[:, ci["none"], :, li_ref], np.float32)
    Xe = np.asarray(H[:, ci["explicit"], :, li_ref], np.float32) - base
    transfer_targets = [t for t in ["implicit", "explicit_far", "implicit_far", "implicit_attr", "implicit_recall"] if t in ci]
    for tgt in transfer_targets:
        Xt = np.asarray(H[:, ci[tgt], :, li_ref], np.float32) - base
        correct[("knows_from_explicit", tgt)] = acc(probe_scores(Xe, groups, a.pca, test_X=Xt))

    table = pd.DataFrame({m: {c: correct[(m, c)].mean() for c in conds} for m in ["says_mc", "says_gen", "knows"]}).round(4)
    tests = []
    pairs = [("explicit", "implicit"), ("explicit", "none"), ("implicit", "none"),
             ("explicit", "explicit_far"), ("implicit", "implicit_far"),
             ("implicit_attr", "implicit"), ("implicit_recall", "implicit")]      # R1 attribution, R2 retrieval
    for m in ["says_mc", "says_gen", "knows"]:
        for x, z in pairs:
            if has(x, z):
                tests.append({"measure": m, "a": x, "b": z, **boot(correct[(m, x)], correct[(m, z)], groups)})
    for c in [c for c in conds if c != "none"]:   # knows - says within condition
        tests.append({"measure": "knows-says_mc", "a": c, "b": c, **boot(correct[("knows", c)], correct[("says_mc", c)], groups)})
    gap = lambda c: correct[("knows", c)] - correct[("says_mc", c)]
    for x, z in [("implicit", "explicit"), ("implicit_far", "explicit_far")]:
        if has(x, z):
            tests.append({"measure": f"gap({x})-gap({z})", "a": x, "b": z, **boot(gap(x), gap(z), groups)})
    for tgt in transfer_targets:
        tests.append({"measure": "knows_transfer_vs_within", "a": f"explicit->{tgt}", "b": tgt,
                      **boot(correct[("knows_from_explicit", tgt)], correct[("knows", tgt)], groups)})

    # R3 logit lens: multiple-choice accuracy read from each layer; where does the personalised answer appear / vanish?
    lens = {}
    if (run / "MC_lens.npy").exists():
        MCL = np.load(run / "MC_lens.npy")
        for li, L in enumerate(meta["layers"]):
            for c in conds:
                lens.setdefault(c, {})[int(L)] = round(float(acc(mc_scores(MCL[:, :, :, li])[:, ci[c]]).mean()), 4)
    # R4 prior competition: does the no-context preference for a distractor predict failure under implicit context?
    prior = {}
    margin = S_mc[:, ci["none"], 0] - S_mc[:, ci["none"], 1:].max(1)      # >0: model already prefers the right reply
    strong_wrong = margin < np.quantile(margin, 1 / 3)
    for c in [c for c in conds if c != "none"]:
        prior[c] = {"says_strong_wrong_prior": round(float(correct[("says_mc", c)][strong_wrong].mean()), 4),
                    "says_other": round(float(correct[("says_mc", c)][~strong_wrong].mean()), 4),
                    "knows_strong_wrong_prior": round(float(correct[("knows", c)][strong_wrong].mean()), 4),
                    "knows_other": round(float(correct[("knows", c)][~strong_wrong].mean()), 4)}
    recall_hit = None
    if (run / "recall.jsonl").exists() and "implicit_recall" in ci:   # did the self-recall mention the preference?
        rec = {json.loads(x)["item"]: json.loads(x) for x in open(run / "recall.jsonl")}
        hit = np.array([len(words(rec[it["item"]]["preference"]) & words(rec[it["item"]]["recall"])) > 0 for it in items])
        recall_hit = {"share_recall_mentions_pref": round(float(hit.mean()), 4),
                      "says_recall_when_mentioned": round(float(correct[("says_mc", "implicit_recall")][hit].mean()), 4) if hit.any() else None,
                      "says_recall_when_not": round(float(correct[("says_mc", "implicit_recall")][~hit].mean()), 4) if (~hit).any() else None}

    # lexical overlap check: does the explicit advantage just come from word overlap with the correct reply?
    ov = np.array([len(words(it["options"][0]) & words(" ".join(t["content"] for t in it["contexts"]["implicit"] if t["role"] == "user")))
                   for it in items])
    hi = ov >= np.median(ov)
    overlap = {f"{m}_{c}_{'hi' if h else 'lo'}_overlap": round(float(correct[(m, c)][hi == h].mean()), 4)
               for m in ["says_mc", "knows"] for c in ["explicit", "implicit"] for h in [True, False]}
    by_type = {t: {m: {c: round(float(correct[(m, c)][[it["pref_type"] == t for it in items]].mean()), 4)
                       for c in ["explicit", "implicit"]} for m in ["says_mc", "knows"]}
               for t in sorted({it["pref_type"] for it in items})}

    # items the sentence-embedding baseline gets wrong (embed_baseline.py): what does the LLM add beyond similarity?
    hard = {}
    eb = Path(a.items).parent / "embed_baseline.json" if a.items else run.parent / "embed_baseline.json"
    if eb.exists():
        pi = json.load(open(eb))["per_item"]
        pos = {iid: k for k, iid in enumerate(pi["item"])}
        for c in [c for c in ["explicit", "implicit", "explicit_far", "implicit_far"] if c in ci]:
            wrong = np.array([pi[c][pos[it["item"]]] == 0 for it in items])
            hard[c] = {"n": int(wrong.sum()), "embed_acc_all": round(float(1 - wrong.mean()), 4),
                       **{m: round(float(correct[(m, c)][wrong].mean()), 4) if wrong.any() else None
                          for m in ["says_mc", "says_gen", "knows"]}}

    out = {"model": meta["model"], "n_items": N, "n_personas": int(len(np.unique(groups))), "ref_layer": L_ref,
           "accuracy": table.to_dict(), "transfer_from_explicit": {t: round(float(correct[("knows_from_explicit", t)].mean()), 4)
                                                                  for t in transfer_targets},
           "tests": tests, "logit_lens_says_mc": lens, "prior_competition": prior, "recall": recall_hit,
           "embedding_hard_items": hard, "overlap_split": overlap, "by_pref_type": by_type}
    json.dump(out, open(run / "summary.json", "w"), indent=1)
    lr.to_csv(run / "knows_by_layer.csv", index=False)
    pd.DataFrame(tests).to_csv(run / "tests.csv", index=False)
    print(f"\n=== {meta['model']}  {N} items, {out['n_personas']} personas, ref layer {L_ref} (chance 0.25) ===")
    print(table.to_string())
    print("transfer explicit->", out["transfer_from_explicit"])
    print("on items the embedding baseline gets wrong:", hard)
    print("logit lens (says_mc by layer):", lens)
    print("prior competition:", prior)
    print("recall:", recall_hit)
    for t in tests:
        print(f"{t['measure']:34s} {t['a']:>20s} - {t['b']:<13s} {t['delta']:+.4f} CI {t['ci']} p={t['p']}{'  *' if t['sig'] else ''}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="../../runs/implicit/qwen7b")
    p.add_argument("--items", default="")
    p.add_argument("--pca", type=int, default=64)
    main(p.parse_args())
