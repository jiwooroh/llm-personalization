"""RQ1 + RQ2: does the residual stream encode what people reward, and how much of that is surface?

For every layer:
  1. pairwise preference probe on (h_win - h_lose), user-level split -> held-out accuracy
  2. preference direction (mean of h_win - h_lose) vs surface-feature directions (length, list format,
     headers/bold, refusal, hedging) and perceived-attribute directions (ridge on PRISM performance ratings)
     -> cosine similarities
  3. concept erasure: project out the surface subspace (and, stronger, everything linearly predictive of
     the surface features) and re-run the probe -> how much preference signal survives
Baselines: surface-features-only logistic regression; random-direction erasure of the same dimensionality.

Usage:
    python src/probes.py --acts data/acts/llama31_8b --out results/llama31_8b
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

SURF = ["log_words", "has_list", "has_header_or_bold", "has_refusal", "n_hedges", "n_paragraphs"]
ATTRS = ["values", "fluency", "factuality", "safety", "diversity", "creativity", "helpfulness"]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def logreg(Xtr, ytr, Xte, yte, wd=1e-3, steps=300):
    """L2 logistic regression (no bias: pairwise differences are sign-symmetric). Returns test acc, weight."""
    Xtr, Xte = torch.tensor(Xtr, device=dev, dtype=torch.float32), torch.tensor(Xte, device=dev, dtype=torch.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = Xtr / sd, Xte / sd  # scale only; centering would break sign symmetry
    ytr_t = torch.tensor(ytr, device=dev, dtype=torch.float32)
    w = torch.zeros(Xtr.shape[1], device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w, ytr_t) + wd * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    acc = ((Xte @ w > 0).float().cpu().numpy() == yte).mean()
    return float(acc), (w.detach() / sd).cpu().numpy()


def flip(D, rng):
    s = rng.choice([-1.0, 1.0], size=len(D))
    return D * s[:, None], (s > 0).astype(np.float32)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def orth_basis(V):
    q, _ = np.linalg.qr(np.stack(V, 1))
    return q  # d x k


def erase(X, Q):
    return X - (X @ Q) @ Q.T


def main(a):
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    resp = pd.read_parquet(a.responses)
    meta = json.load(open(os.path.join(a.acts, "meta.json")))
    A = np.load(os.path.join(a.acts, f"{a.pool}.npy"), mmap_mode="r")
    N = meta["n"]
    resp = resp.iloc[:N]
    pairs = pd.read_parquet(a.pairs)
    pairs = pairs[(pairs.win_row < N) & (pairs.lose_row < N) & (pairs.gap >= a.min_gap)]
    tr, te = pairs[pairs.split == "train"], pairs[pairs.split == a.eval_split]
    print(f"pairs train {len(tr):,} / {a.eval_split} {len(te):,}")

    # surface-only baseline (same pairwise logistic setup)
    F = resp[SURF].to_numpy(np.float32)
    Dtr, ytr = flip(F[tr.win_row] - F[tr.lose_row], rng)
    Dte, yte = flip(F[te.win_row] - F[te.lose_row], rng)
    surf_acc, _ = logreg(Dtr, ytr, Dte, yte)
    print(f"surface-only baseline acc: {surf_acc:.3f}")

    train_rows = resp.index[resp.split == "train"].to_numpy()
    layers = range(meta["n_layers"]) if not a.layers else [int(x) for x in a.layers.split(",")]
    rows = []
    for L in layers:
        H = np.asarray(A[:, L, :], dtype=np.float32)
        Htr = H[train_rows]
        # --- 1. preference probe
        Dtr, ytr = flip(H[tr.win_row] - H[tr.lose_row], rng)
        Dte, yte = flip(H[te.win_row] - H[te.lose_row], rng)
        acc, w = logreg(Dtr, ytr, Dte, yte)
        pref = unit((H[tr.win_row] - H[tr.lose_row]).mean(0))
        # --- 2. surface directions (diff-in-means on train responses)
        sdirs = {}
        f = resp.loc[train_rows]
        q_lo, q_hi = f.log_words.quantile([1 / 3, 2 / 3])
        sdirs["length"] = Htr[(f.log_words >= q_hi).values].mean(0) - Htr[(f.log_words <= q_lo).values].mean(0)
        for name, col in [("list", "has_list"), ("header_bold", "has_header_or_bold"), ("refusal", "has_refusal")]:
            m = (f[col] == 1).values
            if m.sum() > 20 and (~m).sum() > 20:
                sdirs[name] = Htr[m].mean(0) - Htr[~m].mean(0)
        m = (f.n_hedges > 0).values
        sdirs["hedging"] = Htr[m].mean(0) - Htr[~m].mean(0)
        # --- perceived-attribute directions: ridge on chosen responses' performance ratings
        adirs = {}
        for k in ATTRS:  # each attribute uses its own non-missing subset (N/A answers are null)
            ch = f[f.if_chosen.astype(bool) & f[f"perf_{k}"].notna()]
            if len(ch) <= 50:
                continue
            Hc = Htr[np.searchsorted(train_rows, ch.index.to_numpy())]
            Hc = Hc - Hc.mean(0)
            y = ch[f"perf_{k}"].to_numpy(np.float32)
            y = (y - y.mean()) / (y.std() + 1e-6)
            lam = a.ridge * np.trace(Hc @ Hc.T) / len(Hc)
            # dual ridge: w = Hc^T (Hc Hc^T + lam I)^-1 y
            adirs[k] = Hc.T @ np.linalg.solve(Hc @ Hc.T + lam * np.eye(len(Hc)), y)
        rec = {"layer": L, "probe_acc": acc,
               "cos_probe_pref": float(unit(w) @ pref)}
        for k, v in {**sdirs, **{f"attr_{k}": v for k, v in adirs.items()}}.items():
            rec[f"cos_pref_{k}"] = float(pref @ unit(v))
        # --- 3. erasure: surface diff-in-means subspace
        Q = orth_basis([unit(v) for v in sdirs.values()])
        acc_e1, _ = logreg(erase(Dtr, Q), ytr, erase(Dte, Q), yte)
        # random subspace of same size (control)
        R = np.linalg.qr(rng.standard_normal((H.shape[1], Q.shape[1])))[0]
        acc_rand, _ = logreg(erase(Dtr, R), ytr, erase(Dte, R), yte)
        # stronger: erase every direction linearly predictive of the surface features (ridge, k = #features)
        Hm = Htr - Htr.mean(0)
        Fs = f[SURF].to_numpy(np.float32)
        Fs = (Fs - Fs.mean(0)) / (Fs.std(0) + 1e-6)
        sub = rng.choice(len(Hm), size=min(len(Hm), a.ridge_n), replace=False)
        Hs = Hm[sub]
        lam = a.ridge * np.trace(Hs @ Hs.T) / len(Hs)
        Wf = Hs.T @ np.linalg.solve(Hs @ Hs.T + lam * np.eye(len(Hs)), Fs[sub])
        Q2 = orth_basis([unit(Wf[:, i]) for i in range(Wf.shape[1])] + [unit(v) for v in sdirs.values()])
        acc_e2, _ = logreg(erase(Dtr, Q2), ytr, erase(Dte, Q2), yte)
        rec.update({"acc_erase_surface_dim": acc_e1, "acc_erase_random_dim": acc_rand,
                    "acc_erase_surface_pred": acc_e2, "erase_k1": Q.shape[1], "erase_k2": Q2.shape[1]})
        rows.append(rec)
        np.savez(os.path.join(a.out, f"dirs_L{L:02d}.npz"), pref=pref, probe=w,
                 **{f"surf_{k}": v for k, v in sdirs.items()}, **{f"attr_{k}": v for k, v in adirs.items()})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "probe_results.csv"), index=False)
    json.dump({"surface_only_acc": surf_acc, "n_train_pairs": len(tr), "n_eval_pairs": len(te),
               "pool": a.pool, "acts": a.acts}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--acts", default="data/acts/llama31_8b")
    p.add_argument("--responses", default="data/processed/responses.parquet")
    p.add_argument("--pairs", default="data/processed/pairs.parquet")
    p.add_argument("--out", default="results/llama31_8b")
    p.add_argument("--pool", default="mean", choices=["mean", "last"])
    p.add_argument("--eval_split", default="dev")
    p.add_argument("--min_gap", type=int, default=10)
    p.add_argument("--layers", default="")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--ridge_n", type=int, default=4000)
    main(p.parse_args())
