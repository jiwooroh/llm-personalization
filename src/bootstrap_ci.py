"""Bootstrap confidence intervals (CPU-only, no torch needed).

Three tests:
  1. probe vs verbal-calibrated: paired bootstrap on per-user Spearman ρ difference
     — uses users.parquet + verbal.parquet + probe_results.csv
  2. positive vs negative direction accuracy: two-proportion bootstrap on minimal_context.csv
  3. all-pairs method comparison: table of CI for every method pair from knows_vs_says.csv

Output:
  runs/<run>/bootstrap_ci.json   — full results
  runs/<run>/bootstrap_ci.csv    — tidy table (one row per test)
  (printed summary to stdout)

Usage:
  python src/bootstrap_ci.py --run runs/qwen7b
  python src/bootstrap_ci.py --run runs/qwen7b --n_boot 5000 --seed 42
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ATTRS = ["values", "fluency", "factuality", "safety",
         "diversity", "creativity", "helpfulness", "personalisation"]
ALPHAS_CI = [0.90, 0.95, 0.99]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ci(boot_deltas: np.ndarray, alpha: float = 0.95):
    """Percentile CI from bootstrap distribution of a statistic."""
    lo = np.percentile(boot_deltas, (1 - alpha) / 2 * 100)
    hi = np.percentile(boot_deltas, (1 + alpha) / 2 * 100)
    return float(lo), float(hi)


def _p_value(boot_deltas: np.ndarray, observed: float) -> float:
    """Two-sided p-value: fraction of bootstrap samples at least as extreme as observed,
    after re-centering the distribution at 0 (shift trick)."""
    centered = boot_deltas - np.mean(boot_deltas)
    return float(np.mean(np.abs(centered) >= abs(observed)))


def _cv_predict(X: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Ridge 5-fold CV, returns out-of-fold predictions (same logic as probe.py)."""
    from sklearn.linear_model import RidgeCV
    ALPHAS = np.logspace(-1, 5, 13)
    P = np.zeros_like(Y, dtype=float)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
        r = RidgeCV(alphas=ALPHAS).fit((X[tr] - mu) / sd, Y[tr])
        P[te] = r.predict((X[te] - mu) / sd)
    return P


# ---------------------------------------------------------------------------
# Test 1 — probe vs verbal-calibrated: paired bootstrap on per-user ρ
# ---------------------------------------------------------------------------

def test_probe_vs_verbal(run: str, n_boot: int, rng: np.random.Generator) -> dict:
    """
    Paired bootstrap on the difference in Spearman ρ between the best LLM probe
    and calibrated verbal scores, computed over users.

    Strategy: for each bootstrap resample of users, compute ρ(probe_pred, true)
    and ρ(verbal_pred, true) over the sample, then take the difference. The
    observed statistic is the difference on the full dataset.
    """
    print("Test 1: probe vs verbal-calibrated …", flush=True)
    users = pd.read_parquet(os.path.join(run, "users.parquet"))
    verbal = pd.read_parquet(os.path.join(run, "verbal.parquet"))
    assert (users.user_id.values == verbal.user_id.values).all(), \
        "users.parquet and verbal.parquet row order mismatch"

    Y = users[ATTRS].to_numpy(float)
    Yr = Y - Y.mean(1, keepdims=True)
    folds = users.fold.to_numpy()

    # verbal predictions (same CV as verbal_calibrated.py)
    V = verbal[[f"verbal_{k}" for k in ATTRS]].to_numpy(float)
    Vr = V - V.mean(1, keepdims=True)
    Vp_abs = _cv_predict(np.hstack([V, Vr]), Y, folds)
    Vp_rel = _cv_predict(np.hstack([V, Vr]), Yr, folds)

    # probe predictions — load pre-computed out-of-fold scores if available,
    # otherwise fall back to the per-attribute ρ from probe_summary.json
    probe_summary = json.load(open(os.path.join(run, "probe_summary.json")))

    # We re-derive probe OOF predictions from the saved best .npz + H_*.npy
    best_layer = probe_summary["best_layer"]
    best_pool  = probe_summary["best_pool"]
    H_path = os.path.join(run, f"H_{best_pool}.npy")
    has_H = os.path.exists(H_path)

    if has_H:
        H = np.load(H_path, mmap_mode="r")
        X = np.asarray(H[:, best_layer], dtype=np.float32)
        Pp_abs = _cv_predict(X, Y, folds)
        Pp_rel = _cv_predict(X, Yr, folds)
    else:
        # H_*.npy not present locally (normal after downloading only CSVs).
        # Use the per-attribute ρ values directly from probe_summary.json and
        # verbal_calibrated_summary.json to build a scalar-level comparison.
        print("  H_*.npy not found — using summary ρ values (scalar bootstrap).", flush=True)
        return _test_probe_vs_verbal_scalar(run, probe_summary, n_boot, rng)

    # --- paired bootstrap over users
    N = len(users)
    targets = {"abs": (Pp_abs, Vp_abs, Y), "rel": (Pp_rel, Vp_rel, Yr)}
    out = {}
    for t, (Pp, Vp, Yt) in targets.items():
        obs_probe  = np.mean([spearmanr(Pp[:, j], Yt[:, j])[0] for j in range(8)])
        obs_verbal = np.mean([spearmanr(Vp[:, j], Yt[:, j])[0] for j in range(8)])
        obs_diff   = obs_probe - obs_verbal

        boot_diffs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, N, size=N)
            rp = np.mean([spearmanr(Pp[idx, j], Yt[idx, j])[0] for j in range(8)])
            rv = np.mean([spearmanr(Vp[idx, j], Yt[idx, j])[0] for j in range(8)])
            boot_diffs[b] = rp - rv

        ci95 = _ci(boot_diffs, 0.95)
        pv   = _p_value(boot_diffs, obs_diff)
        out[t] = {
            "probe_rho":  round(obs_probe, 4),
            "verbal_rho": round(obs_verbal, 4),
            "delta":      round(obs_diff, 4),
            "ci_95":      [round(ci95[0], 4), round(ci95[1], 4)],
            "p_value":    round(pv, 4),
            "sig_95":     bool(ci95[0] > 0),
        }
        sig = "✓" if out[t]["sig_95"] else "✗"
        print(f"  [{t}] probe {obs_probe:.3f} − verbal {obs_verbal:.3f} "
              f"= {obs_diff:+.3f}  95% CI [{ci95[0]:.3f}, {ci95[1]:.3f}]  "
              f"p={pv:.3f}  {sig}", flush=True)
    return {"method": "paired_bootstrap_oof", **out}


def _test_probe_vs_verbal_scalar(run, probe_summary, n_boot, rng):
    """Fallback when H_*.npy is absent: treat per-attribute ρ from summaries as
    8 paired observations and bootstrap over attributes."""
    vc = json.load(open(os.path.join(run, "verbal_calibrated_summary.json")))
    out = {}
    for t in ["abs", "rel"]:
        probe_rhos  = np.array([probe_summary[f"llm_{t}"][k] for k in ATTRS])
        verbal_rhos = np.array([vc[f"verbal_calibrated_{t}"][k] for k in ATTRS])
        obs_diff = (probe_rhos - verbal_rhos).mean()
        boot_diffs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, 8, size=8)
            boot_diffs[b] = (probe_rhos[idx] - verbal_rhos[idx]).mean()
        ci95 = _ci(boot_diffs, 0.95)
        pv   = _p_value(boot_diffs, obs_diff)
        out[t] = {
            "probe_rho":  round(float(probe_rhos.mean()), 4),
            "verbal_rho": round(float(verbal_rhos.mean()), 4),
            "delta":      round(float(obs_diff), 4),
            "ci_95":      [round(ci95[0], 4), round(ci95[1], 4)],
            "p_value":    round(pv, 4),
            "sig_95":     bool(ci95[0] > 0),
            "note":       "scalar bootstrap over 8 attributes (H_*.npy absent)",
        }
        sig = "✓" if out[t]["sig_95"] else "✗"
        print(f"  [{t}] probe {out[t]['probe_rho']:.3f} − verbal {out[t]['verbal_rho']:.3f} "
              f"= {obs_diff:+.3f}  95% CI [{ci95[0]:.3f}, {ci95[1]:.3f}]  "
              f"p={pv:.3f}  {sig}", flush=True)
    return {"method": "scalar_bootstrap_attrs", **out}


# ---------------------------------------------------------------------------
# Test 2 — positive vs negative direction accuracy
# ---------------------------------------------------------------------------

def test_pos_vs_neg(run: str, n_boot: int, rng: np.random.Generator) -> dict:
    """
    Two-proportion bootstrap: is correct_direction higher for +1 (positive) edits
    than for -1 (negative) edits in minimal_context.csv?
    """
    print("Test 2: positive vs negative accuracy …", flush=True)
    ctx = pd.read_csv(os.path.join(run, "minimal_context.csv"))

    pos = ctx[ctx.dir ==  1].correct_direction.to_numpy()
    neg = ctx[ctx.dir == -1].correct_direction.to_numpy()
    obs_pos  = pos.mean()
    obs_neg  = neg.mean()
    obs_diff = obs_pos - obs_neg

    # per-attribute breakdown
    attr_rows = {}
    for attr in ATTRS:
        p = ctx[(ctx.attr == attr) & (ctx.dir ==  1)].correct_direction.to_numpy()
        n = ctx[(ctx.attr == attr) & (ctx.dir == -1)].correct_direction.to_numpy()
        if len(p) and len(n):
            attr_rows[attr] = {
                "pos": round(float(p.mean()), 3),
                "neg": round(float(n.mean()), 3),
                "delta": round(float(p.mean() - n.mean()), 3),
            }

    # bootstrap: resample (pos, neg) independently
    boot_diffs = np.empty(n_boot)
    Np, Nn = len(pos), len(neg)
    for b in range(n_boot):
        bp = rng.choice(pos, size=Np, replace=True)
        bn = rng.choice(neg, size=Nn, replace=True)
        boot_diffs[b] = bp.mean() - bn.mean()

    ci95 = _ci(boot_diffs, 0.95)
    pv   = _p_value(boot_diffs, obs_diff)
    sig  = bool(ci95[0] > 0)
    print(f"  pos {obs_pos:.3f}  neg {obs_neg:.3f}  "
          f"delta {obs_diff:+.3f}  95% CI [{ci95[0]:.3f}, {ci95[1]:.3f}]  "
          f"p={pv:.3f}  {'✓' if sig else '✗'}", flush=True)
    return {
        "pos_accuracy":  round(float(obs_pos), 4),
        "neg_accuracy":  round(float(obs_neg), 4),
        "delta":         round(float(obs_diff), 4),
        "ci_95":         [round(ci95[0], 4), round(ci95[1], 4)],
        "p_value":       round(pv, 4),
        "sig_95":        sig,
        "per_attribute": attr_rows,
    }


# ---------------------------------------------------------------------------
# Test 3 — all-pairs method comparison (from knows_vs_says.csv)
# ---------------------------------------------------------------------------

def test_all_pairs(run: str, n_boot: int, rng: np.random.Generator) -> dict:
    """
    For each ordered pair of methods, bootstrap the difference in mean ρ.
    Uses knows_vs_says.csv (one mean ρ per method × target); since it has
    no per-user rows, we fall back to attribute-level bootstrap (8 observations).
    """
    print("Test 3: all-pairs method comparison …", flush=True)
    df = pd.read_csv(os.path.join(run, "knows_vs_says.csv"))
    methods = df["method"].unique().tolist()
    pairs_out = {}

    for t in ["abs", "rel"]:
        sub = df[df.target == t].set_index("method")
        pairs_out[t] = {}
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                r1 = np.array([sub.loc[m1, k] for k in ATTRS])
                r2 = np.array([sub.loc[m2, k] for k in ATTRS])
                obs_diff = (r1 - r2).mean()
                boot_diffs = np.empty(n_boot)
                for b in range(n_boot):
                    idx = rng.integers(0, 8, size=8)
                    boot_diffs[b] = (r1[idx] - r2[idx]).mean()
                ci95 = _ci(boot_diffs, 0.95)
                pv   = _p_value(boot_diffs, obs_diff)
                key  = f"{m1} > {m2}"
                pairs_out[t][key] = {
                    "delta":   round(float(obs_diff), 4),
                    "ci_95":   [round(ci95[0], 4), round(ci95[1], 4)],
                    "p_value": round(pv, 4),
                    "sig_95":  bool(ci95[0] > 0),
                }
                sig = "✓" if pairs_out[t][key]["sig_95"] else "✗"
                print(f"  [{t}] {m1:22s} > {m2:22s}  "
                      f"Δ={obs_diff:+.3f}  [{ci95[0]:.3f}, {ci95[1]:.3f}]  "
                      f"p={pv:.3f}  {sig}", flush=True)
    return pairs_out


# ---------------------------------------------------------------------------
# Tidy CSV builder
# ---------------------------------------------------------------------------

def _build_tidy(t1: dict, t2: dict, t3: dict) -> pd.DataFrame:
    rows = []

    # Test 1
    for t in ["abs", "rel"]:
        d = t1[t]
        rows.append({
            "test": "probe_vs_verbal",
            "target": t,
            "comparison": f"LLM probe > verbal calibrated",
            "delta":   d["delta"],
            "ci_lo":   d["ci_95"][0],
            "ci_hi":   d["ci_95"][1],
            "p_value": d["p_value"],
            "sig_95":  d["sig_95"],
        })

    # Test 2
    rows.append({
        "test": "pos_vs_neg",
        "target": "—",
        "comparison": "positive > negative edit accuracy",
        "delta":   t2["delta"],
        "ci_lo":   t2["ci_95"][0],
        "ci_hi":   t2["ci_95"][1],
        "p_value": t2["p_value"],
        "sig_95":  t2["sig_95"],
    })

    # Test 3
    for t, pairs in t3.items():
        for key, d in pairs.items():
            rows.append({
                "test": "method_pair",
                "target": t,
                "comparison": key,
                "delta":   d["delta"],
                "ci_lo":   d["ci_95"][0],
                "ci_hi":   d["ci_95"][1],
                "p_value": d["p_value"],
                "sig_95":  d["sig_95"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(a):
    rng = np.random.default_rng(a.seed)

    print(f"\n{'='*60}")
    print(f"Bootstrap CI  |  n_boot={a.n_boot}  seed={a.seed}  run={a.run}")
    print(f"{'='*60}\n")

    t1 = test_probe_vs_verbal(a.run, a.n_boot, rng)
    print()
    t2 = test_pos_vs_neg(a.run, a.n_boot, rng)
    print()
    t3 = test_all_pairs(a.run, a.n_boot, rng)

    # --- save
    result = {
        "n_boot": a.n_boot,
        "seed": a.seed,
        "probe_vs_verbal": t1,
        "pos_vs_neg": t2,
        "method_pairs": t3,
    }
    json_path = os.path.join(a.run, "bootstrap_ci.json")
    csv_path  = os.path.join(a.run, "bootstrap_ci.csv")
    json.dump(result, open(json_path, "w"), indent=2, default=float)
    tidy = _build_tidy(t1, t2, t3)
    tidy.to_csv(csv_path, index=False)

    # --- print summary
    print(f"\n{'='*60}")
    print("SUMMARY  (95% CI, two-sided)")
    print(f"{'='*60}")
    print(tidy[["test", "target", "comparison", "delta", "ci_lo", "ci_hi",
                "p_value", "sig_95"]].to_string(index=False))
    print(f"\nSaved: {json_path}")
    print(f"       {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run",    default="runs/qwen7b",
                   help="run directory (default: runs/qwen7b)")
    p.add_argument("--n_boot", type=int, default=2000,
                   help="number of bootstrap resamples (default: 2000)")
    p.add_argument("--seed",   type=int, default=0,
                   help="random seed (default: 0)")
    main(p.parse_args())
