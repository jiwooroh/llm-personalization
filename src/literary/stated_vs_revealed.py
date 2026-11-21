"""LiteraryTaste: does a reader's *stated* taste help predict their *revealed* choices
beyond what everyone shares, and how does it compare with a model fit on their own past choices?

Data: 60 readers x 100 A/B choices (each pair judged by 3 readers) + a survey of
14 literary qualities rated Not / Slightly / Somewhat / Very important.

Models (5-fold GroupKFold over pairs, so every test pair is unseen text):
  position      always predict the majority side of the training set
  length        logistic on char-length difference
  population    logistic on emb(A) - emb(B), one model for everyone (shared taste)
  +stated       population + (PCA-diff x stated profile) interactions
  +stated(shuf) same, profiles permuted across readers (control)
  +revealed     per-reader logistic on PCA-diff, offset by the population logit,
                fit on that reader's own training-fold choices
  +revealed(wrong person)  another reader's personal weights (control)
  crowd         majority vote of the *other two* readers on the same pair (shared-taste ceiling)

Also: say-do consistency — can a reader's revealed weight vector predict their stated ratings?

Usage: python stated_vs_revealed.py --emb minilm
       python stated_vs_revealed.py --emb Qwen2.5-0.5B-Instruct --layer 12
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../LiteraryTaste")
ap.add_argument("--run", default="../../runs/literary_taste")
ap.add_argument("--emb", default="minilm")
ap.add_argument("--layer", type=int, default=None)
ap.add_argument("--pca", type=int, default=32)
ap.add_argument("--n_boot", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--with_length", action="store_true", help="add length diff to the population model")
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
run = Path(args.run)

QUALITIES = ["Arouses feelings", "Based on real events", "Challenges the reader", "Entertaining",
             "Gripping content / plot", "Original style", "Represent facts and reality",
             "Rich characters", "Fiction's setting precisely presented in detail",
             "Surprising content / plot", "Thought provoking", "Skillful and rich language"]
LEVEL = {"Not important at all": 0, "Slightly important": 1, "Somewhat important": 2, "Very important": 3}

# ---------------------------------------------------------------- data
a = pd.read_csv(Path(args.data) / "annotated_instances.csv", low_memory=False)
t = pd.read_csv(Path(args.data) / "preference_tasks.csv").sort_values("id").reset_index(drop=True)
is_rev = pd.to_numeric(a.instance_id, errors="coerce").notna()
rev, st = a[is_rev].copy(), a[~is_rev].copy()

rev["iid"] = rev.instance_id.astype(int)
rev["y"] = np.where(rev["preference:::Text A"].notna(), 1,
                    np.where(rev["preference:::Text B"].notna(), 0, -1))
n_unsure = int((rev.y == -1).sum())
rev = rev[rev.y >= 0][["user", "iid", "y"]].reset_index(drop=True)

# stated profile: one 0-3 rating per quality (all levels of a quality live in separate columns)
qual_present = [q for q in QUALITIES if any(c.startswith(q + ":::") for c in a.columns)]
prof = {}
for u, g in st.groupby("user"):
    row = []
    for q in qual_present:
        v = np.nan
        for lvl, s in LEVEL.items():
            c = f"{q}:::{lvl}"
            if c in g.columns and g[c].notna().any():
                v = s
        row.append(v)
    prof[u] = row
S = pd.DataFrame(prof, index=qual_present).T
S = S.fillna(S.mean())
users = sorted(rev.user.unique())
S = S.loc[users]
Sz = ((S - S.mean()) / (S.std() + 1e-9)).to_numpy()
uidx = {u: i for i, u in enumerate(users)}
rev["u"] = rev.user.map(uidx)

# ---------------------------------------------------------------- features
if args.emb == "minilm":
    E = np.load(run / "emb_minilm.npy").astype(np.float64)
    tag = "minilm"
else:
    H = np.load(run / f"emb_{args.emb}_layers.npy")
    layer = args.layer if args.layer is not None else H.shape[2] // 2
    E = H[:, :, layer].astype(np.float64)
    tag = f"{args.emb}_L{layer}"
D_all = E[:, 0] - E[:, 1]                        # (2000, d)
lens = np.c_[t.text1.fillna("").str.len(), t.text2.fillna("").str.len()].astype(float)
Ldiff = (lens[:, 0] - lens[:, 1])[:, None] / 100.0

iid, y, u = rev.iid.to_numpy(), rev.y.to_numpy(), rev.u.to_numpy()
N = len(rev)

# ---------------------------------------------------------------- helpers
def logit_offset_fit(X, y, off, lam, iters=30):
    """L2 logistic regression with fixed offset (Newton). No intercept penalty."""
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1]); R = lam * np.eye(Xb.shape[1]); R[0, 0] = 1e-6
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Xb @ w + off)))
        g = Xb.T @ (p - y) + R @ w
        Hh = (Xb * (p * (1 - p))[:, None]).T @ Xb + R
        step = np.linalg.solve(Hh, g); w -= step
        if np.abs(step).max() < 1e-6: break
    return w

def fit_lr(X, y, C):
    return LogisticRegression(C=C, max_iter=2000).fit(X, y)

def pick_C(X, y, groups, grid):
    gkf = GroupKFold(3); best, bC = -1, grid[0]
    for C in grid:
        acc = []
        for tr, va in gkf.split(X, y, groups):
            acc.append((fit_lr(X[tr], y[tr], C).predict(X[va]) == y[va]).mean())
        if np.mean(acc) > best: best, bC = np.mean(acc), C
    return bC

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
LAM_GRID = [1, 3, 10, 30, 100]

pred = {k: np.full(N, np.nan) for k in
        ["position", "length", "population", "+stated", "+stated(shuf)", "+revealed",
         "+revealed(wrong person)", "crowd"]}
W_personal = np.zeros((len(users), args.pca + 1))  # last fold's personal weights, for say-do
W_acc = np.zeros_like(W_personal); W_n = np.zeros(len(users))
perm = rng.permutation(len(users))
outer = GroupKFold(5)

for fold, (tr, te) in enumerate(outer.split(D_all[iid], y, iid)):
    tr_pairs = np.unique(iid[tr])
    # position
    pred["position"][te] = float(y[tr].mean() >= 0.5)
    # length
    m = fit_lr(Ldiff[iid[tr]], y[tr], 1.0); pred["length"][te] = m.predict(Ldiff[iid[te]])
    # population
    X = np.c_[D_all, Ldiff * 3][iid] if args.with_length else D_all[iid]
    C = pick_C(X[tr], y[tr], iid[tr], C_GRID)
    pop = fit_lr(X[tr], y[tr], C)
    pred["population"][te] = pop.predict(X[te])
    off_tr = pop.decision_function(X[tr]); off_te = pop.decision_function(X[te])
    # PCA on training-pair diffs (both signs so it is symmetric)
    pca = PCA(args.pca, random_state=0).fit(np.r_[D_all[tr_pairs], -D_all[tr_pairs]])
    Z = pca.transform(D_all)[iid]; Z /= Z[tr].std(0) + 1e-9
    # +stated : population offset + interactions z_k * s_j  (fit with offset via Newton)
    for key, prof_idx in [("+stated", np.arange(len(users))), ("+stated(shuf)", perm)]:
        SI = Sz[prof_idx][u]                                  # (N, K)
        Xi = (Z[:, :, None] * SI[:, None, :]).reshape(N, -1)  # (N, pca*K)
        # choose lambda by inner GroupKFold on train
        best, bl = -1, LAM_GRID[0]
        for lam in LAM_GRID:
            acc = []
            for itr, iva in GroupKFold(3).split(Xi[tr], y[tr], iid[tr]):
                w = logit_offset_fit(Xi[tr][itr], y[tr][itr], off_tr[itr], lam)
                s = np.c_[np.ones(len(iva)), Xi[tr][iva]] @ w + off_tr[iva]
                acc.append(((s > 0) == y[tr][iva]).mean())
            if np.mean(acc) > best: best, bl = np.mean(acc), lam
        w = logit_offset_fit(Xi[tr], y[tr], off_tr, bl)
        pred[key][te] = (np.c_[np.ones(len(te)), Xi[te]] @ w + off_te > 0).astype(float)
    # +revealed : per-user model with offset, global lambda chosen by inner CV
    best, bl = -1, LAM_GRID[0]
    for lam in LAM_GRID:
        acc = []
        for itr, iva in GroupKFold(3).split(Z[tr], y[tr], iid[tr]):
            ii_tr, ii_va = tr[itr], tr[iva]
            for k in range(len(users)):
                a_ = ii_tr[u[ii_tr] == k]; b_ = ii_va[u[ii_va] == k]
                if len(b_) == 0 or len(a_) < 5: continue
                w = logit_offset_fit(Z[a_], y[a_], off_tr[np.searchsorted(tr, a_)], lam)
                s = np.c_[np.ones(len(b_)), Z[b_]] @ w + off_tr[np.searchsorted(tr, b_)]
                acc.extend(((s > 0) == y[b_]).tolist())
        if np.mean(acc) > best: best, bl = np.mean(acc), lam
    Wf = np.zeros((len(users), args.pca + 1))
    for k in range(len(users)):
        a_ = tr[u[tr] == k]
        Wf[k] = logit_offset_fit(Z[a_], y[a_], off_tr[np.searchsorted(tr, a_)], bl)
        W_acc[k] += Wf[k]; W_n[k] += 1
    sh = np.roll(np.arange(len(users)), 1)
    for k in range(len(users)):
        b_ = te[u[te] == k]; o = off_te[np.searchsorted(te, b_)]
        Zb = np.c_[np.ones(len(b_)), Z[b_]]
        pred["+revealed"][b_] = (Zb @ Wf[k] + o > 0)
        pred["+revealed(wrong person)"][b_] = (Zb @ Wf[sh[k]] + o > 0)
    print(f"fold {fold}: C={C} lam_rev={bl}", flush=True)

# crowd: other two readers on the same pair (tie -> 0.5 expected accuracy)
votes = rev.groupby("iid").y.agg(["sum", "count"])
for i in range(N):
    s_, c_ = votes.loc[iid[i]]
    s_o, c_o = s_ - y[i], c_ - 1
    pred["crowd"][i] = 0.5 if c_o == 0 or 2 * s_o == c_o else float(2 * s_o > c_o)

# ---------------------------------------------------------------- metrics
def acc_vec(p):  # tie (0.5) counts half
    return np.where(p == 0.5, 0.5, (p == y).astype(float))

per_user = pd.DataFrame({k: pd.Series(acc_vec(v)).groupby(u).mean() for k, v in pred.items()})
per_user.index = users
summary = {"emb": tag, "n_choices": N, "n_unsure_dropped": n_unsure, "n_users": len(users),
           "qualities": qual_present, "acc": {}, "vs_population": {}}
for k in pred:
    summary["acc"][k] = round(float(acc_vec(pred[k]).mean()), 4)

def boot_diff(a_, b_):
    d = a_ - b_; idx = rng.integers(0, len(d), (args.n_boot, len(d)))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return dict(delta=round(float(d.mean()), 4), ci=[round(float(lo), 4), round(float(hi), 4)],
                p=round(float(p), 4), users_better=int((d > 0).sum()), users_worse=int((d < 0).sum()))

for k in ["+stated", "+stated(shuf)", "+revealed", "+revealed(wrong person)", "crowd", "length"]:
    summary["vs_population"][k] = boot_diff(per_user[k].values, per_user["population"].values)
summary["revealed_vs_stated"] = boot_diff(per_user["+revealed"].values, per_user["+stated"].values)

# inter-annotator agreement (pairwise, over pairs with >=2 decided votes)
agree = []
for _, g in rev.groupby("iid"):
    v = g.y.values
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            agree.append(v[i] == v[j])
summary["pairwise_agreement"] = round(float(np.mean(agree)), 4)

# distinctive readers: crowd-agreement vs personal gain
cr = per_user["crowd"].values
for k in ["+stated", "+revealed"]:
    r, p = spearmanr(cr, per_user[k].values - per_user["population"].values)
    summary[f"crowd_agreement_vs_gain_{k}"] = dict(rho=round(float(r), 3), p=round(float(p), 4))

summary["revealed_vs_wrong_person"] = boot_diff(per_user["+revealed"].values,
                                                per_user["+revealed(wrong person)"].values)
summary["length_pref"] = dict(
    p_choose_longer=round(float(np.mean((Ldiff[iid, 0] > 0) == (y == 1))), 4))

# say-do: predict each stated rating from revealed personal weights.
# LOO-CV correlations are biased negative when features carry no signal, so every rho is
# compared with a permutation null of the *same* procedure (stated ratings shuffled across readers).
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
Wm = (W_acc / W_n[:, None])[:, 1:]
Wm = (Wm - Wm.mean(0)) / (Wm.std(0) + 1e-12)
def cv_rho(Wx, yy):
    pr = np.zeros(len(yy))
    for a_, b_ in KFold(5, shuffle=True, random_state=0).split(Wx):
        pr[b_] = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(Wx[a_], yy[a_]).predict(Wx[b_])
    return spearmanr(pr, yy)[0] if np.std(pr) > 0 else 0.0
sd = {}
N_PERM = 200
for j, q in enumerate(qual_present):
    yy = S.iloc[:, j].values.astype(float)
    r = cv_rho(Wm, yy)
    null = np.array([cv_rho(Wm, rng.permutation(yy)) for _ in range(N_PERM)])
    p = (1 + (null >= r).sum()) / (N_PERM + 1)
    sd[q] = dict(rho=round(float(r), 3), null_mean=round(float(null.mean()), 3),
                 excess=round(float(r - null.mean()), 3), p_perm=round(float(p), 3))
summary["say_do"] = sd
summary["say_do_mean_excess"] = round(float(np.mean([v["excess"] for v in sd.values()])), 3)
summary["say_do_n_sig"] = int(sum(v["p_perm"] < 0.05 for v in sd.values()))

out = run / (f"svr_{tag}" + ("_len" if args.with_length else ""))
out.mkdir(exist_ok=True)
per_user.to_csv(out / "per_user_accuracy.csv")
json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))
