"""Presentation figures from a finished run.

python src/make_figures.py --run runs/qwen7b
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ATTRS

BLUE, ORANGE, GREEN, GRAY = "#2f6fde", "#e8743b", "#2a9d6f", "#9ca3af"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160, "font.size": 11})


def main(a):
    out = os.path.join(a.run, "figures")
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(os.path.join(a.run, "probe_results.csv"))

    # 1. layer curve
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, t in zip(axs, ["abs", "rel"]):
        for pool, c in [("mean", BLUE), ("last", ORANGE)]:
            s = df[(df.source == "llm") & (df.pool == pool) & (df.target == t)].sort_values("layer")
            if len(s):
                ax.plot(s.layer, s["mean"], marker="o", ms=3, color=c, label=f"LLM hidden state ({pool})")
        for src, c, ls in [("tfidf", GRAY, "--"), ("sbert", GREEN, "--"), ("shuffled_control", "black", ":")]:
            s = df[(df.source == src) & (df.target == t)]
            if len(s):
                ax.axhline(s["mean"].iloc[0], color=c, ls=ls, label=src)
        ax.set_title({"abs": "absolute rating", "rel": "within-person relative priority"}[t])
        ax.set_xlabel("layer")
    axs[0].set_ylabel("mean Spearman ρ (8 attributes, held-out users)")
    axs[1].legend(fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(out, "1_layer_curve.png")); plt.close()

    # 2. per-attribute: internal probe vs verbal vs text baseline
    summ = json.load(open(os.path.join(a.run, "probe_summary.json")))
    bars = {"LLM internal (probe)": summ["llm_abs"], "TF-IDF": summ["tfidf_abs"]}
    if "sbert_abs" in summ:
        bars["Sentence embedding"] = summ["sbert_abs"]
    vp = os.path.join(a.run, "verbal_summary.json")
    if os.path.exists(vp):
        bars["LLM verbal rating"] = json.load(open(vp))["abs"]
    x = np.arange(len(ATTRS)); w = 0.8 / len(bars)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for i, (k, v) in enumerate(bars.items()):
        ax.bar(x + i * w, [v[a_] for a_ in ATTRS], w, label=k, color=[BLUE, GRAY, GREEN, ORANGE][i])
    ax.set_xticks(x + w * (len(bars) - 1) / 2); ax.set_xticklabels(ATTRS, rotation=20)
    ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("Spearman ρ with person's own survey rating")
    ax.legend(fontsize=8, frameon=False, ncol=4); plt.tight_layout()
    plt.savefig(os.path.join(out, "2_per_attribute.png")); plt.close()

    # 3. ladder + 4. in-context edits
    lp = os.path.join(a.run, "minimal_ladder.csv")
    if os.path.exists(lp):
        L = pd.read_csv(lp)
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for k in ATTRS:
            s = L[L.attr == k].groupby("level").target_score.mean()
            ax.plot(s.index, s.values, marker="o", ms=3, label=k)
        ax.set_xticks(range(6)); ax.set_xticklabels(["do not\nwant", "need\nnot", "don't\ncare", "nice\nif", "really\nwant", "essential"], fontsize=8)
        ax.set_ylabel("probe-read importance"); ax.legend(fontsize=7, frameon=False, ncol=2)
        plt.tight_layout(); plt.savefig(os.path.join(out, "3_ladder.png")); plt.close()
        C = pd.read_csv(os.path.join(a.run, "minimal_context.csv"))
        g = C.groupby(["attr", "dir"]).correct_direction.mean().unstack()
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.bar(x - 0.2, g.loc[ATTRS, 1], 0.4, label='"matters most" appended', color=BLUE)
        ax.bar(x + 0.2, g.loc[ATTRS, -1], 0.4, label='"does not matter" appended', color=ORANGE)
        ax.axhline(0.5, color="black", ls=":", lw=1); ax.set_ylim(0, 1)
        ax.set_xticks(x); ax.set_xticklabels(ATTRS, rotation=20); ax.set_ylabel("moved in stated direction")
        ax.legend(fontsize=8, frameon=False); plt.tight_layout()
        plt.savefig(os.path.join(out, "4_context_edits.png")); plt.close()
    print("figures ->", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/qwen7b")
    main(p.parse_args())
