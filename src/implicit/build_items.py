"""Study 4, step 1: build explicit / implicit / no-preference contexts from PersonaMem-v2 (arXiv 2512.06688, CC BY 4.0).

Each PersonaMem-v2 benchmark row gives, for one user preference:
  preference                    short explicit statement ("Attends local food truck festivals")
  related_conversation_snippet  a real multi-turn exchange in which the user only IMPLIES that preference
  user_query                    a later request whose best answer depends on the preference
  correct_answer / incorrect_answers   1 personalised reply + 3 plausible replies that ignore or conflict with it

Conditions (same query, same 4 replies; only the preceding turns differ):
  none      no preference turn (control: what the replies alone give away)
  explicit  user states the preference outright, assistant acknowledges briefly
  implicit  the PersonaMem-v2 snippet (assistant replies truncated to keep lengths comparable)
  explicit_far / implicit_far   same, followed by 2 unrelated filler exchanges (multi-turn distance)
  implicit_attr  implicit + "That's something I personally care about." (attribution only, no preference content)
  implicit_recall  built at extraction time: implicit + the model's own summary of what it knows about the user

python build_items.py --n_items 800 --out ../../runs/implicit/items.jsonl
"""
import argparse
import ast
import json
import random
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/main/benchmark/text/benchmark.csv"
PREF_TYPES = ["neutral_preferences", "stereotypical_pref", "anti_stereotypical_pref"]
ACK = "Thanks for telling me, I'll keep that in mind."
ATTR = "By the way, that's something I personally care about a lot."


def clip(text, n_words):
    w = text.split()
    return text if len(w) <= n_words else " ".join(w[:n_words]) + " ..."


def snippet_turns(raw, asst_words):
    turns = json.loads(raw)
    return [{"role": t["role"], "content": clip(t["content"], asst_words) if t["role"] == "assistant" else t["content"]}
            for t in turns]


def main(a):
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.parent / "personamem_v2_benchmark.csv"
    if not cache.exists():
        print("downloading PersonaMem-v2 benchmark ->", cache)
        urllib.request.urlretrieve(URL, cache)
    d = pd.read_csv(cache)
    d = d[(d.who == "self") & (~d.updated.astype(bool)) & d.pref_type.isin(PREF_TYPES)].reset_index(drop=True)
    rng = random.Random(a.seed)
    idx = list(range(len(d)))
    rng.shuffle(idx)
    pick = sorted(idx[:a.n_items])

    rows = []
    for i in pick:
        r = d.iloc[i]
        implicit = snippet_turns(r.related_conversation_snippet, a.asst_words)
        explicit = [{"role": "user", "content": f"A quick note about me: {r.preference}."},
                    {"role": "assistant", "content": ACK}]
        # fillers: snippets from OTHER personas with a different preference topic (no leakage)
        pool = d[(d.persona_id != r.persona_id) & (d.topic_preference != r.topic_preference)]
        fill = []
        for j in pool.sample(a.n_fillers, random_state=a.seed + i).index:
            t = snippet_turns(d.loc[j, "related_conversation_snippet"], a.asst_words)
            fill += t[:2]   # one user turn + one assistant turn each
        q = ast.literal_eval(r.user_query)["content"]
        options = [r.correct_answer] + json.loads(r.incorrect_answers)
        rows.append({
            "item": int(i), "persona": int(r.persona_id), "pref_type": r.pref_type, "topic": r.topic_preference,
            "scenario": r.conversation_scenario, "preference": r.preference, "query": q, "options": options,
            "contexts": {"none": [], "explicit": explicit, "implicit": implicit,
                         "explicit_far": explicit + fill, "implicit_far": implicit + fill,
                         "implicit_attr": implicit + [{"role": "user", "content": ATTR}, {"role": "assistant", "content": ACK}]},
        })
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} items, {len({r['persona'] for r in rows})} personas -> {out}")
    print(pd.Series([r["pref_type"] for r in rows]).value_counts().to_dict())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_items", type=int, default=800)
    p.add_argument("--asst_words", type=int, default=80, help="truncate assistant turns to this many words")
    p.add_argument("--n_fillers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="../../runs/implicit/items.jsonl")
    main(p.parse_args())
