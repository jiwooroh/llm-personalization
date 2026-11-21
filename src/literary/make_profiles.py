"""Study 3, step 1: turn each LiteraryTaste reader's survey into a short first-person taste profile.

templates
  full : quality ratings incl. "I don't care about ..." (negation), fiction genres, reasons, hours
  pos  : same but drops the "don't care" sentence (to test negation, cf. Study 1)

Output: <out>/profiles.csv  (user, profile_full, profile_pos)
python make_profiles.py --data ../../LiteraryTaste --out ../../runs/literary_taste
"""
import argparse
from pathlib import Path

import pandas as pd

QUAL = {  # survey item -> phrase
    "Arouses feelings": "stories that move me emotionally",
    "Based on real events": "stories based on real events",
    "Challenges the reader": "writing that challenges me",
    "Entertaining": "being entertained",
    "Gripping content / plot": "a gripping plot",
    "Original style": "an original style",
    "Represent facts and reality": "realism",
    "Rich characters": "rich characters",
    "Fiction's setting precisely presented in detail": "a detailed, vivid setting",
    "Surprising content / plot": "surprises and twists",
    "Thought provoking": "thought-provoking ideas",
    "Skillful and rich language": "skillful, rich language",
}
LEVELS = ["Not important at all", "Slightly important", "Somewhat important", "Very important"]
GENRE_Q = "Which of the following types of texts do you enjoy reading these days? (among fiction):::"
REASON_Q = "Choose specific reasons why you read"
HOURS_Q = "Approximately, how many hours do you spend reading for pleasure per week?:::text_box"


def join(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def profile(g, cols, with_neg=True):
    level = {}
    for q in QUAL:
        for i, lv in enumerate(LEVELS):
            c = f"{q}:::{lv}"
            if c in cols and g[c].notna().any():
                level[q] = i
    very = [QUAL[q] for q, v in level.items() if v == 3]
    some = [QUAL[q] for q, v in level.items() if v == 2]
    none = [QUAL[q] for q, v in level.items() if v == 0]
    genres = [c.split(":::")[1].lower() for c in cols
              if c.startswith(GENRE_Q) and not c.endswith(("free_response", "Others (please specify)"))
              and g[c].notna().any()]
    reasons = [c.split(":::")[1].lower() for c in cols
               if c.startswith(REASON_Q) and g[c].notna().any()][:3]
    hours = pd.to_numeric(g[HOURS_Q], errors="coerce").dropna() if HOURS_Q in cols else pd.Series(dtype=float)

    s = []
    if len(hours):
        s.append(f"I read for pleasure about {hours.iloc[0]:g} hours a week.")
    if genres:
        s.append(f"In fiction I enjoy {join(genres)}.")
    if reasons:
        s.append(f"I read to {join(r[0].lower() + r[1:] for r in reasons)}.")
    if very:
        s.append(f"What matters most to me is {join(very)}.")
    if some:
        s.append(f"I also like {join(some)}.")
    if with_neg and none:
        s.append(f"I don't care about {join(none)}.")
    return " ".join(s)


def main(a):
    df = pd.read_csv(Path(a.data) / "annotated_instances.csv", low_memory=False)
    st = df[pd.to_numeric(df.instance_id, errors="coerce").isna()]
    cols = list(df.columns)
    rows = [{"user": u, "profile_full": profile(g, cols, True), "profile_pos": profile(g, cols, False)}
            for u, g in st.groupby("user")]
    out = pd.DataFrame(rows).sort_values("user")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(a.out) / "profiles.csv", index=False)
    print(f"{len(out)} profiles, mean {out.profile_full.str.split().str.len().mean():.0f} words")
    print("example:", out.profile_full.iloc[0])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../../LiteraryTaste")
    p.add_argument("--out", default="../../runs/literary_taste")
    main(p.parse_args())
