"""Study 3, step 2: show the model a reader's taste profile together with a text pair and ask which text
that reader would prefer. Save (a) the last-token hidden state at selected layers ("knows") and
(b) logit("A") - logit("B") for the answer ("says").

Conditions
  none   no profile ("a typical reader")          -> shared taste only, one row per pair
  own    the reader's own profile                 -> one row per judgment
  other  another reader's profile (fixed derangement) -> control for "uses a profile at all"
  own_pos  own profile without the "don't care" sentence (negation test, optional)
Every prompt is run in both orders (text1 as A, then as B) so position bias cancels in the analysis.

python extract_joint.py --model Qwen/Qwen2.5-7B-Instruct --out ../../runs/literary_taste/joint_qwen7b
python extract_joint.py --model Qwen/Qwen2.5-0.5B-Instruct --limit_users 4 --out ../../runs/literary_taste/joint_test
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROMPT = ("{reader}\n\nText A:\n{a}\n\nText B:\n{b}\n\n"
          "Which text would {who} prefer? Answer with a single letter, A or B.")


def load_judgments(data):
    a = pd.read_csv(Path(data) / "annotated_instances.csv", low_memory=False)
    rev = a[pd.to_numeric(a.instance_id, errors="coerce").notna()].copy()
    rev["iid"] = rev.instance_id.astype(int)
    rev["y"] = np.where(rev["preference:::Text A"].notna(), 1,
                        np.where(rev["preference:::Text B"].notna(), 0, -1))
    return rev[rev.y >= 0][["user", "iid", "y"]].reset_index(drop=True)


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(Path(a.data) / "preference_tasks.csv").set_index("id")
    prof = pd.read_csv(a.profiles).set_index("user")
    J = load_judgments(a.data)
    users = sorted(J.user.unique())
    if a.limit_users:
        users = users[:a.limit_users]
        J = J[J.user.isin(users)].reset_index(drop=True)
    all_users = sorted(prof.index)
    other_of = {u: all_users[(i + 1) % len(all_users)] for i, u in enumerate(all_users)}
    J.to_csv(out / "judgments.csv", index=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    tok.truncation_side = "left"   # never cut the question / answer position at the end
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16 if cuda else torch.float32, device_map="auto" if cuda else None).eval()
    n_layers = model.config.num_hidden_layers + 1
    layers = sorted(set(list(range(0, n_layers, a.layer_step)) + [n_layers - 1]))
    idA = tok.encode("A", add_special_tokens=False)[0]
    idB = tok.encode("B", add_special_tokens=False)[0]
    dev = next(model.parameters()).device

    def build(reader_text, who, a_text, b_text):
        msg = PROMPT.format(reader=reader_text, a=a_text, b=b_text, who=who)
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                       add_generation_prompt=True)

    meta = {"model": a.model, "layers": layers, "conditions": {}}
    for cond in a.conditions.split(","):
        rows = (pd.DataFrame({"iid": sorted(J.iid.unique())}) if cond == "none"
                else J[["user", "iid"]].copy())
        prompts = []
        for r in rows.itertuples():
            if cond == "none":
                reader, who = "Consider a typical adult reader of fiction.", "a typical reader"
            else:
                u = r.user if cond.startswith("own") else other_of[r.user]
                col = "profile_pos" if cond == "own_pos" else "profile_full"
                reader, who = f"Here is how a reader describes their taste:\n\"{prof.loc[u, col]}\"", "this reader"
            t1, t2 = str(t.loc[r.iid, "text1"]), str(t.loc[r.iid, "text2"])
            prompts.append(build(reader, who, t1, t2))   # order 0: text1 = A
            prompts.append(build(reader, who, t2, t1))   # order 1: text1 = B
        H = np.zeros((len(prompts), len(layers), model.config.hidden_size), dtype=np.float16)
        LP = np.zeros(len(prompts), dtype=np.float32)
        t0 = time.time()
        order = np.argsort([len(p) for p in prompts])      # length-sorted batches = less padding
        with torch.no_grad():
            for s in range(0, len(prompts), a.batch_size):
                idx = order[s:s + a.batch_size]
                enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                          truncation=True, max_length=a.max_len).to(dev)
                o = model(**enc, output_hidden_states=True)
                for k, L in enumerate(layers):
                    H[idx, k] = o.hidden_states[L][:, -1].float().cpu().numpy().astype(np.float16)
                lg = o.logits[:, -1].float()
                LP[idx] = (lg[:, idA] - lg[:, idB]).cpu().numpy()
                if (s // a.batch_size) % 50 == 0:
                    done = s + len(idx)
                    print(f"[{cond}] {done}/{len(prompts)}  {time.time() - t0:.0f}s", flush=True)
        rows.to_csv(out / f"rows_{cond}.csv", index=False)
        np.save(out / f"H_{cond}.npy", H.reshape(len(rows), 2, len(layers), -1))
        np.save(out / f"LP_{cond}.npy", LP.reshape(len(rows), 2))
        meta["conditions"][cond] = {"n_rows": len(rows), "seconds": round(time.time() - t0)}
        print(f"[{cond}] done {len(rows)} rows x 2 orders in {time.time() - t0:.0f}s", flush=True)
    json.dump(meta, open(out / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data", default="../../LiteraryTaste")
    p.add_argument("--profiles", default="../../runs/literary_taste/profiles.csv")
    p.add_argument("--out", default="../../runs/literary_taste/joint_qwen7b")
    p.add_argument("--conditions", default="none,own,other,own_pos")
    p.add_argument("--layer_step", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--limit_users", type=int, default=0)
    main(p.parse_args())
