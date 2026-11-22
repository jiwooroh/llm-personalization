"""Study 4, step 2 (GPU): what the model SAYS and what it KNOWS about an explicit vs implicit preference.

For every item x condition (none / explicit / implicit / explicit_far / implicit_far):
  says_mc   multiple choice: the 4 replies are listed as A-D after the user's query and the model is asked which suits
            the user. Run in 4 cyclic rotations so every reply appears in every position (position bias cancels).
            Stored: log-softmax over the letters A-D, shape (rotations, letters).
  says_gen  each reply is placed as the assistant turn; mean log-probability of its tokens = how likely the model
            is to produce that reply itself.
  knows     hidden state at the last token of each reply (every --layer_step layers), for a probe trained across items.
  lens      logit lens on the multiple-choice prompt: letter log-probs read from intermediate layers (final norm + head),
            to see at which depth the answer favours the personalised reply and whether it is lost later.
  recall    for implicit_recall the model first writes (greedy) what it knows about the user; that exchange is
            inserted before the query. Stored in recall.jsonl.

python extract.py --model Qwen/Qwen2.5-7B-Instruct --items ../../runs/implicit/items.jsonl --out ../../runs/implicit/qwen7b
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_model  # noqa: E402

CONDS = ["none", "explicit", "implicit", "explicit_far", "implicit_far", "implicit_attr", "implicit_recall"]
RECALL_Q = ("Before I ask my next question: based on our conversation so far, what do you know about my personal "
            "preferences or interests? Answer in one or two sentences.")
LETTERS = ["A", "B", "C", "D"]


def mc_prompt(query, opts):
    body = "\n".join(f"{L}) {o}" for L, o in zip(LETTERS, opts))
    return (f"{query}\n\nHere are four possible replies:\n{body}\n\n"
            "Which reply would suit me best, given what you know about me? Answer with a single letter: A, B, C, or D.")


def letter_ids(tok):
    ids = []
    for L in LETTERS:
        t = tok.encode(L, add_special_tokens=False)
        assert len(t) == 1, (L, t)
        ids.append(t[0])
    return ids


def build_option_seq(tok, msgs, option, max_len):
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": option}], tokenize=False)
    assert full.startswith(prefix), "chat template is not prefix-stable"
    s = len(prefix) + full[len(prefix):].index(option)
    e = s + len(option)
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    span = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if a < e and b > s]
    ids = enc["input_ids"]
    cut = max(0, len(ids) - max_len)          # left-truncate old context, never the reply
    return ids[cut:], span[0] - cut, span[-1] + 1 - cut


def final_norm(model):
    for name in ["model.norm", "model.model.norm", "transformer.ln_f"]:
        obj = model
        try:
            for part in name.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("final norm not found")


@torch.no_grad()
def generate_recalls(model, tok, items, batch_size, max_len, max_new=60):
    dev = next(model.parameters()).device
    prompts = []
    for it in items:
        msgs = it["contexts"]["implicit"] + [{"role": "user", "content": RECALL_Q}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        prompts.append(list(ids)[-max_len:])
    res = [None] * len(items)
    order = np.argsort([len(p) for p in prompts])
    for b0 in range(0, len(order), batch_size):
        idx = order[b0:b0 + batch_size]
        L = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for k, i in enumerate(idx):
            ids[k, L - len(prompts[i]):] = torch.tensor(prompts[i])
            att[k, L - len(prompts[i]):] = 1
        g = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new,
                           do_sample=False, pad_token_id=tok.pad_token_id)
        for k, i in enumerate(idx):
            res[i] = tok.decode(g[k, L:], skip_special_tokens=True).strip()
    return res


@torch.no_grad()
def run_batches(model, tok, seqs, batch_size, fn):
    dev = next(model.parameters()).device
    order = np.argsort([len(s[0]) for s in seqs])
    for b0 in range(0, len(order), batch_size):
        idx = order[b0:b0 + batch_size]
        L = max(len(seqs[i][0]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        offs = []
        for k, i in enumerate(idx):                     # left padding: last real token is at position L-1
            s = seqs[i][0]
            ids[k, L - len(s):] = torch.tensor(s)
            att[k, L - len(s):] = 1
            offs.append(L - len(s))
        out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False)
        fn(idx, out, offs)
        del out


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(x) for x in open(a.items)]
    if a.limit:
        items = items[:a.limit]
    global CONDS
    if a.conds:
        CONDS = [c for c in CONDS if c in a.conds.split(",")]
    if a.device == "mps":   # Mac pilot: half precision on the Apple GPU
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        tok.pad_token = tok.pad_token or tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to("mps").eval()
    else:
        tok, model = load_model(a.model)
    tok.padding_side = "left"
    nL = model.config.num_hidden_layers + 1
    layers = list(range(0, nL, a.layer_step))
    if nL - 1 not in layers:
        layers.append(nL - 1)
    d = model.config.hidden_size
    N, C = len(items), len(CONDS)
    lid = letter_ids(tok)

    # ---- implicit_recall: let the model state what it knows, then use that exchange as context
    t0 = time.time()
    recalls = generate_recalls(model, tok, items, a.batch_size, a.max_len) if "implicit_recall" in CONDS else [""] * N
    for it, r in zip(items, recalls):
        it["contexts"]["implicit_recall"] = it["contexts"]["implicit"] + [
            {"role": "user", "content": RECALL_Q}, {"role": "assistant", "content": r}]
    with open(out / "recall.jsonl", "w") as f:
        for it, r in zip(items, recalls):
            f.write(json.dumps({"item": it["item"], "preference": it["preference"], "recall": r}) + "\n")
    print(f"recall done {time.time() - t0:.0f}s", flush=True)

    MC = np.zeros((N, C, 4, 4), np.float32)          # item, cond, rotation, letter (log-softmax over A-D)
    MCL = np.zeros((N, C, 4, len(layers), 4), np.float32)   # same, read from intermediate layers (logit lens)
    norm, head = final_norm(model), model.get_output_embeddings()
    LP = np.zeros((N, C, 4), np.float32)             # item, cond, option: mean token log-prob of the reply
    H = np.zeros((N, C, 4, len(layers), d), np.float16)

    # ---- says_mc
    seqs = []
    for i, it in enumerate(items):
        for c, cond in enumerate(CONDS):
            for r in range(4):
                opts = [it["options"][(j + r) % 4] for j in range(4)]   # rotation r: option (j+r)%4 sits at letter j
                msgs = it["contexts"][cond] + [{"role": "user", "content": mc_prompt(it["query"], opts)}]
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
                if hasattr(ids, "input_ids"):
                    ids = ids["input_ids"]
                seqs.append((list(ids)[-a.max_len:], (i, c, r)))

    def take_mc(idx, out, offs):
        lg = torch.log_softmax(out.logits[:, -1, lid].float(), -1).cpu().numpy()
        lens = []
        for L in layers:
            h = out.hidden_states[L][:, -1]
            h = h if L == len(out.hidden_states) - 1 else norm(h)   # last hidden state is already normed
            lens.append(torch.log_softmax(head(h)[:, lid].float(), -1).cpu().numpy())
        for k, j in enumerate(idx):
            i, c, r = seqs[j][1]
            MC[i, c, r] = lg[k]
            MCL[i, c, r] = np.stack([x[k] for x in lens])
    run_batches(model, tok, seqs, a.batch_size, take_mc)
    print(f"mc done {time.time() - t0:.0f}s", flush=True)

    if a.mc_only:   # quick pilot: skip reply scoring and hidden states
        np.save(out / "MC.npy", MC); np.save(out / "MC_lens.npy", MCL)
        json.dump({"model": a.model, "conds": CONDS, "layers": layers, "n_items": N, "mc_only": True,
                   "item_ids": [it["item"] for it in items]}, open(out / "meta.json", "w"), indent=1)
        print("saved (mc only) ->", out); return

    # ---- says_gen + knows
    seqs = []
    for i, it in enumerate(items):
        for c, cond in enumerate(CONDS):
            msgs = it["contexts"][cond] + [{"role": "user", "content": it["query"]}]
            for o, opt in enumerate(it["options"]):
                ids, s, e = build_option_seq(tok, msgs, opt, a.max_len)
                seqs.append((ids, (i, c, o, s, e)))

    def take_opt(idx, out, offs):
        logp = torch.log_softmax(out.logits.float(), -1)
        for k, j in enumerate(idx):
            ids, (i, c, o, s, e) = seqs[j]
            s2, e2 = s + offs[k], e + offs[k]
            tgt = torch.tensor(ids[s:e], device=logp.device)
            LP[i, c, o] = logp[k, s2 - 1:e2 - 1].gather(-1, tgt[:, None]).mean().item()
            for li, L in enumerate(layers):
                H[i, c, o, li] = out.hidden_states[L][k, e2 - 1].float().cpu().numpy().astype(np.float16)
    run_batches(model, tok, seqs, a.batch_size, take_opt)
    print(f"options done {time.time() - t0:.0f}s", flush=True)

    np.save(out / "MC.npy", MC)
    np.save(out / "MC_lens.npy", MCL)
    np.save(out / "LP.npy", LP)
    np.save(out / "H.npy", H)
    json.dump({"model": a.model, "conds": CONDS, "layers": layers, "n_items": N, "items": a.items,
               "item_ids": [it["item"] for it in items], "seconds": round(time.time() - t0)},
              open(out / "meta.json", "w"), indent=1)
    print("saved ->", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--items", default="../../runs/implicit/items.jsonl")
    p.add_argument("--out", default="../../runs/implicit/qwen7b")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--layer_step", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--mc_only", action="store_true")
    p.add_argument("--conds", default="", help="comma-separated subset of conditions (default: all)")
    p.add_argument("--device", default="auto", choices=["auto", "mps"])
    main(p.parse_args())
