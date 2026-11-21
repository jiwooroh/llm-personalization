"""Embed the 4,000 LiteraryTaste snippets (text1/text2 of 2,000 pairs).

Outputs (runs/literary_taste/):
  emb_minilm.npy          (2000, 2, 384)  sentence-transformers/all-MiniLM-L6-v2
  emb_<llm>_L<k>.npy      (2000, 2, d)    mean-pooled hidden state of a causal LM, every layer saved

Usage:
  python embed_texts.py --data ../../LiteraryTaste --out ../../runs/literary_taste
  python embed_texts.py ... --llm Qwen/Qwen2.5-0.5B-Instruct
CPU is fine (MiniLM ~1 min, Qwen-0.5B ~10-20 min).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

def log(*a):
    print(*a, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../LiteraryTaste")
ap.add_argument("--out", default="../../runs/literary_taste")
ap.add_argument("--llm", default=None, help="HF id of a causal LM, e.g. Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--max_tokens", type=int, default=384)
ap.add_argument("--skip_minilm", action="store_true")
args = ap.parse_args()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
t = pd.read_csv(Path(args.data) / "preference_tasks.csv").sort_values("id").reset_index(drop=True)
assert (t.id.values == np.arange(len(t))).all()
texts = list(t.text1.fillna("")) + list(t.text2.fillna(""))
N = len(t)
log(f"{N} pairs, {len(texts)} texts")

if not args.skip_minilm:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    t0 = time.time()
    E = m.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(out / "emb_minilm.npy", np.stack([E[:N], E[N:]], 1).astype(np.float32))
    log(f"MiniLM done {time.time()-t0:.0f}s")

if args.llm:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
    tok = AutoTokenizer.from_pretrained(args.llm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.float32)
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    L = model.config.num_hidden_layers + 1
    d = model.config.hidden_size
    H = np.zeros((len(texts), L, d), dtype=np.float32)
    bs = 8
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), bs), desc=args.llm):
            enc = tok(texts[i:i+bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_tokens)
            hs = model(**enc, output_hidden_states=True).hidden_states  # L x (b,T,d)
            m_ = enc["attention_mask"].unsqueeze(-1).float()
            for l, h in enumerate(hs):
                H[i:i+bs, l] = ((h * m_).sum(1) / m_.sum(1)).numpy()
    tag = args.llm.split("/")[-1]
    np.save(out / f"emb_{tag}_layers.npy", np.stack([H[:N], H[N:]], 1).astype(np.float16))
    log(f"{tag} done {time.time()-t0:.0f}s  shape {(N, 2, L, d)}")
