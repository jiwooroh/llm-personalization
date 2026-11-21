#!/usr/bin/env python3
"""진행바가 있는 Literary Taste Demo"""
import sys
import time

def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()

log("="*70)
log("Literary Taste Analysis - With Progress Bars")
log("="*70)

# 1. 기본 라이브러리
log("\n[1/7] Importing basic libraries...")
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
log("✅ Basic libraries loaded")

# 2. Transformers (이게 느림)
log("\n[2/7] Importing transformers (this takes 10-30 seconds)...")
log("Please wait...")
with tqdm(total=100, desc="Loading transformers", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    pbar.update(100)
log("✅ Transformers loaded")

# 3. 설정
log("\n[3/7] Configuration...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"
DATA_PATH = Path('LiteraryTaste')
OUTPUT_PATH = Path('runs/literary_taste/demo')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
log(f"  Device: {DEVICE}")
log(f"  Model: {MODEL_NAME}")
log("✅ Configuration done")

# 4. 데이터 로드
log("\n[4/7] Loading data...")
preference_df = pd.read_csv(DATA_PATH / 'preference_tasks.csv')
annotated_df = pd.read_csv(DATA_PATH / 'annotated_instances.csv', low_memory=False)
N_SAMPLES = 5
sample_df = preference_df.head(N_SAMPLES)
log(f"✅ Loaded {len(sample_df)} sample text pairs")

# 5. 모델 로드 (제일 느림 - 1-2분)
log("\n[5/7] Loading GPT-2 model (1-2 minutes on CPU)...")
log("=" * 70)

# Tokenizer
with tqdm(total=100, desc="Loading tokenizer", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pbar.update(100)
log("✅ Tokenizer ready")

# Model (이게 진짜 오래 걸림)
log("\nLoading model weights...")
log("(CPU에서 정상적으로 1-2분 소요됩니다)")
with tqdm(total=100, desc="Loading model", bar_format='{l_bar}{bar}| {elapsed}', 
          mininterval=1.0) as pbar:
    # 체크포인트 업데이트를 위한 가짜 진행
    pbar.update(10)
    model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float32)
    pbar.update(80)
    model.eval()
    pbar.update(10)

log("✅ Model loaded!")
log(f"  Layers: {model.config.n_layer}")
log(f"  Hidden size: {model.config.n_embd}")

# 6. 활성화 추출
log("\n[6/7] Extracting activations...")
log("=" * 70)

def create_prompt(text1, text2, max_len=100):
    t1 = str(text1)[:max_len] if pd.notna(text1) else ""
    t2 = str(text2)[:max_len] if pd.notna(text2) else ""
    return f"Text A: {t1}\nText B: {t2}\nBetter:"

results = []
n_layers = model.config.n_layer + 1
d_model = model.config.n_embd

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), 
                     desc="Processing samples", unit="sample"):
    prompt = create_prompt(row['text1'], row['text2'])
    
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    layer_acts = []
    
    for h in hidden_states:
        last_token_hidden = h[0, -1, :].cpu().numpy()
        layer_acts.append(last_token_hidden)
    
    layer_acts = np.stack(layer_acts)
    
    results.append({
        'id': int(row['id']),
        'type': row['type'],
        'source1': row['text1_source'],
        'source2': row['text2_source'],
        'activations': layer_acts,
        'n_tokens': inputs['input_ids'].shape[1]
    })

log(f"\n✅ Extracted {len(results)} samples")
log(f"  Activation shape per sample: ({n_layers}, {d_model})")

# 7. 레이블 매칭 및 저장
log("\n[7/7] Matching labels and saving...")

labels = []
for res in tqdm(results, desc="Matching labels", unit="sample"):
    sample_id = res['id']
    matches = annotated_df[annotated_df['instance_id'] == sample_id]
    
    if len(matches) > 0:
        row = matches.iloc[0]
        if pd.notna(row.get('preference:::Text A')):
            labels.append(0)
        elif pd.notna(row.get('preference:::Text B')):
            labels.append(1)
        else:
            labels.append(-1)
    else:
        labels.append(-1)

labels = np.array(labels)
log(f"  Text A preference: {(labels == 0).sum()}")
log(f"  Text B preference: {(labels == 1).sum()}")
log(f"  Unknown: {(labels == -1).sum()}")

# 저장
all_activations = np.stack([r['activations'] for r in results])

np.savez_compressed(
    OUTPUT_PATH / 'demo_activations.npz',
    activations=all_activations,
    labels=labels,
    sample_ids=[r['id'] for r in results]
)

metadata = {
    'model_name': MODEL_NAME,
    'n_samples': len(results),
    'n_layers': n_layers,
    'd_model': d_model,
    'samples': [
        {
            'id': r['id'],
            'type': r['type'],
            'sources': f"{r['source1']} vs {r['source2']}",
            'n_tokens': r['n_tokens'],
            'label': int(labels[i]) if labels[i] >= 0 else None
        }
        for i, r in enumerate(results)
    ]
}

with open(OUTPUT_PATH / 'demo_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

log(f"\n✅ Saved:")
log(f"  - {OUTPUT_PATH / 'demo_activations.npz'}")
log(f"  - {OUTPUT_PATH / 'demo_metadata.json'}")

# 간단한 분석
log("\n" + "="*70)
log("Quick Analysis")
log("="*70)

valid_mask = labels >= 0
if valid_mask.sum() >= 2:
    X = all_activations[valid_mask]
    y = labels[valid_mask]
    
    log(f"\nValid samples: {len(X)}")
    log(f"Activation shape: {X.shape}")
    
    for layer_idx in range(min(3, X.shape[1])):
        layer_mean = X[:, layer_idx, :].mean()
        layer_std = X[:, layer_idx, :].std()
        log(f"  Layer {layer_idx}: mean={layer_mean:.4f}, std={layer_std:.4f}")

log("\n" + "="*70)
log("✅ DEMO COMPLETED SUCCESSFULLY!")
log("="*70)
log(f"\n📁 Output: {OUTPUT_PATH.absolute()}")
log(f"\n📊 Next step: python literary_probe.py")
