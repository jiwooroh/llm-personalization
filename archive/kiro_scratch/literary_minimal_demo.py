#!/usr/bin/env python3
"""
Literary Taste - Minimal Demo
TransformerLens 없이 Hugging Face만으로 빠른 데모
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

print("="*70)
print("Literary Taste - Minimal Activation Demo")
print("="*70)

# 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2"  # 가장 작은 모델
DATA_PATH = Path('LiteraryTaste')
OUTPUT_PATH = Path('runs/literary_taste/demo')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"\n📱 Device: {DEVICE}")
print(f"🤖 Model: {MODEL_NAME}")

# 1. 데이터 로드 (샘플만)
print("\n[1] Loading sample data...")
preference_df = pd.read_csv(DATA_PATH / 'preference_tasks.csv')
annotated_df = pd.read_csv(DATA_PATH / 'annotated_instances.csv', low_memory=False)

# 첫 5개만
N_SAMPLES = 5
sample_df = preference_df.head(N_SAMPLES)
print(f"✅ Loaded {len(sample_df)} samples")

# 2. 모델 로드
print("\n[2] Loading model... (may take 30-60 seconds)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # CPU 호환
    device_map=DEVICE
)
model.eval()

n_layers = model.config.n_layer
d_model = model.config.n_embd

print(f"✅ Model loaded!")
print(f"   - Layers: {n_layers}")
print(f"   - Hidden size: {d_model}")

# 3. 프롬프트 생성
def create_prompt(text1, text2, max_len=100):
    """간단한 비교 프롬프트"""
    t1 = str(text1)[:max_len] if pd.notna(text1) else ""
    t2 = str(text2)[:max_len] if pd.notna(text2) else ""
    return f"Text A: {t1}\nText B: {t2}\nBetter:"

# 4. 활성화 추출
print("\n[3] Extracting activations...")

results = []

for idx, row in sample_df.iterrows():
    prompt = create_prompt(row['text1'], row['text2'])
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # 각 레이어의 마지막 토큰 활성화
    hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_size)
    
    layer_acts = []
    for layer_idx, h in enumerate(hidden_states):
        # 마지막 토큰
        last_token_hidden = h[0, -1, :].cpu().numpy()
        layer_acts.append(last_token_hidden)
    
    # NumPy 배열로 변환
    layer_acts = np.stack(layer_acts)  # shape: (n_layers+1, d_model)
    
    results.append({
        'id': int(row['id']),
        'type': row['type'],
        'source1': row['text1_source'],
        'source2': row['text2_source'],
        'activations': layer_acts,
        'n_tokens': inputs['input_ids'].shape[1]
    })
    
    print(f"   Sample {idx}: {layer_acts.shape}, {inputs['input_ids'].shape[1]} tokens")

print(f"\n✅ Extracted {len(results)} samples")

# 5. 레이블 매칭
print("\n[4] Matching preference labels...")

labels = []
for res in results:
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
print(f"✅ Labels: {labels.tolist()}")
print(f"   - Text A: {(labels == 0).sum()}")
print(f"   - Text B: {(labels == 1).sum()}")
print(f"   - Unknown: {(labels == -1).sum()}")

# 6. 저장
print("\n[5] Saving results...")

# 모든 샘플의 활성화를 하나의 배열로
all_activations = np.stack([r['activations'] for r in results])  # (n_samples, n_layers+1, d_model)

np.savez_compressed(
    OUTPUT_PATH / 'demo_activations.npz',
    activations=all_activations,
    labels=labels,
    sample_ids=[r['id'] for r in results]
)

# 메타데이터
metadata = {
    'model_name': MODEL_NAME,
    'n_samples': len(results),
    'n_layers': n_layers + 1,  # +1 for embedding layer
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

print(f"✅ Saved:")
print(f"   - {OUTPUT_PATH / 'demo_activations.npz'}")
print(f"   - {OUTPUT_PATH / 'demo_metadata.json'}")

# 7. 간단한 분석
print("\n[6] Quick analysis...")

valid_mask = labels >= 0
if valid_mask.sum() >= 2:
    X = all_activations[valid_mask]  # (n_valid, n_layers+1, d_model)
    y = labels[valid_mask]
    
    print(f"\nValid samples: {len(X)}")
    print(f"Activation shape: {X.shape}")
    
    # 각 레이어의 평균 활성화
    for layer_idx in range(min(3, X.shape[1])):  # 처음 3개 레이어만
        layer_mean = X[:, layer_idx, :].mean()
        layer_std = X[:, layer_idx, :].std()
        print(f"  Layer {layer_idx}: mean={layer_mean:.4f}, std={layer_std:.4f}")
    
    # Text A vs Text B 비교
    if len(np.unique(y)) > 1:
        a_mean = X[y == 0].mean(axis=(0, 2))  # average over samples and features
        b_mean = X[y == 1].mean(axis=(0, 2))
        
        print(f"\nActivation magnitude by layer:")
        print(f"  Text A preference: {a_mean[:3]}")
        print(f"  Text B preference: {b_mean[:3]}")
else:
    print("\n⚠️  Not enough labeled samples for analysis")

print("\n" + "="*70)
print("✅ Demo completed!")
print("="*70)
print(f"\n📊 Next: Run literary_probe.py to train classifiers")
print(f"📁 Output: {OUTPUT_PATH.absolute()}")
