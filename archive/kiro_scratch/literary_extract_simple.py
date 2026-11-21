#!/usr/bin/env python3
"""
Literary Taste - Simple Activation Extraction
작은 샘플로 빠르게 테스트
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

print("="*70)
print("Literary Taste - Quick Activation Extraction")
print("="*70)

# 1. 환경 체크
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n📱 Device: {DEVICE}")
print(f"🔧 PyTorch version: {torch.__version__}")

# 2. 데이터 로드 (소량만)
print("\n[1] Loading sample data...")
DATA_PATH = Path('LiteraryTaste')
preference_df = pd.read_csv(DATA_PATH / 'preference_tasks.csv')

# 샘플 10개만 사용
N_SAMPLES = 10
sample_df = preference_df.head(N_SAMPLES)
print(f"✅ Loaded {len(sample_df)} samples")

# 3. TransformerLens 임포트 및 모델 로드
print("\n[2] Loading TransformerLens model...")
print("   (This may take 1-2 minutes...)")

try:
    from transformer_lens import HookedTransformer
    
    # 가장 작은 모델 사용
    MODEL_NAME = "gpt2-small"
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        cache_dir="./cache"
    )
    print(f"✅ Model loaded successfully!")
    print(f"   - Layers: {model.cfg.n_layers}")
    print(f"   - Hidden size: {model.cfg.d_model}")
    print(f"   - Vocab size: {model.cfg.d_vocab}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\n💡 Tip: Model loading takes time. Try running with longer timeout.")
    exit(1)

# 4. 프롬프트 생성 함수
def create_simple_prompt(text_a, text_b, max_len=150):
    """간단한 비교 프롬프트"""
    text_a = text_a[:max_len] if pd.notna(text_a) else ""
    text_b = text_b[:max_len] if pd.notna(text_b) else ""
    
    return f"Text A: {text_a}\n\nText B: {text_b}\n\nPrefer:"

# 5. 활성화 추출
print("\n[3] Extracting activations...")
OUTPUT_PATH = Path('runs/literary_taste/activations')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

results = []

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing"):
    prompt = create_simple_prompt(row['text1'], row['text2'])
    
    try:
        # Tokenize
        tokens = model.to_tokens(prompt)
        
        # Run with cache
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
        
        # 각 레이어의 마지막 토큰 활성화 추출
        layer_acts = {}
        for layer_idx in range(model.cfg.n_layers):
            act = cache[f'blocks.{layer_idx}.hook_resid_post']
            final_act = act[0, -1, :].cpu().numpy()
            layer_acts[f'layer_{layer_idx}'] = final_act
        
        # 결과 저장
        results.append({
            'id': int(row['id']),
            'type': row['type'],
            'text1_source': row['text1_source'],
            'text2_source': row['text2_source'],
            'activations': layer_acts,
            'n_tokens': int(tokens.shape[1])
        })
        
    except Exception as e:
        print(f"\n⚠️  Error on sample {idx}: {e}")
        continue

print(f"\n✅ Extracted activations for {len(results)} samples")

# 6. 저장
print("\n[4] Saving results...")

# 활성화를 별도 파일로 저장
activations_dict = {}
for i, res in enumerate(results):
    for layer_name, act in res['activations'].items():
        if layer_name not in activations_dict:
            activations_dict[layer_name] = []
        activations_dict[layer_name].append(act)

# NumPy 배열로 변환 및 저장
for layer_name, acts in activations_dict.items():
    acts_array = np.stack(acts)  # shape: (n_samples, d_model)
    np.save(OUTPUT_PATH / f'{layer_name}_sample.npy', acts_array)
    print(f"  ✅ {layer_name}: shape {acts_array.shape}")

# 메타데이터 저장
metadata = {
    'model_name': MODEL_NAME,
    'n_samples': len(results),
    'n_layers': model.cfg.n_layers,
    'd_model': model.cfg.d_model,
    'samples': [
        {
            'id': r['id'],
            'type': r['type'],
            'sources': f"{r['text1_source']} vs {r['text2_source']}",
            'n_tokens': r['n_tokens']
        }
        for r in results
    ]
}

with open(OUTPUT_PATH / 'sample_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Metadata saved: {OUTPUT_PATH / 'sample_metadata.json'}")

# 7. 간단한 분석
print("\n[5] Quick analysis...")
layer_0 = np.load(OUTPUT_PATH / 'layer_0_sample.npy')
layer_last = np.load(OUTPUT_PATH / f'layer_{model.cfg.n_layers-1}_sample.npy')

print(f"\n  Layer 0 statistics:")
print(f"    - Mean: {layer_0.mean():.4f}")
print(f"    - Std: {layer_0.std():.4f}")
print(f"    - Range: [{layer_0.min():.4f}, {layer_0.max():.4f}]")

print(f"\n  Layer {model.cfg.n_layers-1} statistics:")
print(f"    - Mean: {layer_last.mean():.4f}")
print(f"    - Std: {layer_last.std():.4f}")
print(f"    - Range: [{layer_last.min():.4f}, {layer_last.max():.4f}]")

print("\n" + "="*70)
print("✅ Activation extraction completed!")
print("="*70)
print(f"\n📁 Output: {OUTPUT_PATH.absolute()}")
print(f"\n📊 Next: Train probes to predict preferences from activations")
