#!/usr/bin/env python3
"""
Literary Taste - Activation Extraction
TransformerLens를 활용한 활성화 추출 예제
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer
import json

print("=" * 70)
print("Literary Taste - Activation Extraction")
print("=" * 70)

# 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2-small"  # 또는 "pythia-70m"
DATA_PATH = Path('LiteraryTaste')
OUTPUT_PATH = Path('runs/literary_taste/activations')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"\n📱 Device: {DEVICE}")
print(f"🤖 Model: {MODEL_NAME}")

# 1. 모델 로드
print("\n[1] Loading model...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
print(f"✅ Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")

# 2. 데이터 로드
print("\n[2] Loading data...")
preference_df = pd.read_csv(DATA_PATH / 'preference_tasks.csv')
print(f"✅ Loaded {len(preference_df)} text pairs")

# 3. 샘플 프롬프트 생성
def create_prompt(text_a, text_b):
    """텍스트 쌍을 비교하는 프롬프트 생성"""
    return f"""Compare these two texts and decide which one is better written:

Text A: {text_a}

Text B: {text_b}

Which text do you prefer? (A or B):"""

# 4. 활성화 추출 함수
def extract_activations(prompt, layer_idx=-1):
    """특정 레이어의 활성화 추출"""
    tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    # Residual stream activations
    activations = cache[f'blocks.{layer_idx}.hook_resid_post']
    
    # 마지막 토큰의 활성화만 사용 (선호도 예측에 사용)
    final_activation = activations[0, -1, :].cpu().numpy()
    
    return final_activation

# 5. 샘플 데이터로 테스트
print("\n[3] Extracting activations from sample...")
sample = preference_df.iloc[0]
prompt = create_prompt(sample['text1'][:200], sample['text2'][:200])  # 길이 제한

print(f"\nPrompt (첫 100자):\n{prompt[:100]}...")

# 각 레이어에서 활성화 추출
layer_activations = {}
for layer_idx in range(model.cfg.n_layers):
    act = extract_activations(prompt, layer_idx)
    layer_activations[f'layer_{layer_idx}'] = act
    print(f"  Layer {layer_idx}: shape {act.shape}, mean {act.mean():.4f}")

# 6. 결과 저장
print("\n[4] Saving sample activation...")
np.savez_compressed(
    OUTPUT_PATH / 'sample_activation.npz',
    **layer_activations,
    prompt=prompt
)
print(f"✅ Saved to: {OUTPUT_PATH / 'sample_activation.npz'}")

# 7. 모델 정보 저장
model_info = {
    'model_name': MODEL_NAME,
    'n_layers': model.cfg.n_layers,
    'n_heads': model.cfg.n_heads,
    'd_model': model.cfg.d_model,
    'd_head': model.cfg.d_head,
    'n_ctx': model.cfg.n_ctx,
    'vocab_size': model.cfg.d_vocab
}

with open(OUTPUT_PATH / 'model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"✅ Model info saved: {OUTPUT_PATH / 'model_info.json'}")

# 8. 다음 단계 안내
print("\n" + "=" * 70)
print("✅ Activation extraction test completed!")
print("=" * 70)
print("\n📋 Next steps:")
print("  1. Extract activations for all text pairs")
print("  2. Label with user preferences (A/B)")
print("  3. Train linear probes to predict preferences")
print("  4. Analyze layer-wise accuracy")
print("  5. Identify steering directions")
print(f"\n📁 Output directory: {OUTPUT_PATH.absolute()}")
