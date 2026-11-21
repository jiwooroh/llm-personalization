#!/usr/bin/env python3
"""
Mock 활성화 데이터 생성
transformers 없이 분석 파이프라인을 완성하기 위한 가짜 데이터
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

print("="*70)
print("Creating Mock Activation Data")
print("="*70)

# 설정
N_SAMPLES = 50  # 샘플 수
N_LAYERS = 13   # GPT-2 layers (12 + embedding)
D_MODEL = 768   # GPT-2 hidden size
OUTPUT_PATH = Path('runs/literary_taste/demo')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Samples: {N_SAMPLES}")
print(f"  Layers: {N_LAYERS}")
print(f"  Hidden size: {D_MODEL}")

# 1. 실제 데이터에서 샘플 ID 가져오기
print("\n[1] Loading real text pairs...")
DATA_PATH = Path('LiteraryTaste')
preference_df = pd.read_csv(DATA_PATH / 'preference_tasks.csv')
annotated_df = pd.read_csv(DATA_PATH / 'annotated_instances.csv', low_memory=False)

sample_df = preference_df.head(N_SAMPLES)
sample_ids = sample_df['id'].tolist()
print(f"✅ Selected {len(sample_ids)} real text pairs")

# 2. Mock 활성화 생성 (현실적인 분포로)
print("\n[2] Generating mock activations...")

# 실제 GPT-2 활성화와 비슷한 분포로 생성
# - 초기 레이어: 작은 값
# - 중간 레이어: 큰 값
# - 후기 레이어: 중간 값

activations = np.zeros((N_SAMPLES, N_LAYERS, D_MODEL), dtype=np.float32)

for layer_idx in range(N_LAYERS):
    # 레이어마다 다른 분포
    if layer_idx < 3:
        # 초기 레이어: 작은 활성화
        scale = 0.5
    elif layer_idx < 9:
        # 중간 레이어: 큰 활성화
        scale = 1.5
    else:
        # 후기 레이어: 중간 활성화
        scale = 1.0
    
    # 정규분포로 생성
    activations[:, layer_idx, :] = np.random.randn(N_SAMPLES, D_MODEL) * scale

print(f"✅ Generated activations: shape {activations.shape}")
print(f"   Mean: {activations.mean():.4f}")
print(f"   Std: {activations.std():.4f}")

# 3. 레이블 생성 (실제 데이터에서)
print("\n[3] Matching real preference labels...")

labels = []
for sample_id in sample_ids:
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
        # 없으면 랜덤
        labels.append(np.random.choice([0, 1]))

labels = np.array(labels)

print(f"✅ Labels:")
print(f"   Text A: {(labels == 0).sum()}")
print(f"   Text B: {(labels == 1).sum()}")
print(f"   Unknown: {(labels == -1).sum()}")

# 4. 선호도에 따라 활성화에 약간의 신호 주입
# (probe가 학습할 수 있도록)
print("\n[4] Injecting preference signal...")

for i in range(N_SAMPLES):
    if labels[i] == 0:  # Text A
        # 중간 레이어에 약간의 음수 bias
        activations[i, 5:8, :100] -= 0.3
    elif labels[i] == 1:  # Text B
        # 중간 레이어에 약간의 양수 bias
        activations[i, 5:8, :100] += 0.3

print("✅ Signal injected (subtle bias in layers 5-8)")

# 5. 저장
print("\n[5] Saving data...")

np.savez_compressed(
    OUTPUT_PATH / 'demo_activations.npz',
    activations=activations,
    labels=labels,
    sample_ids=sample_ids
)

metadata = {
    'model_name': 'mock-gpt2',
    'note': 'Mock data for testing pipeline',
    'n_samples': int(N_SAMPLES),
    'n_layers': int(N_LAYERS),
    'd_model': int(D_MODEL),
    'samples': [
        {
            'id': int(sid),
            'label': int(labels[i]) if labels[i] >= 0 else None
        }
        for i, sid in enumerate(sample_ids)
    ]
}

with open(OUTPUT_PATH / 'demo_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved:")
print(f"   - {OUTPUT_PATH / 'demo_activations.npz'}")
print(f"   - {OUTPUT_PATH / 'demo_metadata.json'}")

# 6. 요약
print("\n" + "="*70)
print("✅ Mock data created successfully!")
print("="*70)
print(f"\n📊 Data summary:")
print(f"   Shape: {activations.shape}")
print(f"   Valid labels: {(labels >= 0).sum()}/{N_SAMPLES}")
print(f"   Output: {OUTPUT_PATH.absolute()}")
print(f"\n🚀 Next step:")
print(f"   python literary_probe.py")
