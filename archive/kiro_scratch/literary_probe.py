#!/usr/bin/env python3
"""
Literary Taste - Probe Training
활성화에서 사용자 선호도 예측하는 Linear Probe 학습
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

print("="*70)
print("Literary Taste - Probe Training")
print("="*70)

# 1. 데이터 로드
ACT_PATH = Path('runs/literary_taste/demo')
DATA_PATH = Path('LiteraryTaste')

print("\n[1] Loading activations and labels...")

# 메타데이터 로드
with open(ACT_PATH / 'sample_metadata.json') as f:
    metadata = json.load(f)

n_layers = metadata['n_layers']
d_model = metadata['d_model']
sample_ids = [s['id'] for s in metadata['samples']]

print(f"✅ Metadata loaded:")
print(f"   - Samples: {len(sample_ids)}")
print(f"   - Layers: {n_layers}")
print(f"   - Hidden dim: {d_model}")

# 활성화 로드 (모든 레이어)
activations = {}
for layer_idx in range(n_layers):
    layer_file = ACT_PATH / f'layer_{layer_idx}_sample.npy'
    if layer_file.exists():
        activations[layer_idx] = np.load(layer_file)
        print(f"   ✅ Layer {layer_idx}: {activations[layer_idx].shape}")
    else:
        print(f"   ⚠️  Layer {layer_idx}: not found")

# 2. 레이블 로드 (annotated_instances에서)
print("\n[2] Loading preference labels...")
annotated_df = pd.read_csv(DATA_PATH / 'annotated_instances.csv', low_memory=False)

# 샘플 ID와 매칭
labels = []
for sample_id in sample_ids:
    # instance_id와 매칭
    matches = annotated_df[annotated_df['instance_id'] == sample_id]
    
    if len(matches) > 0:
        row = matches.iloc[0]
        # 선호도 추출
        if pd.notna(row.get('preference:::Text A')):
            labels.append(0)  # Text A
        elif pd.notna(row.get('preference:::Text B')):
            labels.append(1)  # Text B
        else:
            labels.append(-1)  # Unknown
    else:
        labels.append(-1)

labels = np.array(labels)
valid_mask = labels >= 0
n_valid = valid_mask.sum()

print(f"✅ Labels loaded:")
print(f"   - Valid labels: {n_valid}/{len(labels)}")
print(f"   - Text A: {(labels == 0).sum()}")
print(f"   - Text B: {(labels == 1).sum()}")

if n_valid < 5:
    print("\n⚠️  Not enough labeled samples for training!")
    print("💡 This is expected with small sample. Run with more data.")
    exit(0)

# 3. Probe 학습 (각 레이어)
print("\n[3] Training probes for each layer...")

results = []
best_layer = None
best_acc = 0

for layer_idx in range(n_layers):
    if layer_idx not in activations:
        continue
    
    X = activations[layer_idx][valid_mask]
    y = labels[valid_mask]
    
    if len(X) < 5:
        print(f"   Layer {layer_idx}: skipped (not enough data)")
        continue
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Logistic Regression probe
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)
    
    # Evaluate
    train_acc = probe.score(X_train, y_train)
    test_acc = probe.score(X_test, y_test)
    
    # Cross-validation (if enough data)
    if len(X) >= 10:
        cv_scores = cross_val_score(probe, X, y, cv=min(3, len(X)//2))
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    else:
        cv_mean = test_acc
        cv_std = 0
    
    results.append({
        'layer': layer_idx,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    })
    
    print(f"   Layer {layer_idx:2d}: Train {train_acc:.3f} | Test {test_acc:.3f} | CV {cv_mean:.3f}±{cv_std:.3f}")
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_layer = layer_idx

print(f"\n✅ Best layer: {best_layer} (accuracy: {best_acc:.3f})")

# 4. 결과 시각화
print("\n[4] Creating visualizations...")

results_df = pd.DataFrame(results)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy by layer
axes[0].plot(results_df['layer'], results_df['train_acc'], 'o-', label='Train', alpha=0.7)
axes[0].plot(results_df['layer'], results_df['test_acc'], 's-', label='Test', alpha=0.7)
axes[0].plot(results_df['layer'], results_df['cv_mean'], '^-', label='CV', alpha=0.7)
axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Random')
axes[0].set_xlabel('Layer', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Probe Accuracy by Layer', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# CV scores with error bars
axes[1].errorbar(
    results_df['layer'], 
    results_df['cv_mean'], 
    yerr=results_df['cv_std'],
    fmt='o-', 
    capsize=5, 
    alpha=0.7
)
axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Random')
axes[1].set_xlabel('Layer', fontsize=12)
axes[1].set_ylabel('CV Accuracy', fontsize=12)
axes[1].set_title('Cross-Validation Accuracy', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig_path = ACT_PATH / 'probe_accuracy_by_layer.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {fig_path}")
plt.close()

# 5. Best probe 상세 분석
print(f"\n[5] Detailed analysis for best layer ({best_layer})...")

X = activations[best_layer][valid_mask]
y = labels[valid_mask]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

best_probe = LogisticRegression(max_iter=1000, random_state=42)
best_probe.fit(X_train, y_train)
y_pred = best_probe.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Text A', 'Text B']))

# Confusion matrix
if len(y_test) >= 2:
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Text A', 'Text B'],
                yticklabels=['Text A', 'Text B'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix - Layer {best_layer}', fontsize=14, fontweight='bold')
    
    cm_path = ACT_PATH / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {cm_path}")
    plt.close()

# 6. 결과 저장
print("\n[6] Saving results...")

summary = {
    'n_samples': int(n_valid),
    'n_layers': n_layers,
    'best_layer': int(best_layer),
    'best_accuracy': float(best_acc),
    'layer_results': results
}

with open(ACT_PATH / 'probe_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Saved: {ACT_PATH / 'probe_results.json'}")

print("\n" + "="*70)
print("✅ Probe training completed!")
print("="*70)
print(f"\n📊 Summary:")
print(f"   - Best layer: {best_layer}")
print(f"   - Best accuracy: {best_acc:.1%}")
print(f"   - Random baseline: 50%")
print(f"\n📁 Output: {ACT_PATH.absolute()}")
