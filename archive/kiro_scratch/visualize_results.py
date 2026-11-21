#!/usr/bin/env python3
"""
Literary Taste - Results Visualization
분석 결과 시각화 (활성화 추출 완료 후 실행)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

print("="*70)
print("Literary Taste - Results Visualization")
print("="*70)

# 경로 설정
DEMO_PATH = Path('runs/literary_taste/demo')
OUTPUT_PATH = Path('runs/literary_taste/figures')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 1. 데이터 로드
print("\n[1] Loading data...")

if not (DEMO_PATH / 'demo_activations.npz').exists():
    print("❌ Activations not found. Please run literary_minimal_demo.py first.")
    exit(1)

data = np.load(DEMO_PATH / 'demo_activations.npz')
activations = data['activations']  # (n_samples, n_layers+1, d_model)
labels = data['labels']  # (n_samples,)
sample_ids = data['sample_ids']

with open(DEMO_PATH / 'demo_metadata.json') as f:
    metadata = json.load(f)

n_samples, n_layers, d_model = activations.shape

print(f"✅ Loaded:")
print(f"   Samples: {n_samples}")
print(f"   Layers: {n_layers}")
print(f"   Hidden size: {d_model}")
print(f"   Labels: {labels.tolist()}")

valid_mask = labels >= 0
n_valid = valid_mask.sum()
print(f"\n   Valid labels: {n_valid}/{n_samples}")

if n_valid < 2:
    print("⚠️  Not enough labeled samples for visualization")
    exit(0)

# 2. 레이어별 활성화 평균 magnitude
print("\n[2] Creating layer-wise activation magnitude plot...")

layer_means = []
layer_stds = []

for layer_idx in range(n_layers):
    acts = activations[:, layer_idx, :]
    layer_means.append(np.abs(acts).mean())
    layer_stds.append(acts.std())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Magnitude
axes[0].plot(range(n_layers), layer_means, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Layer', fontsize=12)
axes[0].set_ylabel('Mean Absolute Activation', fontsize=12)
axes[0].set_title('Activation Magnitude by Layer', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# Std
axes[1].plot(range(n_layers), layer_stds, 's-', color='orange', linewidth=2, markersize=8)
axes[1].set_xlabel('Layer', fontsize=12)
axes[1].set_ylabel('Activation Std', fontsize=12)
axes[1].set_title('Activation Variability by Layer', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig_path = OUTPUT_PATH / '1_layer_magnitude.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {fig_path}")
plt.close()

# 3. PCA 시각화 (best layer)
if n_valid >= 3:
    print("\n[3] Creating PCA visualization...")
    
    # 중간 레이어 선택 (보통 가장 informative)
    best_layer = n_layers // 2
    
    X = activations[valid_mask, best_layer, :]
    y = labels[valid_mask]
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    
    # Plot by label
    for label_val in np.unique(y):
        mask = y == label_val
        label_name = 'Text A' if label_val == 0 else 'Text B'
        plt.scatter(
            X_pca[mask, 0], 
            X_pca[mask, 1],
            label=label_name,
            alpha=0.7,
            s=100,
            edgecolors='black'
        )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    plt.title(f'PCA of Activations (Layer {best_layer})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    fig_path = OUTPUT_PATH / f'2_pca_layer{best_layer}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {fig_path}")
    plt.close()

# 4. t-SNE 시각화 (충분한 데이터가 있을 경우)
if n_valid >= 5:
    print("\n[4] Creating t-SNE visualization...")
    
    # t-SNE는 perplexity < n_samples 필요
    perplexity = min(3, n_valid - 1)
    
    if perplexity >= 2:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        
        for label_val in np.unique(y):
            mask = y == label_val
            label_name = 'Text A' if label_val == 0 else 'Text B'
            plt.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                label=label_name,
                alpha=0.7,
                s=100,
                edgecolors='black'
            )
        
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.title(f't-SNE of Activations (Layer {best_layer})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        
        fig_path = OUTPUT_PATH / f'3_tsne_layer{best_layer}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {fig_path}")
        plt.close()

# 5. 레이어별 활성화 히트맵 (샘플)
print("\n[5] Creating activation heatmap...")

# 각 레이어의 첫 50개 뉴런만 (시각화 목적)
n_neurons = min(50, d_model)
sample_idx = 0  # 첫 번째 샘플

sample_acts = activations[sample_idx, :, :n_neurons]  # (n_layers, n_neurons)

plt.figure(figsize=(12, 8))
sns.heatmap(
    sample_acts.T,
    cmap='RdBu_r',
    center=0,
    cbar_kws={'label': 'Activation'},
    yticklabels=50
)
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Neuron', fontsize=12)
plt.title(f'Activation Pattern (Sample {sample_idx})', fontsize=14, fontweight='bold')

fig_path = OUTPUT_PATH / '4_activation_heatmap.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {fig_path}")
plt.close()

# 6. Text A vs Text B 비교 (if enough data)
if n_valid >= 4 and len(np.unique(y)) > 1:
    print("\n[6] Creating Text A vs Text B comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Layer-wise mean for each preference
    a_means = []
    b_means = []
    
    for layer_idx in range(n_layers):
        acts_a = activations[valid_mask & (labels == 0), layer_idx, :].mean()
        acts_b = activations[valid_mask & (labels == 1), layer_idx, :].mean()
        a_means.append(acts_a)
        b_means.append(acts_b)
    
    # Plot
    axes[0].plot(range(n_layers), a_means, 'o-', label='Text A preference', linewidth=2)
    axes[0].plot(range(n_layers), b_means, 's-', label='Text B preference', linewidth=2)
    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('Mean Activation', fontsize=12)
    axes[0].set_title('Activation by Preference', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Difference
    diff = np.array(a_means) - np.array(b_means)
    axes[1].bar(range(n_layers), diff, alpha=0.7)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('Activation Difference (A - B)', fontsize=12)
    axes[1].set_title('Layer-wise Preference Signal', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = OUTPUT_PATH / '5_preference_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {fig_path}")
    plt.close()

# 7. Summary
print("\n[7] Generating summary...")

summary = {
    'visualization_date': '2026-11-20',
    'n_samples': int(n_samples),
    'n_valid': int(n_valid),
    'n_layers': int(n_layers),
    'd_model': int(d_model),
    'layer_statistics': {
        'mean_magnitude': [float(m) for m in layer_means],
        'std': [float(s) for s in layer_stds]
    },
    'figures_created': [
        '1_layer_magnitude.png',
        '2_pca_layer{}.png'.format(n_layers // 2),
        '3_tsne_layer{}.png'.format(n_layers // 2) if n_valid >= 5 else None,
        '4_activation_heatmap.png',
        '5_preference_comparison.png' if n_valid >= 4 else None
    ]
}

with open(OUTPUT_PATH / 'visualization_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Saved: {OUTPUT_PATH / 'visualization_summary.json'}")

print("\n" + "="*70)
print("✅ Visualization completed!")
print("="*70)
print(f"\n📁 Output directory: {OUTPUT_PATH.absolute()}")
print(f"📊 Created {len([f for f in summary['figures_created'] if f])} figures")
