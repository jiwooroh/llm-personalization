#!/usr/bin/env python3
"""
PRISM vs LiteraryTaste 데이터셋 비교
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*70)
print("Dataset Comparison: PRISM vs LiteraryTaste")
print("="*70)

# 1. PRISM 데이터 로드
print("\n[1] Loading PRISM data...")
prism_path = Path('runs/qwen7b')

if (prism_path / 'users.parquet').exists():
    prism_users = pd.read_parquet(prism_path / 'users.parquet')
    print(f"✅ PRISM users: {len(prism_users)}")
    print(f"   Attributes: {prism_users.columns.tolist()[:10]}")
else:
    print("⚠️  PRISM data not found")
    prism_users = None

# 2. LiteraryTaste 데이터 로드
print("\n[2] Loading LiteraryTaste data...")
lit_path = Path('LiteraryTaste')
lit_annotated = pd.read_csv(lit_path / 'annotated_instances.csv', low_memory=False)
lit_preference = pd.read_csv(lit_path / 'preference_tasks.csv')

print(f"✅ LiteraryTaste:")
print(f"   Users: {lit_annotated['user'].nunique()}")
print(f"   Instances: {len(lit_annotated)}")
print(f"   Text pairs: {len(lit_preference)}")

# 3. 데이터 특성 비교
print("\n[3] Dataset Characteristics Comparison")
print("="*70)

comparison = {
    'Dataset': ['PRISM', 'LiteraryTaste'],
    'Domain': ['General AI preferences', 'Literary text preferences'],
    'Users': [
        len(prism_users) if prism_users is not None else 'N/A',
        lit_annotated['user'].nunique()
    ],
    'Instances': [
        len(prism_users) if prism_users is not None else 'N/A',
        len(lit_annotated)
    ],
    'Attributes': [
        'values, fluency, factuality, safety, etc.',
        'genre, reading habits, leisure activities'
    ],
    'Task Type': [
        'Rating attributes (0-100)',
        'Binary preference (A vs B)'
    ],
    'Text Source': [
        'User survey responses',
        'Gutenberg, Reddit, Poetry'
    ]
}

comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))

# 4. 사용자 프로필 복잡도
print("\n[4] User Profile Complexity")
print("="*70)

# PRISM attributes
if prism_users is not None:
    prism_attrs = [col for col in prism_users.columns if not col.startswith('rel_') and col not in ['user_id', 'fold', 'text']]
    print(f"PRISM attributes: {len(prism_attrs)}")
    print(f"  {prism_attrs}")

# LiteraryTaste survey items
lit_questions = []
for col in lit_annotated.columns:
    if ':::' in col and not col.startswith('preference:::'):
        question = col.split(':::')[0]
        if question not in lit_questions:
            lit_questions.append(question)

print(f"\nLiteraryTaste survey items: {len(lit_questions)}")
print(f"  Samples: {lit_questions[:3]}")

# 5. 데이터 밀도
print("\n[5] Data Density")
print("="*70)

if prism_users is not None:
    prism_attrs_per_user = len(prism_attrs)
    print(f"PRISM: {prism_attrs_per_user} attributes per user")

lit_instances_per_user = len(lit_annotated) / lit_annotated['user'].nunique()
print(f"LiteraryTaste: {lit_instances_per_user:.1f} judgments per user")

# 6. Task 복잡도
print("\n[6] Task Complexity")
print("="*70)

print("PRISM:")
print("  - Rating task: Assign 0-100 score")
print("  - Continuous values")
print("  - 8 dimensions")

print("\nLiteraryTaste:")
print("  - Binary choice: A or B")
print("  - With 'unsure' option")
print("  - Context: full text passages (~700 chars)")

# 7. 연구 질문 매핑
print("\n[7] Research Questions Mapping")
print("="*70)

questions = {
    'RQ': [
        'User representation',
        'Prediction accuracy',
        'Layer analysis',
        'Steering effectiveness'
    ],
    'PRISM': [
        '✅ Attribute vectors in activations',
        '✅ Spearman correlation',
        '✅ Middle layers best',
        '🚧 Minimal pairs'
    ],
    'LiteraryTaste': [
        '🔜 Preference patterns in activations',
        '🔜 Binary classification accuracy',
        '🔜 Layer-wise probe accuracy',
        '🔜 Genre-specific steering'
    ]
}

rq_df = pd.DataFrame(questions)
print(rq_df.to_string(index=False))

# 8. 통합 가능성
print("\n[8] Integration Opportunities")
print("="*70)

print("✅ Shared methodology:")
print("  1. Activation extraction (same approach)")
print("  2. Linear probing (same technique)")
print("  3. Layer-wise analysis (same framework)")
print("  4. Steering vectors (same concept)")

print("\n✅ Complementary insights:")
print("  - PRISM: General AI alignment preferences")
print("  - LiteraryTaste: Domain-specific (literary) preferences")
print("  - Combined: Multi-domain personalization")

print("\n🔮 Future work:")
print("  - Cross-domain transfer learning")
print("  - Unified user representation")
print("  - Multi-task probing")

# 9. 저장
print("\n[9] Saving comparison...")
output_path = Path('runs/literary_taste')
output_path.mkdir(parents=True, exist_ok=True)

summary = {
    'prism': {
        'users': len(prism_users) if prism_users is not None else 0,
        'attributes': len(prism_attrs) if prism_users is not None else 0,
        'task_type': 'rating'
    },
    'literary_taste': {
        'users': int(lit_annotated['user'].nunique()),
        'instances': len(lit_annotated),
        'text_pairs': len(lit_preference),
        'task_type': 'binary_preference'
    },
    'comparison': {
        'domain': {
            'prism': 'general_ai',
            'literary': 'text_preference'
        },
        'task_complexity': {
            'prism': 'continuous_rating',
            'literary': 'binary_choice'
        },
        'shared_methodology': [
            'activation_extraction',
            'linear_probing',
            'layer_analysis',
            'steering'
        ]
    }
}

with open(output_path / 'dataset_comparison.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Saved: {output_path / 'dataset_comparison.json'}")

print("\n" + "="*70)
print("✅ Comparison completed!")
print("="*70)
print("\n💡 Key Takeaway:")
print("   Both datasets can use the same analysis pipeline!")
print("   LiteraryTaste extends PRISM methodology to literary domain.")
