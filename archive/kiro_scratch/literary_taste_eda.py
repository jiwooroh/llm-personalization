#!/usr/bin/env python3
"""
Literary Taste Analysis - EDA Script
LiteraryTaste 데이터셋 탐색적 데이터 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from bs4 import BeautifulSoup

# 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
DATA_PATH = Path('LiteraryTaste')

print("=" * 80)
print("Literary Taste Dataset Analysis")
print("=" * 80)

# 1. 데이터 로딩
print("\n[1] 데이터 로딩 중...")
annotated_df = pd.read_csv(DATA_PATH / 'annotated_instances.csv')
preference_df = pd.read_csv(DATA_PATH / 'preference_tasks.csv')

print(f"✅ Annotated instances: {annotated_df.shape}")
print(f"✅ Preference tasks: {preference_df.shape}")
print(f"✅ Unique users: {annotated_df['user'].nunique()}")

# 2. 컬럼 구조 분석
print("\n[2] 설문 구조 분석...")
def parse_column_name(col):
    if ':::' in col:
        question, answer = col.split(':::', 1)
        return question.strip(), answer.strip()
    return col, None

column_info = [parse_column_name(col) for col in annotated_df.columns]
questions = sorted(set([q for q, a in column_info if a is not None]))

print(f"✅ 총 설문 항목: {len(questions)}")
print("\n주요 설문 카테고리 (샘플):")
for i, q in enumerate(questions[:5], 1):
    print(f"  {i}. {q[:80]}...")

# 3. 선호도 분포
print("\n[3] 선호도 응답 분석...")
preference_cols = [col for col in annotated_df.columns if col.startswith('preference:::')]
print(f"✅ 선호도 타입: {[col.split(':::')[1] for col in preference_cols]}")

for col in preference_cols:
    pref_type = col.split(':::')[1]
    count = annotated_df[col].notna().sum()
    print(f"  - {pref_type}: {count} responses")

# 4. 독서 습관 분석
print("\n[4] 독서 습관 통계...")
reading_time_cols = [
    'Approximately, how many hours do you spend reading for pleasure per week?:::text_box',
    'Approximately, how many hours do you spend reading per week (including hours reading for pleasure)?:::text_box',
    'How many books do you read per month?:::text_box'
]

reading_habits = annotated_df[['user'] + reading_time_cols].copy()
reading_habits.columns = ['user', 'pleasure_hours', 'total_hours', 'books_per_month']

# 사용자별 첫 응답만 사용
reading_summary = reading_habits.groupby('user').first()

print("독서 시간 요약 (사용자별):")
print(reading_summary.describe())

# 5. Fiction 장르 선호도
print("\n[5] Fiction 장르 선호도 분석...")
fiction_genre_cols = [col for col in annotated_df.columns 
                      if 'Which of the following types of texts do you enjoy reading these days? (among fiction):::' in col
                      and 'free_response' not in col
                      and 'Others' not in col]

fiction_genres = [col.split(':::')[-1] for col in fiction_genre_cols]

genre_counts = {}
for col, genre in zip(fiction_genre_cols, fiction_genres):
    genre_counts[genre] = annotated_df[col].notna().sum()

sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
print("\n인기 장르 Top 5:")
for i, (genre, count) in enumerate(sorted_genres[:5], 1):
    print(f"  {i}. {genre}: {count} users")

# 6. 텍스트 쌍 선호도 분석
print("\n[6] 텍스트 쌍 선호도 파싱...")

def extract_texts_from_html(html_str):
    """BeautifulSoup으로 HTML에서 텍스트 추출"""
    if pd.isna(html_str):
        return None, None
    
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        
        # Text A 추출
        text_a_div = soup.find('h3', string='Text A')
        text_a = text_a_div.find_next('p').get_text(strip=True) if text_a_div else None
        
        # Text B 추출
        text_b_div = soup.find('h3', string='Text B')
        text_b = text_b_div.find_next('p').get_text(strip=True) if text_b_div else None
        
        return text_a, text_b
    except:
        return None, None

preference_data = []

for _, row in annotated_df.iterrows():
    text_a, text_b = extract_texts_from_html(row['displayed_text'])
    
    if text_a and text_b:
        pref = None
        if pd.notna(row.get('preference:::Text A')):
            pref = 'A'
        elif pd.notna(row.get('preference:::Text B')):
            pref = 'B'
        elif pd.notna(row.get('preference:::I am not sure')):
            pref = 'Unsure'
        
        preference_data.append({
            'user': row['user'],
            'instance_id': row['instance_id'],
            'text_a': text_a,
            'text_b': text_b,
            'preference': pref,
            'text_a_len': len(text_a),
            'text_b_len': len(text_b)
        })

pref_df = pd.DataFrame(preference_data)
print(f"✅ 파싱된 선호도 판단: {len(pref_df)}개")
print("\n선호도 분포:")
print(pref_df['preference'].value_counts())

# 7. 텍스트 길이 통계
print("\n[7] 텍스트 길이 통계...")
print("\nText A 길이:")
print(pref_df['text_a_len'].describe())
print("\nText B 길이:")
print(pref_df['text_b_len'].describe())

# 8. 샘플 텍스트 출력
print("\n[8] 샘플 텍스트 쌍 (첫 번째 항목):")
sample = pref_df.iloc[0]
print(f"\n사용자: {sample['user']}")
print(f"선호: {sample['preference']}")
print(f"\n--- Text A ({sample['text_a_len']} chars) ---")
print(sample['text_a'][:200] + "...")
print(f"\n--- Text B ({sample['text_b_len']} chars) ---")
print(sample['text_b'][:200] + "...")

# 9. preference_tasks.csv 분석
print("\n[9] Preference Tasks 데이터 분석...")
print(f"✅ 컬럼: {list(preference_df.columns[:10])}")
print(f"\nTask 타입 분포:")
if 'type' in preference_df.columns:
    print(preference_df['type'].value_counts())

# 10. 저장
print("\n[10] 분석 결과 저장...")
output_dir = Path('runs/literary_taste')
output_dir.mkdir(parents=True, exist_ok=True)

pref_df.to_parquet(output_dir / 'parsed_preferences.parquet', index=False)
print(f"✅ 저장 완료: {output_dir / 'parsed_preferences.parquet'}")

# Summary JSON
summary = {
    'total_users': int(annotated_df['user'].nunique()),
    'total_instances': len(annotated_df),
    'parsed_preferences': len(pref_df),
    'preference_distribution': pref_df['preference'].value_counts().to_dict(),
    'top_genres': dict(sorted_genres[:5]),
    'avg_text_a_length': float(pref_df['text_a_len'].mean()),
    'avg_text_b_length': float(pref_df['text_b_len'].mean())
}

import json
with open(output_dir / 'eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ 요약 저장: {output_dir / 'eda_summary.json'}")

print("\n" + "=" * 80)
print("✅ EDA 완료!")
print("=" * 80)
print(f"\n다음 단계: TransformerLens로 모델 로드 및 활성화 추출")
print(f"파일 위치: {output_dir.absolute()}")
