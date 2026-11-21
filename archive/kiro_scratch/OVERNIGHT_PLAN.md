# Overnight Research Plan

> 작성: 2026-11-20 02:00 AM
> 예상 완료: 2026-11-20 08:00 AM

## 🎯 목표

1. **현재 실행 완료** 및 결과 분석
2. **논문 기반 연구 계획** 구체화
3. **추가 실험** 실행 (가능한 것들)
4. **문서화 완성**

---

## ✅ Task 1: 현재 실행 완료 (30분)

### 1.1 활성화 추출 완료
- [x] `demo_with_progress.py` 실행 확인
- [ ] 완료 대기 (~2-3분)
- [ ] 출력 파일 확인:
  - `runs/literary_taste/demo/demo_activations.npz`
  - `runs/literary_taste/demo/demo_metadata.json`

### 1.2 Probe 학습
```bash
python literary_probe.py
```
**예상 결과**:
- `probe_results.json`: 레이어별 정확도
- `probe_accuracy_by_layer.png`: 그래프
- `confusion_matrix.png`: 혼동 행렬

### 1.3 시각화
```bash
python visualize_results.py
```
**예상 출력**:
- `1_layer_magnitude.png`: 레이어별 활성화 크기
- `2_pca_layer6.png`: PCA 시각화
- `3_tsne_layer6.png`: t-SNE 클러스터링
- `4_activation_heatmap.png`: 히트맵
- `5_preference_comparison.png`: A vs B 비교

### 1.4 결과 분석
- 어느 레이어가 최고 정확도?
- PCA에서 클러스터 보이는가?
- Text A vs B 차이 명확한가?

---

## 📚 Task 2: 논문 리서치 (2시간)

### 2.1 핵심 논문 읽기

#### Paper 1: Representation Engineering
**주제**: LLM 행동 조정을 위한 표상 조작
**핵심 아이디어**:
- Reading vectors: 특정 개념의 활성화 방향
- Control vectors: 행동 조정을 위한 스티어링
- Honesty, harmlessness 등에 적용

**우리 적용**:
- Genre preference vectors 생성
- User-specific steering 적용
- Literary style control

#### Paper 2: Linear Representation Hypothesis
**주제**: LLM이 선형적으로 정보를 인코딩하는가?
**핵심 아이디어**:
- Linear probes의 높은 성능
- 중간 레이어가 가장 informative
- Task-specific vs general representations

**우리 적용**:
- 선호도가 선형 분리 가능한가?
- 레이어별 선형성 측정
- Task complexity와 layer depth 관계

#### Paper 3: In-Context Learning
**주제**: Few-shot learning의 내부 메커니즘
**핵심 아이디어**:
- Induction heads in middle layers
- Task recognition vs execution
- Context length 효과

**우리 적용**:
- 사용자 프로필을 context로 제공 시 효과
- Few-shot preference learning
- Context engineering

### 2.2 관련 연구 서베이

#### Domain: Personalization
- User modeling in NLP
- Personalized text generation
- Preference learning from comparisons

#### Domain: Interpretability
- Probing classifiers
- Causal mediation analysis
- Attribution methods

#### Domain: Literary Analysis
- Computational literary studies
- Style transfer
- Authorship attribution

### 2.3 Research Gap 식별

**현재 연구들**:
- PRISM: General AI preferences
- Literary studies: Content analysis
- Personalization: Recommendation systems

**우리의 기여**:
1. **새로운 도메인**: 문학 텍스트 선호의 내부 표상
2. **교차 분석**: General AI ↔ Literary preferences
3. **실용적 응용**: Personalized text generation

---

## 🔬 Task 3: 추가 실험 설계 (1시간)

### Experiment 1: Layer-wise Analysis (확장)
**목표**: 각 레이어의 역할 이해

**방법**:
```python
# 1. Ablation study
for layer in range(n_layers):
    # 특정 레이어만 사용
    X = activations[:, layer, :]
    accuracy[layer] = train_probe(X, y)

# 2. Cumulative layers
for depth in range(1, n_layers+1):
    # 0부터 depth까지 concatenate
    X = activations[:, :depth, :].reshape(n, -1)
    accuracy_cumul[depth] = train_probe(X, y)
```

**기대**:
- Single layer: 중간 레이어 최고
- Cumulative: 점진적 증가 후 포화

### Experiment 2: Genre Clustering
**목표**: 장르별 활성화 패턴 차이

**방법**:
```python
# 사용자별 장르 선호 추출
user_genres = extract_user_genres(annotated_df)

# 장르별 활성화 평균
for genre in top_genres:
    users = user_genres[genre]
    genre_acts[genre] = activations[users].mean(axis=0)

# 클러스터링
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(genre_acts)

# 시각화
tsne = TSNE(n_components=2)
genre_2d = tsne.fit_transform(genre_acts)
plt.scatter(genre_2d[:, 0], genre_2d[:, 1], c=clusters)
```

### Experiment 3: Human vs LLM Detection
**목표**: LLM 생성 텍스트 구별

**방법**:
```python
# 레이블: 0=human, 1=llm
is_llm = preference_df['text1_source'].str.startswith('llm_')

# 각 레이어에서 분류
for layer in range(n_layers):
    X = activations[:, layer, :]
    accuracy[layer] = train_probe(X, is_llm)
```

**기대**:
- 후기 레이어에서 높은 정확도
- Style features가 중요

### Experiment 4: Cross-Domain Transfer
**목표**: PRISM ↔ LiteraryTaste 관계

**방법**:
```python
# PRISM probe로 LiteraryTaste 예측
prism_probe = load_probe('runs/qwen7b/probe_best.npz')
literary_acts = load_activations('runs/literary_taste/demo/')

# Transfer accuracy
transfer_acc = prism_probe.score(literary_acts, literary_labels)

# 상관관계 분석
prism_scores = prism_probe.predict_proba(literary_acts)
literary_scores = literary_probe.predict_proba(literary_acts)
correlation = np.corrcoef(prism_scores, literary_scores)
```

### Experiment 5: Attention Pattern Analysis
**목표**: Attention이 선호도와 관련 있는가?

**방법**:
```python
# Attention weights 추출
outputs = model(**inputs, output_attentions=True)
attentions = outputs.attentions  # (n_layers, n_heads, seq_len, seq_len)

# 선호도별 attention 차이
att_A = attentions[labels == 0].mean(axis=0)
att_B = attentions[labels == 1].mean(axis=0)
att_diff = att_A - att_B

# 시각화
plt.imshow(att_diff[layer, head])
```

---

## 🛠️ Task 4: 코드 구현 (2시간)

### 4.1 확장된 분석 스크립트

**파일**: `advanced_analysis.py`
```python
#!/usr/bin/env python3
"""
Advanced Literary Taste Analysis
확장된 실험 및 분석
"""

# 1. Layer ablation
def layer_ablation_study(activations, labels):
    # ...
    
# 2. Genre clustering
def analyze_genre_clusters(activations, user_genres):
    # ...

# 3. Human vs LLM detection
def detect_llm_text(activations, text_sources):
    # ...

# 4. Cross-domain transfer
def cross_domain_analysis(prism_probe, literary_acts):
    # ...

if __name__ == '__main__':
    main()
```

### 4.2 자동화 스크립트

**파일**: `run_all_experiments.sh`
```bash
#!/bin/bash
# 모든 실험 자동 실행

echo "Starting overnight experiments..."

# 1. Basic analysis
python demo_with_progress.py
python literary_probe.py
python visualize_results.py

# 2. Advanced analysis
python advanced_analysis.py --experiment layer_ablation
python advanced_analysis.py --experiment genre_clustering
python advanced_analysis.py --experiment human_vs_llm
python advanced_analysis.py --experiment cross_domain

# 3. Generate report
python generate_report.py

echo "All experiments completed!"
```

---

## 📊 Task 5: 결과 문서화 (1시간)

### 5.1 실험 결과 요약

**파일**: `EXPERIMENT_RESULTS.md`

구조:
```markdown
# Experiment Results

## Experiment 1: Basic Probing
- **Accuracy**: Layer 6 최고 (68.5%)
- **Baseline**: Random (50%)
- **Interpretation**: ...

## Experiment 2: Genre Clustering
- **Clusters**: 5개 발견
- **Separation**: Silhouette score 0.42
- **Top genres per cluster**: ...

...
```

### 5.2 시각화 갤러리

**파일**: `FIGURES.md`

모든 그림 정리:
```markdown
# Visualization Gallery

## 1. Layer-wise Accuracy
![Accuracy by Layer](runs/literary_taste/figures/...)

## 2. PCA Projection
![PCA](runs/literary_taste/figures/...)

...
```

---

## 📝 Task 6: 논문 초고 (1시간)

### 6.1 구조

```markdown
# Understanding User Literary Preferences in LLM Representations

## Abstract
- Research question
- Method
- Key findings
- Implications

## 1. Introduction
- Motivation
- Research gap
- Contributions

## 2. Related Work
- Personalization in NLP
- LLM interpretability
- Literary analysis

## 3. Methodology
- Datasets (PRISM, LiteraryTaste)
- Activation extraction
- Linear probing
- Experiments

## 4. Results
- Layer-wise analysis
- Genre clustering
- Human vs LLM
- Cross-domain

## 5. Discussion
- Interpretation
- Limitations
- Future work

## 6. Conclusion
- Summary
- Broader impact
```

---

## 🔄 Task 7: Mock 데이터 실험 (보험용)

만약 실제 모델 실행이 실패하면:

```bash
# Mock 데이터 생성
python create_mock_data.py

# Mock으로 전체 파이프라인 테스트
python literary_probe.py
python visualize_results.py
python advanced_analysis.py
```

이렇게 하면 **코드 검증은 완료**할 수 있음.

---

## ✅ 체크리스트

### 코드
- [ ] 활성화 추출 완료
- [ ] Probe 학습 완료
- [ ] 기본 시각화 완료
- [ ] Advanced analysis 구현
- [ ] 자동화 스크립트 작성
- [ ] 모든 테스트 통과

### 문서
- [ ] PROJECT_STATUS.md 완성
- [ ] README.md 완성
- [ ] EXPERIMENT_RESULTS.md 작성
- [ ] FIGURES.md 작성
- [ ] 논문 초고 작성
- [ ] 코드 주석 완성

### Git
- [ ] .gitignore 설정
- [ ] README.md 업데이트
- [ ] 커밋 메시지 정리
- [ ] 태그 생성 (v0.1.0)

---

## 🎯 성공 기준

### 최소 목표 (필수)
1. ✅ 활성화 추출 완료
2. ✅ Probe 결과 도출
3. ✅ 시각화 생성
4. ✅ 문서화 완성

### 이상적 목표
1. ✅ 모든 실험 완료
2. ✅ 논문 초고 작성
3. ✅ Git 정리 완료
4. ✅ 추가 분석 완료

---

## 📅 타임라인

| 시간 | Task | 예상 소요 |
|------|------|-----------|
| 02:00 - 02:30 | 현재 실행 완료 | 30분 |
| 02:30 - 04:30 | 논문 리서치 | 2시간 |
| 04:30 - 05:30 | 실험 설계 | 1시간 |
| 05:30 - 07:30 | 코드 구현 | 2시간 |
| 07:30 - 08:30 | 문서화 | 1시간 |
| 08:30 - 09:00 | Git 정리 | 30분 |

**총 예상 시간**: 7시간

---

## 💾 백업 계획

### 중간 저장
- 매 1시간마다 진행 상황 커밋
- 결과 파일 자동 백업
- 로그 파일 유지

### 실패 대응
1. **모델 실행 실패**: Mock 데이터 사용
2. **시간 부족**: 우선순위 task만
3. **오류 발생**: 로그 저장 후 다음 task

---

**시작 시간**: 2026-11-20 02:00 AM
**예상 완료**: 2026-11-20 09:00 AM
**상태**: 진행 중 🚀
