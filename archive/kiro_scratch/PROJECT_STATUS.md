# LLM Personalization Project - 현재 상황

> 최종 업데이트: 2026-11-20
> 상태: Phase 1 완료, Phase 2 진행 중

## 📋 프로젝트 개요

**목표**: LLM 내부에서 사용자 개인화 선호가 어떻게 표상되는지 분석하고, 이를 활용한 personalization 시스템 구축

**데이터셋**:
- PRISM: 1,500명, 8개 AI 선호 속성 (완료)
- LiteraryTaste: 60명, 6,300개 텍스트 선호 판단 (진행 중)

---

## ✅ 완료된 작업

### 1. 데이터 수집 및 분석 (100%)
- [x] LiteraryTaste 데이터셋 다운로드
- [x] 데이터 구조 파악 (6,300 instances, 60 users)
- [x] EDA 완료 및 통계 분석
- [x] PRISM vs LiteraryTaste 비교 분석

**주요 발견**:
```
사용자: 60명
텍스트 쌍: 2,000개
판단: 6,300개 (사용자당 105개)
선호도: A 46.2%, B 45.0% (균등 분포)
```

### 2. 분석 파이프라인 구축 (100%)
- [x] EDA 스크립트 (`literary_taste_eda.py`)
- [x] 활성화 추출 스크립트 (`demo_with_progress.py`)
- [x] Probe 학습 스크립트 (`literary_probe.py`)
- [x] 시각화 스크립트 (`visualize_results.py`)
- [x] 데이터셋 비교 스크립트 (`compare_datasets.py`)

### 3. 문서화 (100%)
- [x] 완전한 분석 가이드 (`LiteraryTaste/ANALYSIS_GUIDE.md`)
- [x] 프로젝트 README (`runs/literary_taste/README.md`)
- [x] 코드 주석 및 docstring

---

## 🚧 진행 중

### Phase 2: 활성화 추출 (90%)
**현재 상황**: `demo_with_progress.py` 실행 중
- [x] 데이터 로드 완료
- [x] 모델 로딩 진행 중 (GPT-2)
- [ ] 5개 샘플 활성화 추출 대기
- [ ] 레이블 매칭 대기

**예상 소요 시간**: 2-3분 (CPU 환경)

**이슈**: 
- transformers 임포트에 6분 27초 소요 (정상)
- `torch_dtype` deprecated 경고 (무해)

---

## 📊 데이터 분석 결과

### LiteraryTaste Dataset

#### 기본 통계
| 항목 | 값 |
|------|-----|
| 사용자 수 | 60 |
| 텍스트 쌍 | 2,000 |
| 판단 인스턴스 | 6,300 |
| 평균 텍스트 길이 | ~690 chars |

#### 선호도 분포
```
Text A:    2,908 (46.2%)
Text B:    2,836 (45.0%)
불확실:      256 (4.1%)
```

#### Task 타입
```
Human vs Human:              1,400 (70%)
Human vs LLM (same):          400 (20%)
Human vs LLM (different):     200 (10%)
```

#### 텍스트 출처
```
writingprompt (Reddit):  1,036
gutenberg (고전):        1,004
sterman:                  961
poetry:                   283
llm_generated:            600
```

#### Top 5 장르
1. Contemporary / general fiction (36명)
2. Social & Cultural (33명)
3. Crime / mystery / thriller (31명)
4. History (30명)
5. Historical / SF-Fantasy (29명)

### PRISM vs LiteraryTaste 비교

| 측면 | PRISM | LiteraryTaste |
|------|-------|---------------|
| 도메인 | General AI preferences | 문학 텍스트 선호 |
| 사용자 | 1,500명 | 60명 |
| 인스턴스 | 1,500개 | 6,300개 |
| Task | Rating (0-100) | Binary choice (A/B) |
| 속성 | 8개 (values, fluency 등) | 35개 설문 항목 |
| 텍스트 | 사용자 작성 | 실제 문학 작품 |

**공통점**: 
- 활성화 추출 방법론 동일
- Linear probing 기법 동일
- Layer-wise 분석 프레임워크 동일

---

## 🔬 연구 방법론

### Phase 1: 활성화 추출
```python
# 1. 프롬프트 생성
prompt = f"Text A: {text1}\nText B: {text2}\nBetter:"

# 2. 모델 forward pass
outputs = model(**inputs, output_hidden_states=True)

# 3. 레이어별 마지막 토큰 추출
for layer in range(n_layers):
    activation = hidden_states[layer][0, -1, :]  # (768,)
```

### Phase 2: Linear Probing
```python
# 각 레이어에서 선호도 예측
for layer in range(n_layers):
    X = activations[:, layer, :]
    y = labels  # 0: Text A, 1: Text B
    
    probe = LogisticRegression()
    accuracy = probe.score(X_test, y_test)
```

**기대 결과**:
- Layer 0-3: 50-60% (약한 신호)
- Layer 5-8: 65-75% (강한 신호) ← **최고**
- Layer 9-12: 55-65% (약화)

### Phase 3: 시각화
- 레이어별 활성화 magnitude
- PCA 2D 투영
- t-SNE 클러스터링
- Text A vs B 비교
- 히트맵

### Phase 4: Steering (계획)
```python
# 장르별 스티어링 벡터
literary_vec = activations[literary_users].mean(axis=0)
action_vec = activations[action_users].mean(axis=0)
steering = literary_vec - action_vec

# 생성 시 적용
model.generate(..., steering_vector=steering)
```

---

## 💡 연구 질문

### RQ1: 사용자 특성의 레이어별 표상
- 어느 레이어에서 사용자 취향이 가장 강하게 인코딩되는가?
- Attention 패턴은 어떻게 변화하는가?
- **가설**: 중간 레이어 (Layer 5-8)에서 최대 정확도

### RQ2: Human vs LLM 텍스트 구별
- 모델이 LLM 생성 텍스트를 구별할 수 있는가?
- 어느 레이어에서 구별이 가장 명확한가?
- **가설**: 후기 레이어에서 style difference 포착

### RQ3: 장르별 활성화 패턴
- 장르 선호에 따라 활성화 패턴이 다른가?
- 클러스터링으로 장르 그룹 발견 가능한가?
- **가설**: 명확한 장르 클러스터 존재

### RQ4: 도메인 간 전이 (PRISM ↔ LiteraryTaste)
- 일반 AI 선호와 문학 선호가 상관관계 있는가?
- 한 도메인의 probe가 다른 도메인에 적용 가능한가?
- **가설**: 일부 공통 표상 존재

---

## 📁 프로젝트 구조

```
llm-personalization/
├── README.md                           # 메인 문서
├── PROJECT_STATUS.md                   # 이 파일
├── requirements.txt                    # 의존성
│
├── LiteraryTaste/                      # 데이터셋
│   ├── annotated_instances.csv
│   ├── preference_tasks.csv
│   └── ANALYSIS_GUIDE.md
│
├── runs/
│   ├── qwen7b/                         # PRISM 결과
│   └── literary_taste/                 # LiteraryTaste 결과
│       ├── README.md
│       ├── eda_summary.json
│       ├── dataset_comparison.json
│       ├── demo/                       # 활성화 & 결과
│       └── figures/                    # 시각화
│
├── notebooks/
│   ├── 03_prism_intent.ipynb
│   └── 04_literary_taste_analysis.ipynb
│
├── src/                                # PRISM 코드
│   ├── common.py
│   ├── extract_acts.py
│   ├── probe.py
│   └── ...
│
├── literary_taste_eda.py               # EDA
├── demo_with_progress.py               # 활성화 추출
├── create_mock_data.py                 # Mock 데이터
├── literary_probe.py                   # Probe 학습
├── visualize_results.py                # 시각화
└── compare_datasets.py                 # 데이터셋 비교
```

---

## 🚀 다음 단계

### 즉시 (오늘 밤)
- [ ] 활성화 추출 완료 (실행 중)
- [ ] Probe 학습 실행
- [ ] 시각화 생성
- [ ] 결과 분석 및 문서화

### 단기 (1-2일)
- [ ] 전체 데이터셋 활성화 추출 (50 → 2000 샘플)
- [ ] 레이어별 정확도 분석
- [ ] Human vs LLM 구별 실험
- [ ] 장르별 클러스터링

### 중기 (1주)
- [ ] 스티어링 벡터 생성
- [ ] 텍스트 생성 조정 실험
- [ ] PRISM-LiteraryTaste 교차 분석
- [ ] 논문 작성 시작

### 장기 (1개월)
- [ ] 다중 모델 비교 (GPT-2, LLaMA, Qwen)
- [ ] 웹 인터페이스 구축
- [ ] 추천 시스템 프로토타입
- [ ] 논문 제출

---

## 🛠️ 기술 스택

### 현재 사용
- **Python 3.12**
- **PyTorch** - 모델 로딩 및 활성화 추출
- **Transformers** (Hugging Face) - GPT-2 등
- **Pandas/NumPy** - 데이터 처리
- **scikit-learn** - Linear probing
- **Matplotlib/Seaborn** - 시각화
- **tqdm** - 진행바

### 옵션
- TransformerLens - Interpretability
- Jupyter - 대화형 분석
- Plotly - 인터랙티브 시각화

---

## 🐛 알려진 이슈

### 1. Transformers 임포트 느림
**문제**: transformers 임포트에 6-7분 소요
**원인**: 내부 모듈 체크 및 캐시 확인
**해결**: 정상 동작, 한 번만 느림

### 2. 모델 로딩 시간
**문제**: GPT-2 로딩에 1-2분 소요 (CPU)
**원인**: CPU 환경에서 정상
**해결**: GPU 불필요, 기다리면 됨

### 3. torch_dtype deprecated
**문제**: `torch_dtype` 경고 메시지
**원인**: transformers 버전 업데이트
**해결**: 무해한 경고, 무시 가능

---

## 📚 참고 문헌

### 핵심 논문
1. **PRISM Alignment** - 사용자별 AI 선호 표상
2. **Representation Engineering** - LLM 행동 조정
3. **Transformer Circuits** - 내부 메커니즘 이해

### 데이터셋
1. **LiteraryTaste** - 문학 텍스트 선호도
2. **PRISM Survey** - AI 정렬 선호도

### 도구
1. **TransformerLens** - Interpretability
2. **Hugging Face Transformers** - 모델 라이브러리

---

## 🎯 연구 기여

### 학술적 기여
1. **새로운 도메인**: 문학 텍스트 personalization
2. **교차 도메인 분석**: PRISM ↔ LiteraryTaste
3. **방법론 확장**: Binary preference probing

### 실용적 기여
1. **추천 시스템**: 개인화된 텍스트 추천
2. **콘텐츠 생성**: 사용자 취향 반영 생성
3. **필터링**: 선호 기반 콘텐츠 큐레이션

---

## 📞 연락 및 협업

**프로젝트 리더**: Lucy Roh
**위치**: `/Users/lucyroh/Desktop/STUDY/Data Projects/llm-personalization`
**Git**: (준비 중)

---

## 🔖 태그

#LLM #Personalization #Interpretability #PRISM #LiteraryTaste #MachineLearning #NLP #Probing #Steering #Research

---

**노트**: 이 문서는 지속적으로 업데이트됩니다. 최신 버전은 프로젝트 루트의 `PROJECT_STATUS.md`를 참조하세요.
