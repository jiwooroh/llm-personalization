# Literary Taste Analysis Guide

이 문서는 LiteraryTaste 데이터셋을 사용한 사용자 취향 분석 프로젝트의 완전한 가이드입니다.

## 📚 프로젝트 개요

**목표**: 사용자의 문학 취향을 LLM 내부 표상을 통해 분석하고, 개인화된 텍스트 추천 시스템 구축

**데이터**: 
- 60명 사용자
- 6,300개 선호도 판단 (사용자당 105개)
- 2,000개 고유 텍스트 쌍

## 🗂️ 데이터 구조

### 1. `annotated_instances.csv`
사용자 프로필 + 선호도 판단

**주요 컬럼**:
- `user`: 사용자 ID
- `instance_id`: 텍스트 쌍 ID
- `displayed_text`: HTML 형식의 Text A vs Text B
- `preference:::Text A`: Text A 선호 여부
- `preference:::Text B`: Text B 선호 여부
- `preference:::I am not sure`: 불확실
- 150+ 개 설문 항목 (독서 습관, 장르 선호, 인구통계 등)

### 2. `preference_tasks.csv`
텍스트 쌍 원본 데이터

**주요 컬럼**:
- `id`: Task ID
- `type`: Task 유형
  - `human_vs_human`: 사람 vs 사람 (1,400개)
  - `human_vs_llm_samecontent`: 같은 내용, 다른 작가 (400개)
  - `human_vs_llm_diffcontent`: 완전히 다른 내용 (200개)
- `text1`, `text2`: 비교할 텍스트
- `text1_source`, `text2_source`: 텍스트 출처
  - `gutenberg`: Project Gutenberg 고전 문학
  - `writingprompt`: Reddit r/WritingPrompts
  - `sterman`: Sterman 코퍼스
  - `poetry`: 시

## 📊 탐색적 데이터 분석 (EDA)

### 실행 방법
```bash
cd /Users/lucyroh/Desktop/STUDY/Data\ Projects/llm-personalization
source .venv/bin/activate
python literary_taste_eda.py
```

### 주요 발견사항

#### 선호도 분포
- Text A: 46.2% (2,908개)
- Text B: 45.0% (2,836개)
- 불확실: 4.1% (256개)

→ **거의 균등 분포**: 텍스트 품질이 비슷하게 선별됨

#### 인기 장르
1. Contemporary / general fiction (36명)
2. Social & Cultural (33명)
3. Crime / mystery / thriller (31명)
4. History (30명)
5. Historical / SF-Fantasy (각 29명)

#### 텍스트 통계
- 평균 길이: ~690 characters
- 중앙값: 771-772 characters
- 범위: 48 - 1,439 characters

## 🧠 모델 분석 파이프라인

### Phase 1: 활성화 추출 ✅

**목표**: LLM의 각 레이어에서 활성화(hidden states) 추출

**스크립트**:
```bash
# 간단한 데모 (5개 샘플)
python literary_minimal_demo.py

# 전체 데이터 (시간 소요)
python literary_extract_simple.py
```

**출력**:
- `runs/literary_taste/demo/demo_activations.npz`: 활성화 배열
- `runs/literary_taste/demo/demo_metadata.json`: 메타데이터

**활성화 형식**:
```python
activations.shape  # (n_samples, n_layers+1, d_model)
# 예: (10, 13, 768) for GPT-2
```

### Phase 2: Probe 학습 🚧

**목표**: 활성화로부터 사용자 선호 예측

**스크립트**:
```bash
python literary_probe.py
```

**방법**:
1. 각 레이어에서 Logistic Regression 학습
2. Train/Test Split (70/30)
3. Cross-validation (3-fold)
4. 레이어별 정확도 비교

**기대 결과**:
- 초기 레이어: ~50-60% (낮은 정확도)
- 중간 레이어: ~65-75% (최고 정확도)
- 후기 레이어: ~55-65% (약간 감소)

→ **중간 레이어에서 사용자 특성이 가장 강하게 인코딩됨**

### Phase 3: 스티어링 벡터 생성 🔜

**목표**: 특정 취향으로 모델 조정

**방법**:
1. 특정 장르 선호 사용자들의 평균 활성화 계산
2. 스티어링 벡터 = preferred_direction - neutral_direction
3. 생성 시 활성화에 벡터 추가

**예시 코드**:
```python
# 'literary style' 선호 사용자의 평균 활성화
literary_users = users[users['genre_Literary'] == 1]
literary_mean = activations[literary_users].mean(axis=0)

# 'action' 선호 사용자
action_mean = activations[action_users].mean(axis=0)

# 스티어링 벡터
steering_vec = literary_mean - action_mean

# 생성 시 적용
model.generate(..., steering_vector=steering_vec, layer=best_layer)
```

## 🔬 실험 아이디어

### 1. 장르별 활성화 패턴
- 각 장르 선호 사용자의 활성화 클러스터링
- t-SNE로 시각화
- 장르 간 거리 측정

### 2. Human vs LLM 텍스트 구별
- LLM 생성 텍스트가 사람 글과 다르게 인코딩되는가?
- 어느 레이어에서 구별이 가장 명확한가?

### 3. 독서 시간과 선호 복잡도
- 독서 시간이 많은 사용자일수록 복잡한 텍스트 선호?
- 활성화 엔트로피와 독서 습관 상관관계

### 4. 사용자 프로필 재구성
- 활성화만으로 사용자 설문 응답 예측 가능?
- 예: 활성화 → 장르 선호도 예측

## 📝 코드 구조

```
llm-personalization/
├── LiteraryTaste/                   # 원본 데이터
│   ├── annotated_instances.csv
│   ├── preference_tasks.csv
│   └── ANALYSIS_GUIDE.md           # 이 파일
│
├── notebooks/
│   └── 04_literary_taste_analysis.ipynb  # 대화형 분석
│
├── runs/
│   └── literary_taste/
│       ├── README.md               # 프로젝트 요약
│       ├── eda_summary.json        # EDA 통계
│       ├── demo/                   # 데모 출력
│       └── activations/            # 전체 활성화
│
├── literary_taste_eda.py           # EDA 스크립트
├── literary_minimal_demo.py        # 간단한 데모
├── literary_extract_simple.py      # 활성화 추출
└── literary_probe.py               # Probe 학습
```

## 🚀 Quick Start

### 1. 환경 설정
```bash
cd /Users/lucyroh/Desktop/STUDY/Data\ Projects/llm-personalization
source .venv/bin/activate

# 필요 라이브러리
pip install pandas numpy matplotlib seaborn scikit-learn
pip install torch transformers  # 이미 설치됨
```

### 2. EDA 실행
```bash
python literary_taste_eda.py
# 출력: runs/literary_taste/eda_summary.json
```

### 3. 활성화 추출 (소량 테스트)
```bash
python literary_minimal_demo.py
# 출력: runs/literary_taste/demo/
```

### 4. Probe 학습
```bash
python literary_probe.py
# 출력: runs/literary_taste/demo/probe_results.json
```

### 5. 노트북 실행
```bash
jupyter notebook notebooks/04_literary_taste_analysis.ipynb
```

## 📚 참고 자료

### 논문
- **PRISM**: Personalized LLM alignment (기존 프로젝트)
- **Representation Engineering**: Steering LLM behavior
- **TransformerLens**: Interpretability toolkit

### 코드
- 기존 PRISM 구현: `src/` 디렉토리
- TransformerLens: https://github.com/neelnanda-io/TransformerLens
- Hugging Face Transformers: https://huggingface.co/docs/transformers

### 데이터셋
- LiteraryTaste: https://github.com/mj-storytelling/LiteraryTaste
- Project Gutenberg: https://www.gutenberg.org/
- r/WritingPrompts: https://www.reddit.com/r/WritingPrompts/

## 💡 Troubleshooting

### 모델 로딩이 느린 경우
```python
# 더 작은 모델 사용
MODEL_NAME = "gpt2"  # 124M params (가장 빠름)
# vs
MODEL_NAME = "gpt2-small"  # 동일
MODEL_NAME = "pythia-70m"  # 더 작음

# 캐시 활용
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    cache_dir="./cache"  # 한 번만 다운로드
)
```

### 메모리 부족
```python
# 배치 크기 줄이기
batch_size = 4  # 기본 16에서 줄임

# 샘플 수 제한
N_SAMPLES = 50  # 전체 2000 대신
```

### GPU 없음
```python
# CPU로 강제
DEVICE = "cpu"
torch_dtype = torch.float32  # bfloat16 대신
```

## 🎯 다음 단계

### 단기 (1-2일)
- [ ] 전체 데이터셋으로 활성화 추출
- [ ] Probe 학습 및 레이어별 분석
- [ ] 시각화 개선

### 중기 (1주)
- [ ] 스티어링 벡터 생성
- [ ] 텍스트 생성 조정 테스트
- [ ] 장르별 클러스터링

### 장기 (1개월)
- [ ] 사용자 프로필 기반 추천 시스템
- [ ] 웹 인터페이스 구축
- [ ] 논문 작성

---

**Last Updated**: 2026-11-20  
**Author**: LLM Personalization Project
