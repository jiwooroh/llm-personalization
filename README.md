# 사람이 말로 밝힌 선호 vs LLM이 내부에서 읽은 선호

PRISM Alignment Dataset · Qwen2.5-7B-Instruct (Study 1) · LiteraryTaste (Study 2) · 2026-11

## 1. 연구 개요

**질문.** 사람이 "AI가 이랬으면 좋겠다"를 자기 말로 쓸 때, LLM은 그 문장을 내부에서 어떻게 읽는가?
그 내부 해석은 같은 사람이 숫자로 밝힌 선호와 얼마나 맞는가? 모델이 말로 하는 해석과는 같은가?
그리고 말로 밝힌 선호(stated)는 실제 선택(revealed)을 예측하는 데 얼마나 쓸모 있는가?

**왜 이 질문인가.** 개인화 LLM은 사용자가 자연어로 밝힌 선호(시스템 프롬프트, 메모리, 자기소개)를 입력으로 쓴다.
모델이 그 문장을 잘못 읽거나, 읽고도 말로 요약할 때 잃으면 개인화는 시작부터 어긋난다.
출력만 보는 평가(LLM-as-judge)는 개인 선호 판정이 어려우므로, 출력 이전 내부 표상에서 직접 검증한다.

| Study | 데이터 | 입력 (stated) | 정답 |
|---|---|---|---|
| 1 | PRISM 설문 1,500명 | `system_string` 자유 서술 (평균 45단어) | 같은 사람의 8개 속성 중요도 0–100 |
| 2 | LiteraryTaste 60명 | 문학적 특성 12개 중요도 설문 (4단계) | 같은 사람의 A/B 글 선택 100회 (쌍마다 3명) |

## 2. Study 1 — PRISM: 내부 표상 vs 말

### 설계

| 단계 | 스크립트 | 질문 |
|---|---|---|
| 1 표상 추출 | `src/extract_acts.py` | 레이어별 hidden state (사용자 단어 구간 평균 / 마지막 토큰) |
| 2 Probe | `src/probe.py` | ridge, 사용자 단위 5-fold, Spearman ρ. 비교: TF-IDF, 문장 임베딩, 셔플 |
| 3 직접 질문 | `src/verbal_extract.py` | 같은 모델이 8개 점수를 JSON으로 매김 |
| 3b 공정 비교 | `src/verbal_calibrated.py` | 말 점수에도 같은 ridge·5-fold 보정 (CPU) |
| 4 최소쌍 | `src/minimal_pairs.py` | 6단계 강도 사다리 / 실제 문장 + "X matters most" · "X does not matter at all" |
| 5 그림 | `src/make_figures.py` | 4장 |
| 6 Bootstrap | `src/bootstrap_ci.py` | 차이의 신뢰구간 (CPU) |

### 결과 (Qwen2.5-7B-Instruct, 1,500명)

| 방법 | 절대 점수 ρ | 상대 순위 ρ |
|---|---|---|
| 셔플 대조군 | −0.01 | −0.02 |
| TF-IDF | 0.09 | 0.10 |
| 말 (원래) | 0.10 | 0.12 |
| 말 (보정 후) | 0.12 | 0.15 |
| 문장 임베딩 (mpnet) | 0.15 | 0.16 |
| **LLM 내부 probe** (마지막 토큰, 15층) | **0.20** | **0.22** |

- 최소쌍: 강도 사다리 속성 평균 Spearman 0.87. "matters most" → 올바른 방향 98%, "does not matter at all" → 79%.
  부정 정답률은 속성마다 크게 다름 (factuality 38%, fluency 60%, 나머지 80–97%).
- 사람별 오차: 8개 점수를 고르게 준 사람보다 **속성 간 차이가 큰 사람(뚜렷한 취향)을 더 못 읽음** (사람 내 표준편차 vs 말-보정 오차 ρ=0.28, p<.001).
- 천장 속성(factuality·helpfulness·fluency)을 빼도 probe ρ는 0.204 → 0.208로 거의 같음. 천장 효과가 낮은 ρ의 주원인은 아님.

### 유의성 — 어디까지 말할 수 있나

- **probe > 말(보정 후)**: Δ=+0.082 (abs), +0.073 (rel). 8개 중 7개 속성에서 probe가 높음 (부호 검정 단측 p=0.035).
  `bootstrap_ci.py`의 CI [0.047, 0.112]는 **속성 8개를 재추출**한 것이라 사용자 단위 검정이 아님.
  사용자 단위 bootstrap은 hidden state(`H_*.npy`)가 있어야 하므로 다음 GPU 실행 때 함께 돌린다.
- **긍정 vs 부정 정답률**: 98.0% vs 79.2%, Δ=+0.188, 95% CI [0.167, 0.208] (문장 1,600개씩 재추출).
- 문장 임베딩 vs 말(보정 후)은 구분되지 않음 (p=0.23 / 0.66).

### 해석

1. 내부는 단어 이상을 읽는다. probe > 문장 임베딩 > TF-IDF, 셔플 ≈ 0.
2. 아는 것 > 말하는 것. 보정해도 차이 ≈0.07–0.08이 남는다. 예외는 diversity 하나.
3. 부정은 반영되지만 덜 안정적이다. 특히 모두가 중요하다고 여기는 속성(factuality)에서 "중요하지 않다"는 잘 안 읽힌다.

### 확장 분석 (모두 약하거나 비유의, 2026-11-21)

- 명시적 선호 단어 밀도 vs 오차: ρ=−0.06. 실질적으로 무시할 수준.
- "personalise/personal/tailor"를 쓴 사람의 personalisation 점수는 쓰지 않은 사람과 같음 (67.8 vs 68.0).
- PRISM 대화 평점과 연결: 선호 예측 오차 vs 대화 만족도 ρ=−0.02 (p=0.43). 대화 평점은 사람별 전반적 후한 정도(response bias)가 지배적 (설문 평균 vs 대화 평점 ρ=0.44).
- PRISM-X(Kirk et al. 2026)는 논문에서 공개를 밝혔지만 2026-11-21 기준 HF·GitHub에서 찾지 못함.

## 3. Study 2 — LiteraryTaste: 말한 취향은 실제 선택을 예측하나

PRISM 대화 평점의 response bias 문제를 피하려고, **강제 선택(A/B)** 데이터에서 같은 질문을 던진다.
쌍마다 3명이 판정했으므로 "남들끼리 얼마나 일치하나"가 공유 취향의 기준선이 된다.

### 설계 (`src/literary/`)

- `embed_texts.py`: 4,000개 글을 임베딩 (MiniLM, Qwen2.5-0.5B-Instruct 전 레이어 평균 풀링). CPU.
- `stated_vs_revealed.py`: 쌍 단위 5-fold (테스트 쌍은 처음 보는 글). "모름" 256건 제외, 5,744개 선택.

| 모델 | 내용 |
|---|---|
| population | 모두에게 같은 로지스틱, 입력 = emb(A) − emb(B) (공유 취향) |
| + stated | population + (임베딩 차이 PCA 32차원 × 설문 12개 점수) 상호작용 |
| + stated (셔플) | 설문을 사람끼리 섞은 대조군 |
| + revealed | 그 사람의 학습 fold 선택만으로 개인 가중치 (population 위에 offset) |
| + revealed (다른 사람) | 다른 사람의 개인 가중치를 적용한 대조군 |
| crowd | 같은 쌍을 본 다른 두 명의 다수결 |

### 결과 (MiniLM)

| 방법 | 정확도 |
|---|---|
| 위치 (항상 A) | 0.506 |
| 길이 차이만 | 0.594 |
| crowd (다른 두 명) | 0.587 |
| **population** | **0.610** |
| + stated | 0.575 |
| + stated (셔플) | 0.576 |
| + revealed | 0.615 |
| + revealed (다른 사람) | 0.586 |

**길이 통제 + Qwen2.5-0.5B-Instruct 레이어별** (`--with_length`, population에 길이 차이 공변량 추가). Δ는 population 대비, 괄호는 사용자 단위 bootstrap 95% CI.

| 표현 | population | + stated (Δ) | + revealed (Δ) | revealed − 다른 사람 |
|---|---|---|---|---|
| MiniLM | 0.624 | −0.029 [−0.044, −0.013] | +0.018 [−0.001, 0.039] | +0.029 (p=0.002) |
| Qwen L6 | 0.647 | −0.005 [−0.019, 0.010] | **+0.018 [0.003, 0.037]** | +0.034 (p<.001) |
| Qwen L12 | 0.647 | −0.007 [−0.022, 0.008] | **+0.022 [0.006, 0.041]** | +0.039 (p<.001) |
| Qwen L18 | 0.646 | −0.012 [−0.026, 0.003] | **+0.019 [0.003, 0.038]** | +0.036 (p<.001) |
| Qwen L24 | 0.616 | +0.003 [−0.008, 0.014] | **+0.019 [0.006, 0.037]** | +0.026 (p<.001) |

- 읽는 사람끼리 일치율은 58.7%. 공유 취향이 약하고, 그중 상당 부분이 **길이**다 (긴 글을 고른 비율 59.3%).
  LLM 중간층 표현은 길이를 넘어서는 공유 취향을 더 잘 잡는다 (0.647 > MiniLM 0.624 > 길이만 0.594).
- **stated는 도움이 안 된다**: 어떤 표현에서도 개선이 없고, MiniLM에선 오히려 악화 (셔플한 설문과 차이 없음).
- **revealed는 사람 고유 신호가 있다**: Qwen 표현에서는 네 레이어 모두 population보다 유의하게 높고 (+1.8–2.2%p),
  다른 사람의 가중치를 쓰면 2.6–3.9%p 떨어진다. 레이어를 골라서 나온 결과가 아니다.
- MiniLM, 길이 통제 없음: revealed > stated +0.042 (CI [0.024, 0.061]).
- 말-행동 일치: 개인 가중치로 설문 점수를 예측하면 12개 중 2개만 순열 null보다 높음 ("surprising plot", "skillful language", p≈0.03, 다중비교 보정 전).

### 해석

말로 밝힌 문학 취향(설문)은 실제 선택을 공유 취향 이상으로 설명하지 못한다. Chung et al.(2025)의 "stated preferences had limited utility"를 임베딩 모델로 재현.
반면 같은 사람의 과거 선택에는 작지만(+2%p) 유의한 그 사람 고유의 신호가 있다. Study 1과 함께 보면,
**자연어 선호는 모델 내부에서는 어느 정도 읽히지만(ρ≈0.2), 그 선호가 실제 선택을 예측하는지는 별개의 문제**이고,
개인화는 말보다 행동 이력에서 더 많은 정보를 얻는다.

## 4. 한계

- PRISM `stated_prefs`는 잡음 있는 자기보고라 상한이 낮고, 사람 독자 기준선이 없다.
- probe 입력 차원(3,584)이 크다. 768차원 문장 임베딩과의 비교로 일부 반박.
- 부정/긍정 문장은 부정 여부 외에도 표현이 다르고, 천장 효과로 위·아래 여유가 다르다.
- 모델·프롬프트 각 1개, probe는 선형.
- LiteraryTaste는 60명 × 100쌍이라 개인 모델 학습량이 작다 (fold당 사람당 ~77개). 설문은 12개 특성 중요도만 사용 (장르·독서 습관 미사용).
- LiteraryTaste의 글 표현은 작은 모델(MiniLM, Qwen-0.5B)의 평균 풀링이다.

## 5. 레포 구조

```
llm-personalization/
├── README.md · run_all.sh · requirements.txt
├── src/
│   ├── common.py · extract_acts.py · probe.py · verbal_extract.py   Study 1 (GPU)
│   ├── verbal_calibrated.py · bootstrap_ci.py · minimal_pairs.py · make_figures.py
│   └── literary/  embed_texts.py · stated_vs_revealed.py           Study 2 (CPU)
├── notebooks/03_prism_intent.ipynb
├── runs/
│   ├── qwen7b/          Study 1 결과 (csv·json·figures만 git, *.npy/*.parquet 제외)
│   └── literary_taste/  Study 2 결과 (svr_*/summary.json, per_user_accuracy.csv)
├── data/ · LiteraryTaste/   내려받는 데이터 (git 제외)
└── archive/             이전 설계안, Kiro 임시 스크립트·mock 데이터 (결과로 쓰지 않음)
```

## 6. 실행 방법

```bash
git clone <this repo> && cd llm-personalization
python3.12 -m venv ~/.venvs/llm-pers && source ~/.venvs/llm-pers/bin/activate
pip install -r requirements.txt
```

> macOS에서 레포가 iCloud 동기화 폴더(Desktop/Documents) 안에 있으면 venv를 그 밖에 만든다.
> 클라우드로 내려간 파일 때문에 `import transformers`가 CPU 0%로 몇 분씩 멈춘다.

**Study 1 (GPU, vast.ai 24GB+ 1장, 1시간 이내)**

```bash
mkdir -p data && curl -L -o data/survey.jsonl https://raw.githubusercontent.com/HannahKirk/prism-alignment/main/data/survey.jsonl
bash run_all.sh Qwen/Qwen2.5-7B-Instruct runs/qwen7b
cd src && python verbal_calibrated.py --run ../runs/qwen7b && python bootstrap_ci.py --run ../runs/qwen7b
```

**Study 2 (CPU, 약 30분)**

```bash
git clone https://github.com/mj-storytelling/LiteraryTaste.git
cd src/literary
python embed_texts.py                                   # MiniLM
python embed_texts.py --skip_minilm --llm Qwen/Qwen2.5-0.5B-Instruct
python stated_vs_revealed.py --emb minilm
python stated_vs_revealed.py --emb Qwen2.5-0.5B-Instruct --layer 12 --with_length
```

## 7. 진행 상황

- 2026-11-20: Qwen2.5-7B로 PRISM 1,500명 실행, 말 점수 보정 비교.
- 2026-11-21: bootstrap, 사람별 오차 분석, PRISM 대화 평점 연결(비유의), LiteraryTaste stated vs revealed 실험.
- 다음: 사용자 단위 bootstrap + Llama-3.1-8B 재현 (GPU) → LiteraryTaste 설문을 자연어로 바꿔 LLM 내부 probe (Study 1 방법을 Study 2에 적용) → 선행연구 대비 차별화 한 문장.

## 참고문헌

- Kirk et al. (2024). *The PRISM Alignment Dataset.* NeurIPS D&B. [arXiv 2404.16019](https://arxiv.org/abs/2404.16019)
- Kirk et al. (2026). *PRISM-X.* [arXiv 2605.13307](https://arxiv.org/abs/2605.13307)
- Chung et al. (2025). *LiteraryTaste.* [arXiv 2511.09310](https://arxiv.org/abs/2511.09310) · [data](https://github.com/mj-storytelling/LiteraryTaste)
- Kantharuban et al. (2025). *Stereotype or Personalization?* Findings ACL. [arXiv 2410.05613](https://arxiv.org/abs/2410.05613)
- Neplenbroek et al. (2025). *Reading Between the Prompts.* EMNLP. [arXiv 2505.16467](https://arxiv.org/abs/2505.16467)
