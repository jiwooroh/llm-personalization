# 사람이 말로 밝힌 선호 vs LLM이 내부에서 읽은 선호

PRISM Alignment Dataset · Qwen2.5-7B-Instruct · Llama-3.1-8B-Instruct (Study 1) · LiteraryTaste (Study 2, 3) · 2026-11

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
| 3 | LiteraryTaste 60명 | 설문을 옮긴 1인칭 취향 프로필 + 글 쌍을 LLM에 함께 입력 | 같은 사람의 A/B 선택 |

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
| 6 Bootstrap | `src/user_bootstrap.py` | 사용자 단위 paired bootstrap. `probe.py`가 저장한 사람별 예측(`probe_oof.npz`)만 씀 (CPU, 1분) |

### 결과 (1,500명, 괄호는 사용자 단위 bootstrap 95% CI)

| 방법 | Qwen-7B 절대 ρ | Qwen-7B 상대 ρ | Llama-8B 절대 ρ | Llama-8B 상대 ρ |
|---|---|---|---|---|
| 셔플 대조군 | −0.01 | −0.02 | −0.02 | −0.01 |
| TF-IDF | 0.09 | 0.10 | 0.09 | 0.10 |
| 말 (원래) | 0.10 | 0.12 | 0.11 | 0.10 |
| 말 (보정 후) | 0.12 | 0.15 | 0.13 | 0.13 |
| 문장 임베딩 (mpnet) | 0.14 | 0.16 | 0.14 | 0.16 |
| **LLM 내부 probe** (마지막 토큰) | **0.20** [0.18, 0.23] | **0.22** [0.19, 0.24] | **0.21** [0.18, 0.23] | **0.21** [0.19, 0.23] |

probe 레이어는 절대 점수 기준 최적 레이어 (Qwen 15/28층, Llama 10/32층). 상대 순위도 같은 레이어로 평가.

- 최소쌍 (Qwen / Llama): 강도 사다리 속성 평균 Spearman 0.87 / 0.70. "matters most" → 올바른 방향 98% / 97%, "does not matter at all" → 79% / 50%.
  부정 정답률은 속성마다 크게 다름. Qwen은 factuality 38%, fluency 60%, 나머지 80–97%.
  Llama는 factuality 4%, fluency 6%, helpfulness 30%로, **모두가 중요하게 여기는 속성의 부정을 거의 못 읽는다**.
- 사람별 오차: 8개 점수를 고르게 준 사람보다 **속성 간 차이가 큰 사람(뚜렷한 취향)을 더 못 읽음** (사람 내 표준편차 vs 말-보정 오차 ρ=0.28, p<.001).
- 천장 속성(factuality·helpfulness·fluency)을 빼도 probe ρ는 0.204 → 0.208로 거의 같음. 천장 효과가 낮은 ρ의 주원인은 아님.

### 유의성 — 어디까지 말할 수 있나

사용자 1,500명을 재추출하는 paired bootstrap (2,000회, `src/user_bootstrap.py`). probe − 비교 대상의 ρ 차이:

| 비교 | Qwen-7B 절대 | Qwen-7B 상대 | Llama-8B 절대 | Llama-8B 상대 |
|---|---|---|---|---|
| probe − 말(보정 후) | **+0.082** [0.055, 0.109] | **+0.071** [0.046, 0.097] | **+0.083** [0.055, 0.111] | **+0.078** [0.051, 0.105] |
| probe − 문장 임베딩 | **+0.059** [0.037, 0.080] | **+0.061** [0.037, 0.083] | **+0.064** [0.041, 0.085] | **+0.051** [0.029, 0.074] |

- 모두 p<0.001. **"아는 것 > 말하는 것"이 사용자 단위로, 두 모델 계열에서 유의하다.**
- 문장 임베딩 vs 말(보정 후)은 대체로 구분되지 않음 (Qwen p=0.12 / 0.48, Llama p=0.19 / 0.04).
- **긍정 vs 부정 정답률** (Qwen): 98.0% vs 79.2%, Δ=+0.188, 95% CI [0.167, 0.208] (문장 1,600개씩 재추출).

### 해석

1. 내부는 단어 이상을 읽는다. probe > 문장 임베딩 > TF-IDF, 셔플 ≈ 0. 두 모델에서 크기가 거의 같다 (ρ≈0.21).
2. 아는 것 > 말하는 것. 말 점수를 같은 방식으로 보정해도 차이 ≈0.07–0.08이 남고, 사용자 단위 CI가 0에서 멀다.
3. 부정은 덜 안정적이다. 모두가 중요하다고 여기는 속성(factuality·fluency)의 "중요하지 않다"는 잘 안 읽히고, Llama에서 훨씬 심하다.

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

## 4. Study 3 — 말한 취향을 모델에 넣으면, 모델이 그 사람의 선택에 쓰나

LiteraryTaste 설문을 1인칭 취향 문장(평균 88단어)으로 바꿔 글 쌍과 함께 Qwen2.5-7B-Instruct에 넣고, "이 독자는 어느 글을 고를까?"를 묻는다.

| 조건 | 입력 | 의미 |
|---|---|---|
| none | 프로필 없음 ("a typical reader") | 공유 취향 |
| own | 본인 프로필 | 말한 선호를 선택에 쓰나 |
| other | 다른 독자 프로필 | 대조군 (핵심 비교: own − other) |
| own_pos | 본인 프로필에서 "I don't care about…" 문장 제거 | 부정 문장 효과 |

- knows: 마지막 토큰 hidden state의 두 순서(A/B 교체) 차이 → PCA 32 + 로지스틱, Study 2와 같은 쌍 단위 5-fold.
- says: logit("A") − logit("B")를 두 순서로 상쇄. 학습 없음.
- 독자 단위 paired bootstrap (2,000회). 비교 기준: Study 2의 +revealed 이득(+1.8%p).
- 스크립트: `src/literary/make_profiles.py` → `extract_joint.py` (GPU) → `probe_joint.py` (CPU).

### 결과 (60명, 선택 5,744개, 정확도)

| 조건 | knows (probe, 12층) | says (A/B 로짓) |
|---|---|---|
| none | 0.636 | 0.614 |
| own | 0.639 | 0.596 |
| other | 0.635 | 0.593 |
| own_pos | 0.636 | 0.601 |

| 비교 | knows Δ [95% CI] | says Δ [95% CI] |
|---|---|---|
| **own − other** | +0.003 [−0.003, 0.009] | +0.003 [−0.013, 0.017] |
| own − none | +0.002 [−0.005, 0.010] | **−0.018** [−0.031, −0.007] |
| other − none | −0.001 [−0.010, 0.007] | **−0.021** [−0.036, −0.006] |
| own − own_pos | +0.002 [−0.004, 0.008] | −0.005 [−0.013, 0.002] |
| knows − says | own +0.043, other +0.042, own_pos +0.035, none +0.023 (모두 CI가 0 위) | |

### 해석

1. **본인 프로필과 남의 프로필이 구분되지 않는다** (own ≈ other, knows·says 모두). 모델은 프로필을 읽어도 그 사람의 선택을 맞히는 데 쓰지 않는다.
   own − other (+0.3%p)는 Study 2의 행동 이력 이득(+1.8%p)보다 훨씬 작다. "말보다 행동"이 LLM 안에서도 유지된다.
2. **프로필을 넣으면 말(says)은 오히려 나빠진다** (−1.8 ~ −2.1%p, 본인·남 구분 없이). 프로필은 답에 잡음으로 작용한다.
3. **knows > says가 모든 조건에서 유의하다** (+2–4%p). Study 1의 "아는 것 > 말하는 것"이 선택 예측에서도 나타난다.
   다만 knows는 공유 취향 수준(0.64, Study 2 population 0.62–0.65)이고, 프로필이 더하는 것은 없다.
4. "I don't care about…" 문장을 빼도 차이 없음.

## 5. Study 4 — 명시 vs 암묵 선호: 대화 속에 은근히 드러난 선호를 쓰나

PersonaMem-v2 (2025) 800문항, persona 195명. 같은 질문·답변 4개에 앞선 대화만 바꾼다: 선호를 직접 말함(explicit) / 대화에서 은근히 드러냄(implicit) / 각각 + 무관한 대화 2턴(_far). 4지선다 정답률(우연 0.25), 기준선은 mpnet 임베딩 유사도. 분석: `notebooks/04_study4_implicit.ipynb`, 계획·피벗 기록: `docs/study4_plan.md`.

| | none | explicit | implicit | explicit_far | implicit_far |
|---|---|---|---|---|---|
| Qwen2.5-7B | 0.23 | 0.92 | 0.51 | 0.83 | 0.41 |
| Llama-3.1-8B | 0.27 | 0.96 | 0.74 | 0.96 | 0.60 |
| 임베딩 | — | 0.87 | 0.54 | 0.50 | 0.45 |

1. 명시 선호는 모두 반영한다. 암묵 선호와의 격차는 Qwen +0.40, Llama +0.23 (둘 다 유의).
2. Qwen은 암묵 조건에서 임베딩과 차이가 없다(−0.03, n.s.). Llama는 +0.19이고 임베딩이 틀리는 문항에서도 0.65.
3. 무관한 대화가 끼면 암묵 선호가 더 많이 사라진다 (−0.11~−0.14, 명시는 0.00~−0.09).
4. 암묵 조건에서 stereotypical 선호가 anti-stereotypical보다 약 0.09 높다.

## 6. 한계

- PRISM `stated_prefs`는 잡음 있는 자기보고라 상한이 낮고, 사람 독자 기준선이 없다.
- probe 입력 차원(3,584)이 크다. 768차원 문장 임베딩과의 비교로 일부 반박.
- 부정/긍정 문장은 부정 여부 외에도 표현이 다르고, 천장 효과로 위·아래 여유가 다르다.
- Study 1은 모델 2개(Qwen, Llama), 프롬프트 1개, probe는 선형. Study 3은 Qwen-7B 하나.
- LiteraryTaste는 60명 × 100쌍이라 개인 모델 학습량이 작다 (fold당 사람당 ~77개). 설문은 12개 특성 중요도만 사용 (장르·독서 습관 미사용).
- LiteraryTaste의 글 표현은 작은 모델(MiniLM, Qwen-0.5B)의 평균 풀링이다.
- Study 3 프로필은 설문 숫자를 템플릿으로 옮긴 문장이지 독자가 직접 쓴 글이 아니다.

## 7. 레포 구조

```
llm-personalization/
├── README.md · run_all.sh · gpu_session.sh · gpu_session_study4.sh · requirements.txt
├── src/
│   ├── common.py · extract_acts.py · probe.py · verbal_extract.py   Study 1 (GPU)
│   ├── verbal_calibrated.py · user_bootstrap.py · minimal_pairs.py · make_figures.py
│   ├── literary/  embed_texts.py · stated_vs_revealed.py           Study 2 (CPU)
│   │              make_profiles.py · extract_joint.py · probe_joint.py   Study 3
│   └── implicit/  build_items.py · embed_baseline.py · extract.py · analyze_mc.py   Study 4
├── notebooks/03_prism_intent.ipynb · 04_study4_implicit.ipynb
├── docs/study4_plan.md · figures/
├── runs/
│   ├── qwen7b/ · llama8b/      Study 1 결과 (csv·json·figures만 git, *.npy/*.parquet 제외)
│   ├── qwen05b/                Study 1 파이프라인 검증 (Mac, Qwen-0.5B)
│   ├── implicit/               Study 4 결과 (qwen7b/ · llama8b/ · pilot15/)
│   └── literary_taste/         Study 2 결과 (svr_*/), Study 3 결과 (joint_qwen7b/ 요약만)
└── data/ · LiteraryTaste/      내려받는 데이터 (git 제외)
```

## 8. 실행 방법

```bash
git clone <this repo> && cd llm-personalization
python3.12 -m venv ~/.venvs/llm-pers && source ~/.venvs/llm-pers/bin/activate
pip install -r requirements.txt
```

> macOS에서 레포가 iCloud 동기화 폴더(Desktop/Documents) 안에 있으면 venv를 그 밖에 만든다.
> 클라우드로 내려간 파일 때문에 `import transformers`가 CPU 0%로 몇 분씩 멈춘다.

**GPU 한 번에 전부 (24GB+ GPU 1장, 디스크 80GB, 약 3시간)**: Study 1 (Qwen-7B, Llama-3.1-8B) + Study 3.
Llama는 Hugging Face 접근 승인과 토큰(`HF_TOKEN`)이 필요하고, 없으면 건너뛴다. Study 3은 `LiteraryTaste/`가 있어야 한다.

```bash
PYTHONUNBUFFERED=1 bash gpu_session.sh 2>&1 | tee session.log   # 일부만: SKIP=B1,B2,A 중 선택
```

> 디스크가 모자라면(32GB 등) 모델 가중치가 깨진 채 저장될 수 있다. Llama가 엉뚱한 글을 내면 가중치 checksum부터 확인한다.

**Study 1만 (GPU 24GB+ 1장, 1시간 이내)**

```bash
mkdir -p data && curl -L -o data/survey.jsonl https://raw.githubusercontent.com/HannahKirk/prism-alignment/main/data/survey.jsonl
bash run_all.sh Qwen/Qwen2.5-7B-Instruct runs/qwen7b
# run_all.sh가 verbal_calibrated.py와 user_bootstrap.py까지 돌린다
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

## 9. 진행 상황

- 2026-11-20: Qwen2.5-7B로 PRISM 1,500명 실행, 말 점수 보정 비교.
- 2026-11-21: bootstrap, 사람별 오차 분석, PRISM 대화 평점 연결(비유의), LiteraryTaste stated vs revealed 실험.
- 2026-11-21 (밤): 사용자 단위 bootstrap 준비 (`probe.py`가 사람별 예측 저장, `user_bootstrap.py`), Qwen-0.5B로 Mac에서 Study 1 전체 검증.
  Study 3 스크립트 3개 작성, 0.5B·독자 2명으로 테스트. `gpu_session.sh`로 GPU 한 세션에 묶음.
- 2026-11-22: GPU 세션. Qwen-7B 재실행 + 사용자 단위 bootstrap, Llama-3.1-8B 재현, Study 3 전체 실행.
  "아는 것 > 말하는 것"이 두 모델에서 사용자 단위로 유의. Study 3: 본인 프로필 ≈ 남의 프로필.
- 다음: 선행연구 대비 차별화 한 문장, Study 3를 독자가 직접 쓴 프로필로 확장할 수 있는지 검토.

## 참고문헌

- Kirk et al. (2024). *The PRISM Alignment Dataset.* NeurIPS D&B. [arXiv 2404.16019](https://arxiv.org/abs/2404.16019)
- Kirk et al. (2026). *PRISM-X.* [arXiv 2605.13307](https://arxiv.org/abs/2605.13307)
- Chung et al. (2025). *LiteraryTaste.* [arXiv 2511.09310](https://arxiv.org/abs/2511.09310) · [data](https://github.com/mj-storytelling/LiteraryTaste)
- Kantharuban et al. (2025). *Stereotype or Personalization?* Findings ACL. [arXiv 2410.05613](https://arxiv.org/abs/2410.05613)
- Neplenbroek et al. (2025). *Reading Between the Prompts.* EMNLP. [arXiv 2505.16467](https://arxiv.org/abs/2505.16467)
