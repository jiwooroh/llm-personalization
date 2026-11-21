# 사람이 말로 밝힌 선호 vs LLM이 내부에서 읽은 선호

*CUAI NLP 프로젝트 · PRISM Alignment Dataset · 2026-11*

## 1. 연구 개요

### 질문
사람이 "AI가 이랬으면 좋겠다"를 **자기 말로** 쓸 때, LLM은 그 문장을 내부에서 어떻게 읽는가?
그 내부 해석은 같은 사람이 **숫자로** 밝힌 선호와 얼마나 맞는가? 어긋난다면 어떤 **언어 형태**(부정, 강도, hedging) 때문인가?

### 왜 이 질문인가
- 개인화 LLM은 사용자가 자연어로 밝힌 선호(시스템 프롬프트, 메모리, 자기소개)를 입력으로 쓴다. 모델이 그 문장을 잘못 읽으면 개인화는 시작부터 어긋난다.
- 출력만 보는 평가(LLM-as-judge)는 개인 선호 판정이 어렵다고 알려져 있다. 그래서 **출력 이전 단계, 즉 모델 내부 표상**에서 의사가 보존되는지를 직접 검증한다.
- NLP 관점: 부정·강도 표현에 대한 모델의 둔감함을 실제 사용자 문장 위에서 측정한다.

### 데이터: PRISM 설문 (1,500명, 모두 두 형태를 다 가짐)
| 필드 | 형태 | 역할 |
|---|---|---|
| `system_string` | 자유 서술 "AI가 이랬으면…" (평균 45단어) | **입력** — 모델이 읽는 문장 |
| `stated_prefs` | 같은 사람이 8개 속성 중요도를 0–100으로 매김 | **정답** |

8개 속성: values, fluency, factuality, safety, diversity, creativity, helpfulness, personalisation.
helpfulness·factuality는 62%가 90점 이상(천장 효과)이라, 절대 점수와 함께 **사람 안의 상대 순위**(그 사람 평균 대비)도 분석한다.

출처: Kirk et al., *The PRISM Alignment Dataset*, NeurIPS 2024 ([GitHub](https://github.com/HannahKirk/prism-alignment)).

### 실험 설계
| 단계 | 질문 | 방법 | 정답 |
|---|---|---|---|
| 1 표상 추출 | — | 각 문장을 모델에 넣고 레이어별 hidden state 저장 (사용자 단어 구간 평균 / 마지막 토큰) | — |
| 2 Probe (내부) | 내부 표상이 그 사람의 숫자 평점을 담고 있나 | 레이어별 ridge 회귀, 사용자 단위 5-fold, 속성별 Spearman ρ. 비교: TF-IDF, 문장 임베딩, 셔플 대조군 | `stated_prefs` |
| 3 직접 질문 (말) | 모델이 말로 하는 해석은 맞나 | 같은 모델에게 8개 속성 점수를 JSON으로 매기게 함. 2와 비교 = "아는 것 vs 말하는 것" | `stated_prefs` |
| 4 최소쌍 (언어 형태) | 내부 해석이 부정·강도를 따라가나 | A) 속성별 6단계 강도 사다리(essential → do not want). B) 실제 문장 + "X matters most" / "X does not matter at all" 추가 | 만들 때 정해진 방향 |
| 5 그림 | — | 발표용 그림 4장 | — |

### 예상 결과의 해석
- probe > TF-IDF·임베딩 → 모델이 단어 이상을 읽는다
- probe와 직접 질문이 속성별로 갈림 → 모델이 아는 것과 말하는 것이 다르다
- 최소쌍에서 부정("does not matter")의 정답률이 긍정보다 낮음 → 부정에 둔감한 표상 = 사람의 의사와 내부 표상이 어긋나는 구체적 원인

## 2. 레포지토리 구조

```
llm-personalization/
├── README.md                ← 이 문서
├── run_all.sh               ← 실험 1–5단계를 한 번에 실행
├── requirements.txt
├── src/                     ← 본 실험 코드
│   ├── common.py            데이터 로딩(PRISM 설문), 프롬프트 형식, hidden state 추출
│   ├── extract_acts.py      1단계: 레이어별 hidden state 저장
│   ├── probe.py             2단계: probe + TF-IDF·임베딩·셔플 기준선, 최적 레이어 probe 저장
│   ├── verbal_extract.py    3단계: 모델에게 직접 점수 매기게 하기
│   ├── minimal_pairs.py     4단계: 강도 사다리 + 문장 추가 실험
│   └── make_figures.py      5단계: 발표용 그림
├── notebooks/
│   ├── 03_prism_intent.ipynb          ← 본 실험을 단계별로 실행·해석하는 노트북
│   ├── 01_transformer_lens_basics.ipynb   (사전 실습) TransformerLens 기초
│   └── 02_activation_steering.ipynb       (사전 실습) ActAdd·CAA steering
├── slides/                  사전 실습 슬라이드 (PDF)
├── figures/                 사전 실습 그림 (TransformerLens 작동 구조)
├── stated_revealed/         (보관) 이전 설계안 — 이번 실험에는 쓰지 않음
├── Claude outputs/          (보관) 이전 작업 산출물
├── data/                    실행 시 생성 — PRISM survey.jsonl (git 제외)
└── runs/<모델>/             실행 시 생성 — 결과 (git 제외)
```

### 실행 결과물 (`runs/<모델>/`)
| 파일 | 내용 |
|---|---|
| `users.parquet` | 사용자, 문장, 8개 점수, fold |
| `H_mean.npy`, `H_last.npy` | hidden state [사용자, 레이어, 차원] (용량 큼, 공유 시 제외) |
| `probe_results.csv`, `probe_summary.json` | 레이어·속성별 Spearman, 기준선, 최적 레이어 |
| `verbal.parquet`, `verbal_summary.json` | 모델이 직접 매긴 점수와 원문 응답, 정답과의 상관 |
| `minimal_ladder.csv`, `minimal_context.csv`, `minimal_summary.json` | 최소쌍 결과 |
| `figures/` | 1_layer_curve, 2_per_attribute, 3_ladder, 4_context_edits |

## 3. 실행 방법 (vast.ai)

- GPU 24GB 이상 1장 (RTX 4090이면 7B 모델 충분), 디스크 60GB
- 기본 모델 `Qwen/Qwen2.5-7B-Instruct` (접근 승인 불필요). Llama를 쓰려면 `meta-llama/Llama-3.1-8B-Instruct` + `HF_TOKEN`

```bash
pip install -r requirements.txt
mkdir -p data && curl -L -o data/survey.jsonl \
  https://raw.githubusercontent.com/HannahKirk/prism-alignment/main/data/survey.jsonl
bash run_all.sh Qwen/Qwen2.5-7B-Instruct runs/qwen7b      # 스크립트로 한 번에
# 또는 notebooks/03_prism_intent.ipynb 를 Jupyter에서 위에서부터 실행
```

예상 시간: GPU 1장 기준 1시간 이내. 노트북은 결과 파일이 이미 있으면 그 단계를 건너뛴다(재실행 시 시간 절약).

## 4. 한계

- `stated_prefs`도 잡음이 있는 자기보고다. 같은 사람의 글과 숫자가 원래 어긋날 수 있어서 일치도의 상한이 낮다.
- 최소쌍 문장은 템플릿이다. 자연스러움이 아니라 형태에 대한 민감도를 본다.
- probe는 선형이다. "안 읽힌다"는 "그 레이어에서 선형으로 안 읽힌다"는 뜻이다.

## 5. 진행 상황

- 2026-11-20: 1–5단계 코드와 노트북 완성. 로컬 CPU에서 작은 모델(Qwen2.5-0.5B, 사용자 60–150명)로 전체 흐름 동작 확인. 이 숫자는 의미 없음.
- 다음: vast.ai에서 7B 모델로 전체 1,500명 실행 → 결과 해석 → CUAI 발표.
