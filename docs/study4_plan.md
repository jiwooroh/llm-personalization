# Study 4 — 명시적 선호 vs 암묵적 선호: LLM은 "알면서 말하지 않는가"

*2026-11-22 작성. GPU 세션 1회로 실행할 수 있게 코드·검증까지 끝낸 상태.*

## 질문

사용자가 선호를 **직접 말할 때**("I love surfing")와 **대화 속에 은근히 드러낼 때**(바닷가 서핑 스팟에 대해 묻는 대화)
LLM의 개인화가 어떻게 다른가?

1. **Says**: 나중 질문에서 모델이 그 선호에 맞는 답을 고르는가 (명시 vs 암묵)
2. **Knows**: 모델 내부 표상에는 그 선호가 반영되는가 (명시 vs 암묵)
3. **Gap**: 암묵적 선호는 "내부엔 있는데 답에는 안 쓰는" 정도(knows − says)가 명시보다 큰가
4. **Distance**: 사이에 다른 대화가 끼면(multi-turn 거리) 명시·암묵 중 무엇이 더 빨리 사라지나
5. **Same direction?**: 명시 선호로 학습한 probe가 암묵 선호에도 통하는가 (같은 표상을 쓰나)

## 왜 이 질문인가 (근거)

- **우리 Study 1**: 모델은 사용자의 자연어 선호를 내부에서 더 많이 안다 (probe ρ 0.21 vs 말 0.12, Qwen·Llama).
  진단 결과, 말은 **글에 명시된 속성만** 반영하고 언급 안 된 속성은 크게 낮게 매겼다 (언급 여부 상관: 사람 0.06, Llama 말 0.36),
  반면 probe는 언급 여부와 무관하게 유지됐다 (0.20 vs 0.20). → "명시된 것만 말한다"는 가설.
  Study 4는 이 가설을 **같은 선호를 명시/암묵 두 형태로 준 통제 실험**으로 직접 검증한다.
- **인지·언어 근거**: 사람은 말하지 않은 것을 함축(Grice)과 공통 기반으로 채운다. 암묵적 정보는 이해 과정에서 추론되지만 명시 정보보다 덜 접근 가능하다는 담화 이해 연구와 연결된다(이론적 배경, 2025 이전 고전).
- **선행연구 (2025+)**
  - Kantharuban et al. 2025 (*Findings of ACL*): 명시/암묵 **정체성** 단서가 추천을 바꾼다. 출력만 분석, 내부 표상·개인 선호 충돌은 미검증.
  - Neplenbroek et al. 2025 (*EMNLP*): 대화 속 고정관념 단서가 **사용자 정체성 표상**(probe)을 바꾼다.
  - Bouchaud & Ramaciotti 2025 (arXiv 2512.10065): 명시 공개로 학습한 정체성 probe가 암묵 단서에도 통한다.
  - Yan et al. 2026 (arXiv 2608.11735): 암묵 개인화의 활성 방향을 찾아 제거.
  - **PersonaMem-v2** (2025, arXiv 2512.06688): 암묵적 선호 개인화에서 최신 모델도 37–48%. **출력만** 평가.
- **빈 곳 = 이 연구의 새로움**: 위 연구들은 (a) **정체성**의 명시/암묵을 내부까지 보거나, (b) **선호**의 암묵 개인화를 **출력**으로만 본다.
  **같은 선호 내용을 명시/암묵 두 형태로 통제해서, 내부(knows)와 출력(says)을 함께, multi-turn 거리까지** 본 연구는 찾지 못했다.

## 피봇 기록 (오늘 대화에서 무엇이 왜 바뀌었나)

| 순서 | 계획 | 바꾼 이유 |
|---|---|---|
| 1 | 말·숫자·행동 + Big Five (Twin-2K-500) | 사용자: 핵심 질문은 "왜 아는데 말 못 하나" → 그 원인 분석으로 전환 |
| 2 | "아는데 말 못 함"의 원인 분석 (PRISM) | Mac 진단으로 "명시된 것만 말한다"(문자주의) 근거 확보. 사용자: 결론이 마음에 안 들고, **명시 vs 암묵 요구에 따른 개인화 차이**를 보고 싶음 |
| 3 | 단어 정의 차이 / multi-turn (PRISM) | PRISM 확인 결과: 모델이 사용자 요구를 받지 않았고, 50단어 제한, 후속 정정 발화 ~1% → multi-turn 분석 불가. 사용자: **데이터·참고문헌 기준은 2025년 이후** → PRISM(2024) 제외 |
| 4 | 인지심리 관점 5개 옵션 → 옵션 4 (고정관념 vs 개인 선호, Kantharuban 확장) | 문헌 확인: Neplenbroek 2025, Bouchaud 2025, Yan 2026이 **정체성 명시/암묵의 내부 표상**을 이미 다룸. 사용자: **정체성 고정관념은 NLP에서 벗어남** |
| 5 | **현재: 같은 선호의 명시 vs 암묵 표현 (화용론적 함축), knows vs says, multi-turn 거리** | NLP 중심(화용·함축), Study 1 진단 결과를 직접 검증, 2025 데이터(PersonaMem-v2, CC BY 4.0)에 명시 문장·암묵 대화·정답/오답 선택지가 모두 있음, GPU 1회로 끝남 |

## 원인 검증 (사용자 요청: "왜 알면서 안 쓰나")

| 원인 후보 | 조건 / 분석 | 원인이라면 |
|---|---|---|
| R1 귀속 실패: 대화 내용을 "이 사람의 선호"로 연결하지 않음 | `implicit_attr`: 암묵 대화 + "that's something I personally care about" (내용 없음) | 내용 없이 귀속만 해도 says 상승 |
| R2 인출 실패: 답할 때 앞 대화를 스스로 꺼내 보지 않음 | `implicit_recall`: 모델이 먼저 "내 선호에 대해 아는 것"을 요약 후 선택 | 스스로 떠올리면 says 상승 (요약이 선호를 언급한 비율도 기록) |
| R3 출력 경로 손실 | logit lens: A–D 선택을 레이어마다 읽음 | 중간층 정답률 > 최종 |
| R4 사전 선호와 경쟁 | 문맥 없이도 오답을 강하게 선호하는 문항에서 실패 집중 여부 | 사전 오답 선호가 강할수록 implicit 실패 |
| (기준선) 표면 유사도 | `embed_baseline.py` (mpnet): explicit 0.87, implicit 0.54 | LLM 결과는 임베딩이 틀린 문항에서 따로 보고 |

## 데이터

**PersonaMem-v2** (arXiv 2512.06688, HF `bowen-upenn/PersonaMem-v2`, CC BY 4.0) `benchmark/text/benchmark.csv`
- 사용: `who=self`, `updated=False`, `pref_type ∈ {neutral, stereotypical, anti_stereotypical}` → 2,020행, 페르소나 200명
- 각 행: `preference`(명시 문장), `related_conversation_snippet`(암묵 대화, 2–6턴), `user_query`, 정답 1 + 오답 3
- 확인: 선호 문장이 암묵 대화에 **그대로 나오는 경우 0%** (누수 없음)
- 무작위 800개 (seed 0, 195명)

## 설계

| 조건 | 질문 앞에 오는 대화 |
|---|---|
| none | 없음 (선택지 글자만으로 맞힐 수 있는 정도 = 기준선) |
| explicit | "A quick note about me: {preference}." + 짧은 응답 |
| implicit | PersonaMem-v2 대화 (assistant 답은 80단어로 자름) |
| explicit_far / implicit_far | 위 + 다른 페르소나·다른 주제 대화 2쌍 (거리) |

측정 (모두 정답률, 우연 0.25):
- **says_mc**: 4개 답을 A–D로 보여주고 고르게 함. 4번 회전해 위치 편향 상쇄
- **says_gen**: 각 답을 assistant 턴에 넣었을 때 토큰 평균 log-prob가 문맥으로 얼마나 오르나 (LP[cond] − LP[none])
- **knows**: 각 답의 마지막 토큰 hidden state에서 none 조건을 뺀 차이(문맥이 바꾼 부분)로 선형 probe.
  PCA 64 + 로지스틱, 페르소나 단위 5-fold, 내부 CV로 C 선택. 비교 레이어 = explicit 최적 레이어(사전 규칙)
- **transfer**: explicit로 학습한 probe를 implicit에 적용
- 검정: 페르소나 단위 paired bootstrap 2,000회

## 미리 정한 해석

| 결과 | 결론 |
|---|---|
| says: explicit > implicit, knows: explicit ≈ implicit | **암묵 선호는 표상되지만 답에 안 쓰인다** (Study 1 문자주의의 직접 증거, 가장 좋은 결과) |
| says·knows 모두 explicit > implicit | 암묵 선호는 애초에 덜 추론된다 (읽기 단계의 한계) |
| gap(implicit) > gap(explicit) 유의 | 함축 정보의 "아는데 안 씀"이 더 크다 |
| transfer ≈ within | 명시·암묵 선호가 같은 표상 방향을 공유 |
| far에서 implicit이 더 크게 하락 | 암묵 선호는 대화가 길어지면 더 빨리 잊힌다 |
| 고/저 lexical overlap에서 explicit 우위 유지 | explicit 우위가 단어 겹침 때문만은 아님 |

## 한계 (미리 적어둠)

- PersonaMem-v2 대화·답은 LLM이 생성한 합성 데이터 (실제 사용자 아님)
- 명시 문장은 템플릿("A quick note about me: Attends local food truck festivals.")이라 문법이 어색할 수 있음
- 정답 답변은 선호를 단어로 언급하는 경우가 많아 explicit 조건에서 단어 겹침이 유리 → overlap 분할로 점검
- 모델 2개 (Qwen2.5-7B, Llama-3.1-8B)

## 실행

```bash
read -s HF_TOKEN && export HF_TOKEN                 # Llama용
PYTHONUNBUFFERED=1 bash gpu_session_study4.sh 2>&1 | tee study4.log
```
- 디스크 60GB 이상 권장 (모자라면 두 모델 사이에 캐시 자동 삭제)
- 각 모델 전에 "capital of France" 확인 (깨진 가중치 방지)
- 결과: `study4_results.tar.gz` (hidden state `H.npy` 제외), `runs/implicit/<model>/summary.json`
- 예상 시간: 모델당 약 1–1.5시간 (800 문항 × 5 조건 × 8 시퀀스)

## 파일럿 (Mac, Qwen2.5-1.5B, 40문항, 객관식만, 11/22)

| 조건 | says_mc | 임베딩 기준선 |
|---|---|---|
| none | 0.28 | — |
| explicit | 0.83 | 0.87 |
| implicit | 0.55 | 0.54 |
| implicit_attr (R1) | 0.58 | — |
| implicit_recall (R2) | 0.58 | — |

- 임베딩이 틀리는 암묵 문항(11개)에서도 implicit 0.45 > 우연 0.25 → 표면 유사도 이상을 조금은 함
- **R1 귀속·R2 인출로는 거의 안 오름** (+0.03)
- **self-recall 문장에서 모델이 "As an AI, I don't have access to your personal information"이라며 거부하거나, 선호가 아닌 글쓰기 스타일을 요약**하는 경우가 많음 → 대화 내용을 "사용자에 대한 정보"로 취급하지 않는 경향 (R1의 언어 버전)
- logit lens: 정답 선호는 20층 이후에야 생기고, 마지막 층까지 줄지 않음 → 1.5B에서는 R3(늦은 손실) 근거 없음
- 해석 주의: 1.5B·40문항이라 잡음이 크고, knows(probe)는 측정하지 않음

**방향 판단**: 피봇하지 않는다. 다만 핵심 결론 후보를 조정한다.
- "암묵 선호 개인화 ≈ 표면 유사도 수준"이 큰 모델에서도 유지되는지 (LLM vs 임베딩, 임베딩-어려운 문항)
- 내부(knows)가 그보다 높은지 = "알면서 안 쓰나"의 직접 검증
- self-recall이 선호를 언급하는 비율과 그때의 선택 정확도 (모델이 말로 인정하는 사용자 정보 vs 선택에 쓰는 정보)
- lens 분석의 동점(초기층 평평한 로짓) 처리 버그 수정함

## Pivot 6 (final scope, 2026-11-22): drop the mechanism, keep the behavioral result

**Why:** the knows-vs-says design could not show that preferences vanish *in written replies* (it scores fixed options), the probe cannot separate "knows" from reply-context fit, and nothing was causal. Making it convincing needed free generation + a judge + steering (more code, more GPU, more risk). The pilot already gives a clean behavioral story, so the study is cut to that.

**Question:** in multi-turn chat, how much do LLMs use a preference the user only implies, and is it more than surface similarity?

**Run:** `extract.py --mc_only --conds none,explicit,implicit,explicit_far,implicit_far` on Qwen2.5-7B and Llama-3.1-8B (about 1h total), analyzed by `analyze_mc.py`.

**Results to report:**
1. explicit vs implicit gap (and near vs far), compared with the 1.5B pilot
2. LLM vs embedding baseline, overall and on items the embedding gets wrong
3. which implicit preferences are lost: by distance, preference type, lexical overlap

Dropped: probe, logit lens, attribution / recall conditions, steering.

## Results (2026-11-22, 800 items, 195 personas, MC accuracy, chance 0.25)

| | none | explicit | implicit | explicit_far | implicit_far |
|---|---|---|---|---|---|
| Qwen2.5-7B | 0.23 | 0.92 | 0.51 | 0.83 | 0.41 |
| Llama-3.1-8B | 0.27 | 0.96 | 0.74 | 0.96 | 0.60 |
| embedding (mpnet) | — | 0.87 | 0.54 | 0.50 | 0.45 |

- Explicit − implicit: Qwen +0.40 [0.37, 0.44], Llama +0.23 [0.20, 0.26].
- vs embedding on implicit: Qwen −0.03 [−0.08, 0.02] (no better than similarity); Llama +0.19 [0.15, 0.23]. On items the embedding gets wrong: Qwen 0.46, Llama 0.65.
- Distance: implicit drops 0.11–0.14 with two filler exchanges; explicit drops 0.09 (Qwen) and 0.00 (Llama).
- Pref type (implicit): stereotypical > anti-stereotypical in both (Qwen 0.56 vs 0.47, Llama 0.80 vs 0.71).
- Lexical overlap: small effect (implicit hi vs lo +0.05 / +0.08).

## Follow-ups (2026-11-22)

- **Error analysis** (`docs/study4_error_analysis.md`): Llama wins where a concrete cue is buried inside an editing/translation task (the embedding fails there, Llama does not). Qwen loses to the embedding even on direct, lexically obvious cues, so its problem is not subtlety but not treating conversation content as information about the user. Stereotypical > anti-stereotypical is not significant once no-context accuracy is subtracted; it shows up only on items with no surface cue (small n).
- **Study 4b, scale sweep** (`gpu_session_study4_scale.sh`, 48GB GPU, ~1.5h): Qwen 1.5B/3B/14B and Llama-3.2 1B/3B, same items and conditions. If Qwen-14B stays near the embedding while small Llamas beat it, the difference is family (training), not size. Notebook section 7 plots it once results are in `runs/implicit/`.
