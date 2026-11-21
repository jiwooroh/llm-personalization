# TransformerLens · Steering 실습 가이드

PRISM 프로젝트 RQ2(어느 레이어가 응답을 촉발하는가)·RQ3(어떻게 개입할 것인가)에 들어가기 전, 도구 감을 잡기 위한 짧은 실습. 상위 노트 → [[PRISM 개인화 이득]] · 방법론 조사 → [[03_지표·Steering 방법론 조사-개인화 지표 제안 · Steering 방법론 · 개입 시점 조사]]

## 실습 파일

위치: `~/Desktop/STUDY/Data Projects/llm-personalization/notebooks/`

| 순서 | 노트북 | 내용 | 프로젝트 연결 |
|---|---|---|---|
| 1 | `01_transformer_lens_basics.ipynb` | 모델 로드·토큰화 → `run_with_cache` → logit lens → induction head 찾기 → head ablation(hook) → activation patching (IOI) | RQ2: 레이어별 정보 형성, 인과적 개입 지점 |
| 2 | `02_activation_steering.ipynb` | ActAdd(Love−Hate) → CAA(감정 대조 쌍 24개, diff-in-means) → 벡터 진단(held-out 분리도, 레이어 간 cosine) → 레이어 sweep(효과 vs KL) → α sweep(효과 vs perplexity) → 생성 비교 | RQ1 trade-off 곡선, RQ3 개입 레이어 선택 |

- 모델: GPT-2 small (124M). Colab GPU 권장, CPU로도 돔
- 2번 노트북의 축은 **감정(positive↔negative)** — GPT-2 base에서도 효과가 눈에 보이는 축으로 파이프라인부터 익히고, 그다음 대조 쌍만 PRISM 고득점/저득점 응답으로 교체 (노트북 7절에 교체 표 있음)
- [!] **검증 범위**: 두 노트북 모두 TransformerLens 4.3.0 + torch 2.14에서 처음부터 끝까지 에러 없이 도는 것까지 확인함. 단, 클라우드 샌드박스에서 HuggingFace가 막혀 있어 **무작위 초기화한 가짜 GPT-2로 실행 경로만 검증** — 실제 수치·생성 결과는 Colab에서 처음 확인하게 됨. 특히 2번의 강도(`LAYER_SWEEP_COEFF`, `COEFFS`)는 보정 안 된 값이라 결과 보고 조정 필요

## 환경 설정

```python
!pip install -q "transformer-lens==4.3.0"
from transformer_lens.model_bridge import TransformerBridge
model = TransformerBridge.boot_transformers("gpt2", device="cuda")
model.enable_compatibility_mode()   # HookedTransformer와 같은 수치(LN folding, weight centering) + 구버전 훅 이름
```

> [!warning] TransformerLens 4.0 API 변경 (2026-09-28 처음 확인, 10-26 재확인: 최신 4.3.0)
> - `HookedTransformer`는 4.0에서 **삭제**됨 → `TransformerBridge.boot_transformers(name)` + `enable_compatibility_mode()`
> - ARENA, Neel Nanda Main Demo, CAA repo, emo-llm 등 기존 튜토리얼·코드는 대부분 구버전 → 그대로 돌리려면 `pip install "transformer-lens<4"` 환경 따로 (emo-llm은 `==2.7.0` pin)
> - compatibility mode에선 훅 이름(`blocks.{L}.hook_resid_pre`, `blocks.{L}.attn.hook_z` 등), `run_with_cache`, `run_with_hooks`, `model.hooks(...)`, `generate` 모두 구버전과 같은 방식으로 쓸 수 있음 (실습 노트북에서 확인)
> - compatibility mode 안 켜면 raw HF 가중치 그대로 → 생성·loss는 같지만 logit lens·residual norm 비교 값이 달라짐

| 구버전 (≤3.x) | 4.x |
|---|---|
| `HookedTransformer.from_pretrained("gpt2")` | `TransformerBridge.boot_transformers("gpt2")` + `.enable_compatibility_mode()` |
| `transformer_lens.utils` | `transformer_lens.utilities` |
| `HookedTransformerConfig` | `TransformerBridgeConfig` |

## 핵심 개념 (실습하며 확인할 것)

- **Residual stream**: 각 블록이 읽고 더하는 공용 통로. steering 벡터를 더하는 곳 (`hook_resid_pre`)
- **HookPoint / hook 함수**: `fn(activation, hook) -> activation`. 읽기(캐싱)와 쓰기(개입)가 같은 메커니즘 — ablation, patching, steering 모두 hook 하나
- **Logit lens**: 중간 레이어 residual에 최종 LN + unembed를 바로 적용 → "그 레이어에서 멈췄다면 뭘 예측했나"
- **Activation patching**: corrupted 실행에 clean activation을 (레이어, 위치) 단위로 끼워 넣어 회복 정도 측정 → probing("정보가 있다")과 달리 **인과적** 증거
- **ActAdd**: 프롬프트 쌍 1개의 activation 차이를 더함. 데이터 불필요, 잡음 큼
- **CAA**: 대조 쌍 N개의 평균 차이(diff-in-means). 쌍 수가 늘수록 표면 정보 상쇄. PRISM의 같은-prompt 고/저득점 응답 쌍과 구조가 맞아서 팀이 채택 (2026-09-28)
- **분리도 ≠ steerability**: 방향이 잘 분리돼도 그 방향으로 밀었을 때 행동이 바뀐다는 보장은 없음 → 2번 노트북 3절 vs 4절에서 따로 확인

## 참고 자료

### 튜토리얼 (실습 순서대로)
- [TransformerLens 공식 문서](https://TransformerLensOrg.github.io/TransformerLens/) · [v4 마이그레이션 가이드](https://TransformerLensOrg.github.io/TransformerLens/content/migrating_to_v4.html) · [GitHub](https://github.com/TransformerLensOrg/TransformerLens) (`demos/` 폴더)
- [Main Demo 노트북 (Neel Nanda)](https://neelnanda.io/transformer-lens-demo) — 라이브러리 기능 전체 투어. 구버전 API
- [ARENA Chapter 1: Transformer Interpretability](https://learn.arena.education/chapter1_transformer_interp/) — 표준 커리큘럼. 특히 [1.2 Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) (TransformerLens 기초, induction heads), 선택 세트 중 **IOI**(patching), **Function vectors & model steering**, **Steering vectors in GPT2-XL** · [ARENA GitHub](https://github.com/callummcdougall/ARENA_3.0)

### Steering 핵심 논문
- Turner et al., [Steering Language Models With Activation Engineering (ActAdd)](https://arxiv.org/abs/2308.10248) — 프롬프트 쌍 1개, GPT-2 XL "Love−Hate" 예시의 원조
- Rimsky (Panickssery) et al., [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) (ACL 2024) — CAA. 코드: [nrimsky/CAA](https://github.com/nrimsky/CAA)
- Zou et al., [Representation Engineering](https://arxiv.org/abs/2310.01405) — 상위 프레임워크
- [Understanding Unreliability of Steering Vectors in Language Models](https://arxiv.org/abs/2602.17881) — steering이 언제 안 먹히는지 (한계 서술 근거로)
- 개입 레이어: [Activation-Space Personality Steering: Hybrid Layer Selection](https://arxiv.org/html/2511.03738) — 2번 노트북의 레이어 sweep(KL)·α 한계선 아이디어의 출처
- 그 외 SAE vs diff-in-means 비교 등은 [[03_지표·Steering 방법론 조사-개인화 지표 제안 · Steering 방법론 · 개입 시점 조사]] 참고문헌 참조

### 라이브러리 (TransformerLens 대신/함께)
- [steering-vectors](https://pypi.org/project/steering-vectors) — HF 모델에 CAA류 steering을 몇 줄로. 실험 규모 키울 때 후보
- [repeng](https://pypi.org/project/repeng) — control vector(RepE 계열) 학습·적용
- PRISM 실험에서 instruct 모델(Qwen2.5-0.5B/1.5B-Instruct, Llama-3.2-1B-Instruct)로 넘어갈 때, TransformerBridge가 지원하면 같은 코드 재사용 / 아니면 위 라이브러리로

## 진행 체크리스트

- [ ] `01_transformer_lens_basics.ipynb` Colab 실행 — logit lens에서 " Paris"가 몇 번째 레이어부터 top-1인지, induction head가 알려진 5.1/5.5/6.9/7.2/7.10과 맞는지 확인
- [ ] `02_activation_steering.ipynb` Colab 실행 — α 값 보정, 선택된 레이어 기록
- [ ] 2번 노트북 연습 과제 2(쌍 수 줄이기) → PRISM 군집별 쌍 수가 적을 때 위험도 감 잡기
- [ ] 대조 쌍을 PRISM 고/저득점 응답으로 교체 (`prism_caa_test.ipynb`의 쌍 생성 코드 재사용), 모델을 instruct 모델로 교체
- [ ] 결과 요약을 이 노트 아래 "실행 로그"에 추가

## 실행 로그

- 2026-10-26: 실습 노트북 2개 제작 (TransformerLens 4.3.0 기준). 가짜 GPT-2로 실행 경로 검증 완료, 실제 결과 미확인

---
*2026-10-26 작성*
