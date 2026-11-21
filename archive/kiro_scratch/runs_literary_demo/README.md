# Literary Taste Analysis

LiteraryTaste 데이터셋을 활용한 사용자별 문학 선호도 분석 프로젝트

## 📊 Dataset Overview

- **사용자 수**: 60명
- **총 판단 인스턴스**: 6,300개 (사용자당 105개)
- **텍스트 쌍 태스크**: 2,000개

### 선호도 분포
- Text A 선호: 2,908 (46.2%)
- Text B 선호: 2,836 (45.0%)
- 잘 모르겠음: 256 (4.1%)
- 자유 응답: 987 (15.7%)

## 📚 Task Types

| 유형 | 개수 | 설명 |
|------|------|------|
| `human_vs_human` | 1,400 | 사람이 쓴 텍스트 vs 사람이 쓴 텍스트 |
| `human_vs_llm_samecontent` | 400 | 사람 vs LLM (같은 내용) |
| `human_vs_llm_diffcontent` | 200 | 사람 vs LLM (다른 내용) |

## 📖 Text Sources

주요 텍스트 출처:
- **writingprompt**: 1,036 (Reddit r/WritingPrompts)
- **gutenberg**: 1,004 (Project Gutenberg 고전 문학)
- **sterman**: 961 (Sterman 코퍼스)
- **poetry**: 283 (시)
- **llm_***: LLM 생성 텍스트

## 🎭 Top Genre Preferences

1. **Contemporary / general fiction** (36명)
2. **Social & Cultural** (33명)
3. **Crime / mystery / thriller** (31명)
4. **History** (30명)
5. **Historical** (29명)
6. **Science fiction / fantasy** (29명)
7. **Personal Development** (29명)
8. **Classics** (28명)
9. **Science & Nature** (28명)
10. **Autobiography / biography / memoir** (27명)

## 📏 Text Length Statistics

- **평균 텍스트 길이**: ~690 characters
- **중앙값**: 771-772 characters
- **범위**: 48 - 1,439 characters

## 🔬 Analysis Pipeline (Planned)

### Phase 1: EDA ✅
- [x] 데이터 로드 및 구조 파악
- [x] 사용자 프로필 분석
- [x] 텍스트 특성 통계
- [x] 장르 선호도 분석

### Phase 2: TransformerLens Integration 🚧
- [ ] GPT-2 또는 Pythia 모델 로드
- [ ] 프롬프트 설계 (사용자 프로필 + 텍스트)
- [ ] 레이어별 활성화 추출
- [ ] 활성화 패턴 시각화

### Phase 3: Probing 🔜
- [ ] 사용자 특성 → 활성화 매핑
- [ ] Linear probe 학습
- [ ] 레이어별 정확도 분석
- [ ] 중요 뉴런 식별

### Phase 4: Steering 🔜
- [ ] 스티어링 벡터 생성
- [ ] 특정 취향으로 모델 조정
- [ ] 생성 텍스트 평가

## 📁 Files

```
runs/literary_taste/
├── README.md                 # This file
├── eda_summary.json          # EDA 요약 통계
└── (upcoming)
    ├── activations/          # 추출된 활성화
    ├── probes/               # 학습된 프로브
    └── figures/              # 시각화 결과
```

## 🚀 Quick Start

```bash
# 1. EDA 실행
python literary_taste_eda.py

# 2. 노트북 열기
jupyter notebook notebooks/04_literary_taste_analysis.ipynb

# 3. (다음 단계) TransformerLens 설치
pip install transformer-lens
```

## 📝 Research Questions

1. **사용자 특성이 LLM 내부에 어떻게 인코딩되는가?**
   - 어느 레이어에서 사용자 정보가 가장 강하게 나타나는가?
   - Attention 패턴은 어떻게 변화하는가?

2. **텍스트 선호도를 예측할 수 있는가?**
   - 활성화만으로 사용자의 선호를 예측 가능한가?
   - 어떤 특징이 가장 중요한가?

3. **모델을 특정 취향으로 조정할 수 있는가?**
   - Steering vector로 생성 스타일 변경 가능한가?
   - 변경 효과는 어느 정도인가?

## 📚 References

- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)
- [LiteraryTaste GitHub](https://github.com/mj-storytelling/LiteraryTaste)
- Original PRISM paper implementation in `../src/`

---

**Last Updated**: 2026-11-20
