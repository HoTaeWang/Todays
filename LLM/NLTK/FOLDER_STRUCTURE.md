# NLTK 폴더 구조 문서

**경로:** `LLM/NLTK/`  
**목적:** NLTK(Natural Language Toolkit) 기반 자연어 처리 학습 및 실습

---

## 1. 폴더 트리

```
NLTK/
├── 00_NLTK_downloader.ipynb      # NLTK 데이터 다운로드
├── 00_setups.ipynb               # 환경 설정 (경로, punkt 등)
├── 00_tokenizationInNLP.ipynb    # NLP에서의 토큰화 개요
├── 01_tokenize_nltk.ipynb        # NLTK 토큰화 실습 (word_tokenize 등)
├── 01_translator_example.ipynb   # 번역 예제
├── 02_perceptron_usingPytorch.ipynb  # PyTorch 퍼셉트론
├── 03_predict_result.ipynb       # 예측 결과 실습
├── 04_frequency_distribution.ipynb   # 빈도 분포
├── 04_text_summarization.ipynb   # 텍스트 요약
├── nltk_utils.py                 # 공통 유틸 (FreqDist, stopwords 등)
├── HowToInstall_NLTKdata.md      # NLTK 데이터 설치 가이드
├── wordnet-Sample_Usage.md       # WordNet 사용법 요약
├── README.md                     # NLTK 설정 및 NLTK_DATA 안내
└── FOLDER_STRUCTURE.md           # 본 문서 (폴더 구조 정리)
```

---

## 2. 파일 종류별 요약

| 종류 | 개수 | 확장자 | 설명 |
|------|------|--------|------|
| Jupyter 노트북 | 9 | `.ipynb` | NLTK·PyTorch 실습 및 예제 |
| Python 모듈 | 1 | `.py` | `nltk_utils` — 빈도·스톱워드 등 |
| Markdown 문서 | 4 | `.md` | 설치, WordNet, README, 구조 문서 |

---

## 3. 노트북별 설명

### 3.1 설정 및 기초 (00_*)

| 파일 | 내용 |
|------|------|
| **00_NLTK_downloader.ipynb** | NLTK 코퍼스·모델·토크나이저 등 데이터 다운로드 |
| **00_setups.ipynb** | NLTK 사용 전 환경 설정 (경로, `punkt` 등) |
| **00_tokenizationInNLP.ipynb** | NLP에서 토큰화의 역할, `word_tokenize` 예제 |

### 3.2 토큰화·번역 (01_*)

| 파일 | 내용 |
|------|------|
| **01_tokenize_nltk.ipynb** | `word_tokenize`, `sent_tokenize` 등 NLTK 토큰화 실습 |
| **01_translator_example.ipynb** | 번역 파이프라인 또는 NLTK 기반 번역 예제 |

### 3.3 모델·예측 (02_, 03_)

| 파일 | 내용 |
|------|------|
| **02_perceptron_usingPytorch.ipynb** | PyTorch로 퍼셉트론 구현·학습 |
| **03_predict_result.ipynb** | 학습된 모델로 예측 및 결과 확인 |

### 3.4 분석·요약 (04_*)

| 파일 | 내용 |
|------|------|
| **04_frequency_distribution.ipynb** | `FreqDist`로 단어 빈도 분포 분석 |
| **04_text_summarization.ipynb** | 텍스트 요약 실습 |

---

## 4. Python 모듈: `nltk_utils.py`

**역할:** NLTK 관련 공통 함수 모음

| 함수 | 설명 |
|------|------|
| `unique_word_count(freq_dist)` | `FreqDist`에서 고유 단어 목록 반환 및 개수 출력 |
| `remove_english_stopwords(freq_dist)` | 영어 스톱워드·구두점 제거 후 단어 목록 반환 |

**사용 예:** `04_frequency_distribution.ipynb` 등에서 `FreqDist` 기반 분석 시 import하여 사용.

---

## 5. Markdown 문서

| 파일 | 내용 |
|------|------|
| **README.md** | NLTK 설정 요약, `NLTK_DATA` 경로, 심볼릭 링크 및 `nltk.data.path` 설정 |
| **HowToInstall_NLTKdata.md** | NLTK 데이터 설치: 대화형/CLI/수동 설치, 프록시 설정 |
| **wordnet-Sample_Usage.md** | WordNet 개념(Synset, Hypernym, Holonym 등) 및 Python 사용 예 |
| **FOLDER_STRUCTURE.md** | 본 문서 — NLTK 폴더 구조 및 파일 설명 |

---

## 6. 권장 실행 순서

1. **설치·설정:** `HowToInstall_NLTKdata.md` → `00_NLTK_downloader.ipynb` → `00_setups.ipynb`
2. **개념:** `00_tokenizationInNLP.ipynb`
3. **토큰화:** `01_tokenize_nltk.ipynb`
4. **번역·분석·요약:** `01_translator_example.ipynb`, `04_frequency_distribution.ipynb`, `04_text_summarization.ipynb`
5. **모델:** `02_perceptron_usingPytorch.ipynb` → `03_predict_result.ipynb`

---

## 7. 상위 폴더(LLM)와의 관계

- **위치:** `Todays/LLM/NLTK/`
- **LLM 폴더:** 토큰화(BPE, tiktoken), 어텐션, Transformer, Gensim, PyTorch 기초 등 다양한 NLP/LLM 학습 자료 포함.
- **NLTK 역할:** 전통적인 NLP(토큰화, WordNet, 빈도 분석, 텍스트 요약) 및 PyTorch 퍼셉트론 입문.

자세한 LLM 프로젝트 구조는 `../README.md` 및 `../LLM-Summary.md` 참고.

---

## 8. 의존성 및 참고

- **Python 패키지:** `nltk`, `torch` (퍼셉트론·예측 노트북)
- **NLTK 데이터:** `punkt`, `punkt_tab`, `stopwords`, `wordnet` 등 (노트북에서 사용하는 항목 기준)
- **설치·경로:** `README.md`, `HowToInstall_NLTKdata.md`

---

*문서 작성: NLTK 폴더 구조 분석 기준.*
