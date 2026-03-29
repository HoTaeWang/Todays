# NLTK 설치 및 환경구성



**Day 1-2: NLTK 설치 및 환경 구성**

- NLTK 라이브러리 설치
- 필수 데이터셋 다운로드 (`nltk.download()`)
- Jupyter Notebook 또는 IDE 세팅

```
import nltk

nltk.download()
```



## 📥 **NLTK 데이터 다운로드 완벽 가이드**

### **방법 1: GUI 다운로더 사용 (추천 - 초보자용)**



python

~~~python
import nltk
nltk.download()
```

이 명령을 실행하면 **NLTK Downloader** 창이 나타납니다. 창의 구조는 다음과 같습니다:
```
┌─────────────────────────────────────────────┐
│  NLTK Downloader                            │
├─────────────────────────────────────────────┤
│  Collections  │  Corpora  │  Models  │ All  │
├─────────────────────────────────────────────┤
│  ☐ abc                                      │
│  ☐ alpino                                   │
│  ☐ averaged_perceptron_tagger              │
│  ☐ basque_grammars                         │
│  ...                                        │
├─────────────────────────────────────────────┤
│  [Download]  [Cancel]                       │
└─────────────────────────────────────────────┘
~~~

### **필수 다운로드 항목 (학습 계획용)**

#### **🎯 초급 단계 (Level 1-2) - 필수 다운로드**

1. book

    (전체 책 컬렉션) 

   - 체크박스 찾기: Collections 탭 → `book` 선택
   - 용도: NLTK 튜토리얼의 모든 샘플 텍스트 포함
   - 크기: ~10MB

2. stopwords

    (불용어 리스트) 

   - Corpora 탭 → `stopwords` 선택
   - 용도: 영어 및 다국어 불용어 제거
   - 크기: ~125KB

3. punkt

    (문장/단어 토크나이저) 

   - Models 탭 → `punkt` 선택
   - 용도: 문장과 단어 분리
   - 크기: ~13MB

4. averaged_perceptron_tagger

    (품사 태거) 

   - Models 탭 → `averaged_perceptron_tagger` 선택
   - 용도: 품사(POS) 태깅
   - 크기: ~6MB

#### **🎯 중급 단계 (Level 3-4) - 추가 다운로드**

1. wordnet

    (어휘 데이터베이스) 

   - Corpora 탭 → `wordnet` 선택
   - 용도: 단어 의미, 동의어, 상하위어 관계
   - 크기: ~10MB

2. maxent_ne_chunker

    (개체명 인식 청커) 

   - Models 탭 → `maxent_ne_chunker` 선택
   - 용도: 인명, 지명, 조직명 추출
   - 크기: ~1.5MB

3. words

    (영어 단어 리스트) 

   - Corpora 탭 → `words` 선택
   - 용도: 철자 검사, 단어 유효성 확인
   - 크기: ~550KB

4. treebank

    (구문 분석 코퍼스) 

   - Corpora 탭 → `treebank` 선택
   - 용도: 구문 트리 학습
   - 크기: ~3MB

#### **🎯 고급 단계 (Level 5) - 선택 다운로드**

1. movie_reviews

    (영화 리뷰 데이터) 

   - Corpora 탭 → `movie_reviews` 선택
   - 용도: 감성 분석 실습
   - 크기: ~3MB

2. reuters

    (로이터 뉴스 코퍼스) 

   - Corpora 탭 → `reuters` 선택
   - 용도: 텍스트 분류 실습
   - 크기: ~8MB

3. brown

    (브라운 코퍼스) 

   - Corpora 탭 → `brown` 선택
   - 용도: 다양한 장르의 텍스트 분석
   - 크기: ~4MB

------

### **방법 2: 개별 다운로드 (프로그래밍 방식)**

GUI 대신 코드로 직접 다운로드할 수 있습니다:



python

```python
import nltk

# 필수 패키지 한번에 다운로드
essential_packages = [
    'book',              # NLTK 책 데이터
    'punkt',             # 토크나이저
    'stopwords',         # 불용어
    'averaged_perceptron_tagger',  # 품사 태거
    'wordnet',           # 워드넷
    'maxent_ne_chunker', # 개체명 인식
    'words',             # 단어 리스트
]

for package in essential_packages:
    nltk.download(package)
```

### **방법 3: 전체 다운로드 (빠른 설정)**

모든 것을 한번에 다운로드 (약 3GB, 시간 소요):



python

```python
import nltk
nltk.download('all')
```

⚠️ **주의**: 용량이 크므로 필요한 것만 선택하는 것을 추천합니다.

------

### **방법 4: 특정 카테고리만 다운로드**



python

```python
import nltk

# 초보자용 필수 패키지만
nltk.download('popular')  # 약 100MB, 가장 많이 사용되는 패키지들

# 또는 책 관련만
nltk.download('book')
```

------

## 🔧 **다운로드 위치 확인**

다운로드된 데이터의 저장 위치 확인:



python

```python
import nltk
print(nltk.data.path)
```

일반적인 경로:

- **Windows**: `C:\Users\사용자명\AppData\Roaming\nltk_data`
- **Mac/Linux**: `/home/사용자명/nltk_data` 또는 `/usr/share/nltk_data`

------

## 📋 **Day 1 체크리스트**

실제로 설치할 때 다음 순서로 진행하세요:



python

```python
# 1. NLTK 설치 확인
import nltk
print(f"NLTK 버전: {nltk.__version__}")

# 2. 필수 패키지 다운로드
packages = ['punkt', 'stopwords', 'book', 'averaged_perceptron_tagger']

for pkg in packages:
    try:
        nltk.download(pkg)
        print(f"✓ {pkg} 다운로드 완료")
    except Exception as e:
        print(f"✗ {pkg} 다운로드 실패: {e}")

# 3. 설치 확인
from nltk.book import *
print("✓ NLTK book 데이터 로드 성공!")
```

------

## 🎯 **Day 1 미션 완료 조건**

-  Python 설치 확인 (3.7 이상)
-  NLTK 라이브러리 설치
-  필수 패키지 4개 다운로드 (punkt, stopwords, book, averaged_perceptron_tagger)
-  `from nltk.book import *` 성공적으로 실행
-  `text1` 출력해보기

------

## 💡 **문제 해결 팁**

### **문제 1: 다운로드 창이 안 뜰 때**



python

```python
# 직접 다운로드 디렉토리 지정
import nltk
nltk.download('punkt', download_dir='D:/nltk_data')
```

### **문제 2: SSL 인증서 오류**



python

```python
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
```

### **문제 3: 프록시 환경에서 다운로드**

GUI 다운로더 대신 수동으로 다운로드:

1. https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip 접속
2. 파일 다운로드
3. `C:\Users\사용자명\AppData\Roaming\nltk_data\tokenizers\` 폴더에 압축 해제





**Day 3-5: 텍스트 데이터 다루기**

- `nltk.book` 모듈 활용
- Concordance, Similar words 함수
- Dispersion plot 시각화



```
from nltk.book import *

print(text1)
```

