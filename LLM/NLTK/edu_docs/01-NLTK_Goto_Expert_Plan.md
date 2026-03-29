# 🎮 **NLTK 전문가 되기 -  학습 플랜**

## 📊 **전체 개요**
- **총 학습 기간**: 8주 (56일)
- **일일 학습 시간**: 1-1.5시간
- **난이도 레벨**: 5단계 (Beginner → Expert)
- **최종 목표**: NLTK를 활용한 실전 NLP 프로젝트 완성

---

## 🎯 **5단계 학습 구조**

### **LEVEL 1: 환경 설정 & 기초 (Week 1-2)**
**목표**: NLTK의 기본 구조와 텍스트 처리 기초 마스터

#### 📚 **학습 내용**
- **Day 1-2: NLTK 설치 및 환경 구성**
  - NLTK 라이브러리 설치
  - 필수 데이터셋 다운로드 (`nltk.download()`)
  - Jupyter Notebook 또는 IDE 세팅

- **Day 3-5: 텍스트 데이터 다루기**
  - `nltk.book` 모듈 활용
  - Concordance, Similar words 함수
  - Dispersion plot 시각화

- **Day 6-8: 기본 텍스트 전처리**
  - Tokenization (문장, 단어)
  - Stop words 제거
  - 대소문자 정규화

- **Day 9-12: Corpus 이해하기**
  - Corpus의 개념
  - 내장 Corpus 탐색
  - Custom Corpus 만들기

- **Day 13-14: 레벨 1 보스전** 

#### 🎮 **미션 & 과제**

**미션 1-1: "텍스트 탐험가" (Day 3-5)**
```python
# 과제: 다음을 수행하는 코드 작성
# 1. text1에서 'whale'이 나오는 모든 문맥 찾기
# 2. text1과 text2에서 공통으로 나타나는 단어 5개 찾기
# 3. 'ocean'과 비슷한 맥락의 단어 찾기
```
**성공 기준**: 3가지 모두 정확히 출력

**미션 1-2: "나만의 텍스트 분석기" (Day 6-8)**
```python
# 과제: 좋아하는 소설/뉴스 기사 하나를 선택해서
# 1. 총 단어 수, 고유 단어 수 계산
# 2. 가장 빈번한 단어 top 10 추출
# 3. Stop words 제거 후 다시 top 10 추출
```
**성공 기준**: 전후 비교 리포트 작성

**보스전 1: "커스텀 텍스트 파이프라인 구축" (Day 13-14)**
- 자신이 선택한 텍스트(영문 500단어 이상)를 입력받아
- 자동으로 전처리(토큰화, 정규화, 불용어 제거)하고
- 통계 정보를 출력하는 프로그램 작성
- **보상**: Level 2 진입 + "텍스트 마스터" 뱃지 🏆

---

### **LEVEL 2: 형태소 분석 & 품사 태깅 (Week 3-4)**
**목표**: 단어의 구조와 문법적 역할 이해

#### 📚 **학습 내용**
- **Day 15-18: Stemming vs Lemmatization**
  - Porter Stemmer, Lancaster Stemmer
  - WordNet Lemmatizer
  - 차이점 비교 실습

- **Day 19-22: POS Tagging (품사 태깅)**
  - Penn Treebank 태그셋 학습
  - NLTK 태거 활용
  - 정확도 평가

- **Day 23-26: N-grams & Collocations**
  - Bigrams, Trigrams 추출
  - Collocation 찾기
  - PMI, Chi-square 통계

- **Day 27-28: 레벨 2 보스전**

#### 🎮 **미션 & 과제**

**미션 2-1: "어근 사냥꾼" (Day 15-18)**
```python
# 과제: 다음 단어들의 어간 추출 비교
words = ['running', 'runs', 'ran', 'easily', 'fairly']
# Porter, Lancaster, Lemmatizer 3가지로 처리하고
# 결과 차이를 표로 정리
```
**성공 기준**: 3가지 방법의 장단점 설명 포함

**미션 2-2: "품사 탐정" (Day 19-22)**
```python
# 과제: 다음 문장의 품사 태깅
sentence = "The quick brown fox jumps over the lazy dog"
# 1. 품사 태깅 수행
# 2. 명사만 추출
# 3. 동사만 추출
# 4. 형용사만 추출
```
**성공 기준**: 각 품사별 정확한 추출

**미션 2-3: "연어 발견가" (Day 23-26)**
```python
# 과제: 뉴스 기사에서
# 1. 가장 빈번한 bigram 10개
# 2. 가장 빈번한 trigram 10개
# 3. PMI 점수가 높은 collocation 추출
```
**성공 기준**: 의미있는 연어 발견 및 해석

**보스전 2: "스마트 텍스트 정규화 시스템"**
- 입력 텍스트의 모든 단어를 lemmatize하고
- 품사별로 분류하며
- 의미있는 단어 조합(bigram)을 추출하는 시스템 구축
- **보상**: "형태소 마스터" 뱃지 🏆 + Level 3 진입

---

### **LEVEL 3: 구문 분석 & 개체명 인식 (Week 5-6)**
**목표**: 문장 구조 파악과 정보 추출

#### 📚 **학습 내용**
- **Day 29-32: Chunking (구문 분석)**
  - Noun Phrase Chunking
  - Verb Phrase Chunking
  - 정규표현식 기반 Chunking

- **Day 33-36: Named Entity Recognition**
  - NER 개념과 활용
  - NLTK의 NER 시스템
  - Custom NER 규칙 작성

- **Day 37-40: Dependency Parsing**
  - 의존 구문 분석 기초
  - Parse Tree 시각화
  - 문장 구조 분석

- **Day 41-42: 레벨 3 보스전**

#### 🎮 **미션 & 과제**

**미션 3-1: "구문 채굴자" (Day 29-32)**
```python
# 과제: 뉴스 기사에서
# 1. 모든 명사구(NP) 추출
# 2. 모든 동사구(VP) 추출
# 3. 패턴 규칙을 만들어 "형용사+명사" 조합 찾기
```
**성공 기준**: 최소 20개 이상의 유의미한 구문 추출

**미션 3-2: "정보 사냥꾼" (Day 33-36)**
```python
# 과제: 뉴스 기사 3개를 분석해서
# 1. 모든 인명(PERSON) 추출
# 2. 모든 조직명(ORGANIZATION) 추출
# 3. 모든 지명(LOCATION) 추출
# 4. 빈도수 상위 5개씩 정리
```
**성공 기준**: 정확도 80% 이상

**보스전 3: "자동 뉴스 요약기"**
- 뉴스 기사를 입력받아
- 주요 인물, 장소, 조직을 자동 추출하고
- 핵심 명사구를 기반으로 3줄 요약 생성
- **보상**: "정보 추출 마스터" 뱃지 🏆 + Level 4 진입

---

### **LEVEL 4: 의미 분석 & 워드넷 (Week 6-7)**
**목표**: 단어의 의미와 관계 이해

#### 📚 **학습 내용**
- **Day 43-46: WordNet 활용**
  - Synsets 이해
  - Hypernyms, Hyponyms
  - Similarity 계산

- **Day 47-50: Semantic Similarity**
  - Path similarity
  - Wu-Palmer similarity
  - Lesk algorithm

- **Day 51-52: Sentiment Analysis 기초**
  - SentiWordNet 활용
  - 감성 점수 계산
  - 극성 분류

- **Day 53-54: 레벨 4 보스전**

#### 🎮 **미션 & 과제**

**미션 4-1: "의미망 탐험가" (Day 43-46)**
```python
# 과제: 'dog', 'cat', 'animal' 세 단어에 대해
# 1. 각각의 모든 synsets 나열
# 2. Hypernym 트리 시각화
# 3. 'dog'과 'cat'의 공통 상위어 찾기
```
**성공 기준**: 의미 관계도 작성

**미션 4-2: "유사도 측정기" (Day 47-50)**
```python
# 과제: 다음 단어 쌍의 유사도 계산
pairs = [('car', 'automobile'), ('car', 'bike'), ('car', 'tree')]
# 1. Path similarity
# 2. Wu-Palmer similarity
# 3. 결과 비교 분석
```
**성공 기준**: 유사도 점수 해석 포함

**미션 4-3: "감성 분석가" (Day 51-52)**
```python
# 과제: 영화 리뷰 10개를 수집해서
# 1. 각 리뷰의 긍정/부정 점수 계산
# 2. 전체적인 극성 분류 (긍정/부정/중립)
# 3. 정확도 측정
```
**성공 기준**: 분류 정확도 70% 이상

**보스전 4: "지능형 유사 문서 검색 시스템"**
- 문서 컬렉션에서 쿼리와 가장 유사한 문서 찾기
- WordNet 기반 의미 유사도 활용
- 검색 결과 상위 5개 반환
- **보상**: "의미 분석 마스터" 뱃지 🏆 + Level 5 진입

---

### **LEVEL 5: 고급 응용 & 최종 프로젝트 (Week 8)**
**목표**: 실전 프로젝트로 모든 기술 통합

#### 📚 **학습 내용**
- **Day 55-56: Classification 기초**
  - Naive Bayes Classifier
  - Feature Extraction
  - 모델 평가

- **Day 57-58: 최종 프로젝트 기획**
  - 프로젝트 주제 선정
  - 데이터 수집 계획
  - 시스템 설계

- **Day 59-62: 최종 프로젝트 구현**
  - 코드 작성
  - 테스트 및 디버깅
  - 성능 최적화

- **Day 63: 최종 보스전**

#### 🎮 **최종 미션**

**최종 보스: "NLP 통합 애플리케이션 개발"**

다음 중 하나를 선택하여 구현:

**옵션 A: 스마트 뉴스 분석기**
- 뉴스 기사 입력
- 자동 요약 (핵심 문장 3개)
- 주요 개체 추출 (인물, 장소, 조직)
- 감성 분석 (긍정/부정/중립)
- 관련 키워드 추출

**옵션 B: 텍스트 유사도 검사기**
- 두 문서의 유사도 측정
- 표절 의심 부분 하이라이팅
- 의미적 유사도 점수 제공
- 시각화 리포트 생성

**옵션 C: 챗봇 기본 시스템**
- 사용자 입력 이해 (의도 분류)
- 개체명 인식
- 적절한 응답 생성
- 대화 히스토리 관리

**성공 기준**:
- 모든 Level 1-4 기술 최소 1개 이상 활용
- 코드 실행 가능
- README 문서 작성
- 10분 데모 발표 준비

**최종 보상**: "NLTK 전문가" 인증서 🎓 + 포트폴리오 프로젝트 완성

---

## 📈 **성과 측정 시스템**

### **경험치(XP) 시스템**
- 일일 과제 완료: +10 XP
- 미션 완료: +50 XP
- 보스전 클리어: +100 XP
- 추가 도전 과제: +25 XP

### **레벨업 조건**
- Level 1→2: 200 XP
- Level 2→3: 400 XP
- Level 3→4: 600 XP
- Level 4→5: 800 XP
- 전문가 달성: 1000 XP

### **뱃지 컬렉션** 🏆
- 🥉 텍스트 마스터 (Level 1)
- 🥈 형태소 마스터 (Level 2)
- 🥇 정보 추출 마스터 (Level 3)
- 💎 의미 분석 마스터 (Level 4)
- 👑 NLTK 전문가 (Level 5)

---

## 🎯 **일일 학습 루틴 (1시간)**

**15분**: 이론 학습 (PDF 읽기 또는 문서 참조)  
**30분**: 코드 실습 (미션 수행)  
**10분**: 복습 및 노트 정리  
**5분**: 다음 날 계획 수립

---

## 📝 **학습 추적 템플릿**

매일 다음을 기록하세요:
```
날짜: ____
레벨: ____
오늘의 미션: ____
완료 여부: [ ]
어려웠던 점: ____
새로 배운 것: ____
누적 XP: ____
```

---

## 📚 **참고 자료**

### **필수 문서**
- NLTK.pdf - 메인 학습 교재
- NLTK-readthedocs-io-en-latest.pdf - 공식 문서
- python-3-text-processing-with-nltk-3-cookbook.pdf - 실전 레시피
- P07. Text ANALYTICS for Beginners using NLTK - DataCamp marked.pdf - 초보자 가이드
- Traditional NLP.md - 전통적 NLP 개념 정리

### **온라인 리소스**
- NLTK 공식 문서: https://www.nltk.org/
- NLTK Book: https://www.nltk.org/book/
- Python 공식 문서: https://docs.python.org/3/

---

## 💡 **학습 팁**

1. **매일 코딩하기**: 이론만 보지 말고 반드시 코드를 직접 작성하세요.
2. **작은 프로젝트 만들기**: 각 레벨에서 배운 내용으로 미니 프로젝트를 만드세요.
3. **커뮤니티 활용**: Stack Overflow, Reddit r/LanguageTechnology 활용
4. **실제 데이터 사용**: 뉴스, 트위터, 영화 리뷰 등 실제 데이터로 연습하세요.
5. **에러 두려워하지 않기**: 에러는 학습의 일부입니다. 디버깅 능력도 함께 키우세요.

---


