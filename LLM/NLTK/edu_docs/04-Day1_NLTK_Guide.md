# Day 1: NLTK 기초 탐험 - 완전 실습 가이드

## 🎯 학습 목표
1. nltk.book의 9개 텍스트 이해하기
2. Concordance, Similar, Common Contexts 함수 사용법 익히기
3. 텍스트 기본 통계 계산하기

---

## 📚 Step 1: 텍스트 목록 확인

```python
from nltk.book import *

# 사용 가능한 텍스트들
# text1: Moby Dick by Herman Melville 1851
# text2: Sense and Sensibility by Jane Austen 1811
# text3: The Book of Genesis
# text4: Inaugural Address Corpus
# text5: Chat Corpus
# text6: Monty Python and the Holy Grail
# text7: Wall Street Journal
# text8: Personals Corpus
# text9: The Man Who Was Thursday by G. K. Chesterton 1908

# 확인해보기
print(text1)
print(text2)
```

**출력 예시:**
```
<Text: Moby Dick by Herman Melville 1851>
<Text: Sense and Sensibility by Jane Austen 1811>
```

---

## 🔍 Step 2: Concordance - 단어가 사용된 문맥 찾기

**설명:** 특정 단어가 어떤 문맥에서 사용되었는지 확인

```python
# 'whale'이라는 단어가 text1에서 어떻게 사용되었는지
text1.concordance("whale")

# 출력 예시:
# Displaying 25 of 906 matches:
# ong the former , one was of a most monstrous size . ... This came towards us , 
# ON OF THE PSALMS . "Touching that monstrous bulk of the whale or ork we have r
# ll over with a heathenish array of monstrous clubs and spears . Some were thick
# d as you gazed , and wondered what monstrous cannibal and savage could ever hav
```

**실습 과제:**
```python
# 1. text1에서 'sea'라는 단어의 문맥 찾기
text1.concordance("sea")

# 2. text2에서 'love'라는 단어의 문맥 찾기 (제인 오스틴 소설)
text2.concordance("love")

# 3. text4에서 'freedom'이라는 단어의 문맥 찾기 (대통령 연설문)
text4.concordance("freedom")

# 4. 결과를 제한하고 싶다면 (처음 5개만)
text1.concordance("whale", lines=5)
```

---

## 🎯 Step 3: Similar - 비슷한 맥락의 단어 찾기

**설명:** 특정 단어와 비슷한 문맥에서 사용된 다른 단어들 찾기

```python
# 'monstrous'와 비슷한 맥락으로 사용된 단어들
text1.similar("monstrous")

# 출력 예시:
# true contemptible christian abundant few part mean careful puzzled
# mystifying passing curious loving wise doleful gamesome singular
# delightfully perilous fearful threatening
```

**실습 과제:**
```python
# 1. text1에서 'ship'과 비슷한 맥락의 단어
text1.similar("ship")

# 2. text2에서 'happy'와 비슷한 맥락의 단어
text2.similar("happy")

# 3. text3에서 'God'와 비슷한 맥락의 단어 (성경)
text3.similar("God")
```

**왜 이게 중요한가요?**
- 단어의 의미를 문맥을 통해 이해할 수 있습니다
- 동의어나 유사한 개념을 발견할 수 있습니다
- 작가의 어휘 사용 패턴을 파악할 수 있습니다

---

## 🔗 Step 4: Common Contexts - 공통 문맥 찾기

**설명:** 두 단어가 공통적으로 사용되는 문맥 찾기

```python
# 'ship'과 'boat'가 공통으로 나타나는 문맥
text1.common_contexts(["ship", "boat"])

# 출력 예시:
# the_is a_. the_was
```

**해석:** "the ship is", "the boat is", "a ship.", "a boat." 등의 패턴에서 공통적으로 사용됨

**실습 과제:**
```python
# 1. text1에서 'sea'와 'ocean'의 공통 문맥
text1.common_contexts(["sea", "ocean"])

# 2. text2에서 'man'과 'woman'의 공통 문맥
text2.common_contexts(["man", "woman"])

# 3. text4에서 'people'과 'citizens'의 공통 문맥
text4.common_contexts(["people", "citizens"])
```

---

## 📊 Step 5: 텍스트 기본 통계

```python
# 총 단어 수 (토큰 개수)
print(len(text1))
# 출력: 260819

# 고유 단어 수 (중복 제거)
print(len(set(text1)))
# 출력: 19317

# 어휘 다양성 (Lexical Diversity)
# = 고유 단어 수 / 전체 단어 수
lexical_diversity = len(set(text1)) / len(text1)
print(f"어휘 다양성: {lexical_diversity:.4f}")
# 출력: 어휘 다양성: 0.0741

# 특정 단어의 빈도수
print(text1.count("whale"))
# 출력: 906

# 특정 단어가 전체에서 차지하는 비율 (%)
word_percentage = 100 * text1.count("whale") / len(text1)
print(f"'whale'의 비율: {word_percentage:.4f}%")
# 출력: 'whale'의 비율: 0.3473%
```

**실습 과제: 텍스트 비교 분석**
```python
# 여러 텍스트의 어휘 다양성 비교
texts = [text1, text2, text3, text4, text5]
names = ["Moby Dick", "Sense & Sensibility", "Genesis", "Inaugural", "Chat"]

for text, name in zip(texts, names):
    diversity = len(set(text)) / len(text)
    print(f"{name}: {diversity:.4f}")

# 예상 결과:
# Moby Dick: 0.0741
# Sense & Sensibility: 0.0485
# Genesis: 0.0620
# Inaugural: 0.0617
# Chat: 0.1332  <- 채팅은 어휘가 다양함!
```

---

## 📈 Step 6: Dispersion Plot (시각화)

**설명:** 텍스트 전체에서 특정 단어들이 어디에 등장하는지 시각화

```python
# matplotlib가 설치되어 있어야 합니다
# pip install matplotlib

from nltk.book import text4  # 대통령 연설문

# 주요 정치 용어들의 분포 확인
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
```

**그래프 없이 위치 확인하기:**
```python
# 특정 단어가 나타나는 위치 찾기
def find_word_positions(text, word, max_positions=10):
    positions = [i for i, w in enumerate(text) if w.lower() == word.lower()]
    print(f"'{word}' 등장 횟수: {len(positions)}회")
    print(f"처음 {max_positions}개 위치: {positions[:max_positions]}")
    return positions

# 실습
find_word_positions(text1, "whale", 10)
find_word_positions(text4, "freedom", 10)
```

---

## 🎮 Day 1 미션: 텍스트 탐험가

**미션 1-1: 단어 탐정**
```python
# TODO: text1에서 'captain'이라는 단어를 조사하세요
# 1. concordance로 문맥 확인
# 2. similar로 비슷한 단어 찾기
# 3. count로 빈도수 확인

# 여기에 코드 작성:



```

**미션 1-2: 텍스트 비교 분석가**
```python
# TODO: text1(소설)과 text4(연설문)을 비교하세요
# 1. 각각의 총 단어 수
# 2. 각각의 고유 단어 수  
# 3. 각각의 어휘 다양성
# 4. 어느 텍스트가 더 다양한 어휘를 사용하는가?

# 여기에 코드 작성:



```

**미션 1-3: 패턴 발견자**
```python
# TODO: text2(제인 오스틴 소설)에서
# 1. 'love'와 'hate'의 공통 문맥 찾기
# 2. 'Mr'와 'Mrs'의 공통 문맥 찾기
# 3. 두 패턴을 비교하고 분석하기

# 여기에 코드 작성:



```

---

## ✅ Day 1 완료 체크리스트

- [ ] `from nltk.book import *` 성공적으로 실행
- [ ] text1부터 text9까지 어떤 텍스트인지 확인
- [ ] concordance() 함수로 최소 3개 단어의 문맥 확인
- [ ] similar() 함수로 최소 3개 단어의 유사어 찾기
- [ ] common_contexts() 함수로 최소 2쌍의 공통 문맥 확인
- [ ] len(), set(), count() 함수로 통계 계산
- [ ] 어휘 다양성(lexical diversity) 개념 이해
- [ ] 미션 1-1, 1-2, 1-3 완료

---

## 🏆 Day 1 성과

**획득 경험치:** +10 XP (일일 과제) + 50 XP (미션 완료) = **60 XP**

**배운 개념:**
- NLTK의 Text 객체
- Concordance (문맥 분석)
- Similarity (유사어 찾기)
- Common Contexts (공통 문맥)
- 어휘 다양성 (Lexical Diversity)
- 빈도 분석

**다음 단계:** Day 2 - 토큰화(Tokenization) 기초

---

## 💡 추가 도전 과제 (+25 XP)

```python
# 도전 과제: 9개 텍스트 중 가장 어휘가 다양한 텍스트 찾기

texts_dict = {
    "Moby Dick": text1,
    "Sense & Sensibility": text2,
    "Genesis": text3,
    "Inaugural": text4,
    "Chat": text5,
    "Monty Python": text6,
    "WSJ": text7,
    "Personals": text8,
    "Chesterton": text9
}

# TODO: 각 텍스트의 어휘 다양성을 계산하고
# 가장 높은 것과 가장 낮은 것을 찾으세요
# 왜 그런 차이가 나는지 분석해보세요

# 여기에 코드 작성:



```

---


