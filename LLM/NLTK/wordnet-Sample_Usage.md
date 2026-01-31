# Wordnet

* NLTK에서 **WordNet**은 자연어 처리를 위한 가장 강력하고 방대한 **영어 어휘 시맨틱(Semantic) 데이터 베이스**
* 프린스턴 대학교에서 구축한 이 DB는 단순한 사전적 정의를 넘어, 단어 사이의 의미적 관계를 네트워크 형태로 구축해 놓은 것이 특징



### 1. WordNet의 핵심 개념: Synset (유의어 집합)

WordNet의 가장 기본 단위는 Synset (Synonym Set)

* 단순히 단어(Word)별로 나열하는 것이 아니라, **동일한 개념을 공유하는 단어들을 하나의 집합** 으로 묶습니다.
* 예를 들어, 'car'라는 개념에는 'automobile', 'motocar'등이 포함되며, 이들은 하나의 synset으로 관리됩니다.
* 다의어 처리: 하나의 단어가 여러 의미를 가질 경우 (예: 'bank'- 강둑, 은행), 각각 서로 다른 Synset에 소속됩니다.



### 2. WordNet의 주요 기능 및 관계

WordNet은 단어들 간의 다양한 관계를 정의하여, 컴퓨터가 단어의 의미를 이해하도록 돕습니다.

####  ① 상위어(Hypernyms)와 하위어(Hyponyms) - 계층 구조

* **상위어(Hypernym)**: 특정 단어보다 더 포괄적인 의미를 가진 단어 (예: '포유류'는 '개'의 상위어)
* **하위어(Hyponym)**: 전체를 나타내는 단어 (예: '자동차'는 '바퀴'의 전체어)
* 이를 통해 **HAS-A 관계** (A는 B를 가지고 있다)를 파악할 수 있습니다.

#### ② 전체어(Holonyms)와 부분어(Meronyms) - 구성관계

*    **부분어(Meronym):** 구성 요소를 나타내는 단어 (예: '바퀴'는 '자동차'의 부분어)
* **전체어(Holonym):** 전체를 나타내는 단어 (예: '자동차'는 '바퀴'의 전체어) 
* 이를 통해 **HAS-A 관계**(A는 B를 가지고 있다)를 파악할 수 있습니다. 
  

#### ③ 반의어(Antonyms)
* 서로 반대되는 의미를 가진 단어를 연결합니다. (예: 'Good' ↔ 'Bad') 

#### ④ 정의(Definition) 및 예문(Examples)
*   각 Synset에 대한 사전적 정의와 실제 사용 예문을 제공합니다. 



### 3. WordNet의 활용: 단어 유사도 측정

WordNet의 가장 강력한 기능 중 하나는 두단어가 얼마나 유사한지 **수치로 계산** 할 수 있다는 점
* 경로 유사도(Path Similarity)



### 4. NLTK 실전 사용 예시 (Python) 

NLTK를 통해 WordNet을 사용하는 기본 방법입니다.

```python 
import nltk from nltk.corpus 
import wordnet 

# 1. 특정 단어의 Synset들 조회 
syns = wordnet.synsets("program") 
print(syns[0].name()) 
# 출력: plan.n.01 (계획이라는 의미의 첫 번째 명사) 

# 2. 정의와 예문 확인 
print(syns[0].definition()) 
# 정의 출력 
print(syns[0].examples())   

# 예문 출력 
# 3. 상위어 확인 
print(syns[0].hypernyms()) 
# 4. 단어 유사도 측정 (Dog와 Cat) 
dog = wordnet.synset('dog.n.01') 
cat = wordnet.synset('cat.n.01') 
similarity = dog.path_similarity(cat) 
print(f"Dog와 Cat의 유사도: {similarity}") # 0.2 정도로 나옴 
```



### 5. WordNet의 주요 용도 요약 

1.  **단어 의미 중의성 해소(WSD, Word Sense Disambiguation):** 문맥 속에서 단어가 어떤 의미로 쓰였는지 분류. 
2.  **질의 확장(Query Expansion):** 검색 시 사용자가 입력한 단어의 유의어를 함께 검색하여 결과 범위 확대. 
3.  **텍스트 요약 및 분류:** 단어의 상위어 개념을 추출하여 텍스트의 주제를 파악. 
4.  **감성 분석:** 형용사의 반의어나 유의어를 추적하여 텍스트의 긍정/부정 판단에 활용. 



#### 요약하자면 WordNet은 단순한 사전이 아니라 **"단어들이 의미적으로 어떻게 연결되어 있는지를 보여주는 지도"**와 같습니다. 자연어 처리(NLP) 초기에 의미론적 분석을 위해 필수적으로 사용되었으며, 현재도 지식 그래프 구축이나 데이터 보강(Augmentation) 등에 널리 활용되고 있습니다. 
