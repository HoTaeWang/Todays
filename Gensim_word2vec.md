# Gensim Word2Vec



### Word2Vec

단어를 벡터로 변환하는데 사용되는 네트워크 기반의 단어 임베딩 방법

주어진 단어가 주변 단어들과 어떤 관계를 맺고 있는지를 파악하여, 그 단어를 고차원적인 벡터공간 상에 위치시키는 방식으로 작동

Word2Vec은 크게 두가지 모델, CBOW(Continuous Bag-Of-Words)화 SG(Skip-Gram)을 사용한다.

* CBOW(Continuous Bag-Of-Words): 주변 단어들을 이용해, 중앙에 위치한 타겟 단어를 예측하는 방식
* SK (Skip-Gram): 중앙에 위치한 단어를 이용해 주변 단어들을 예측하는 방식



#### 왜 Word2Vec을 사용하는가?

Word2Vec은 단어 간의 의미론적 유사성을 잘 반영한다.

예: "King" - "main" + "Woman" = "Queen"



#### Gensim 설치

```
pip install gensim
```



#### 단어 목록 준비

```
sentences = [['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']]
```



### 간단한 Word2Vec 모델 학습

Gensim의 Word2Vec 클래스를 사용하여 모델을 학습시킬 수 있다.

```
from gensim.models import Word2Vec

# 파라미터 설정
embedding_size = 100  # 단어 벡터의 차원 수
window_size = 5  # 컨텍스트 윈도우 크기
min_word_count = 1  # 최소 단어 빈도 수

# Word2Vec 모델 초기화 및 학습
model = Word2Vec(sentences, vector_size=embedding_size, window=window_size, min_count=min_word_count, workers=4)

# 모델 저장
model.save("word2vec.model")
```



### **모델 사용 예제**

```
# 'fox'와 가장 유사한 단어 5개 찾기
similar_words = model.wv.most_similar('fox', topn=5)
print(similar_words)
```

