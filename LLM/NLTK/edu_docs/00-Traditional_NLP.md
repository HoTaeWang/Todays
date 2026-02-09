

# Traditional NLP



Corpus => pl. Corpora

Feature Engineering





## Lemmatization and Stemming

한국어는 영어와는 다른 Lemmatization과 Stemming의 적용방식 차이

한국어는 조사와 어미가 발달한 교착어(Agglutinative language)이므로 형태소 분석(Morpheme Analysis)라는 더 복잡한 과정이 필요

한국어 형태소 분석기: KoNLPy라이브러리 Mecab, Hannanum, Kkma, Okt(Open Korean Text)



* Categorizing Words: POS Tagging
* Categorizing Spans: Chunking and Named Entity Recognition





## NLTK



```import nltk
import nltk
nltk.download()
```



```
from nltk.book import *
text1.concordance('string')
text2.similar('string')
text1.common_contexts('string')
text2.common_contexts(['string1', 'string2'])

text4.dispersion_plot(['citizens', 'democracy', 'freedom', 'duties', 'America'])
```

