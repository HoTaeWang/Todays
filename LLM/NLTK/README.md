# NLTK

* NLTK는 Python 기반의 대표적인 자연어 처리(NLP) 오픈소스 Toolkit
* 토큰화, 형태 처리, 품사 Tagging, Chunking, Parsing, 의미 해석등 전통적 NLP 전과정을 위한 모듈과 함께 50여종 이상의 Corpus/WordNet등 표준화된 인터페이스 제공
* 교육, 연구, 프로토타이핑에 특히 적합한 "종합 실습용 Toolbox"





## Key Modules

1) WordNet  ([Wordnet Sample Usage](./wordnet-Sample_Usage.md))







## Confliction Points of NLTK



* NLTK_DATA들을 Proxy를 거쳐 Download 받거나 NLTK_DATA의 환경변수 설정 필요



### ✅ 심볼릭 링크 생성 방법  (Configuration)

```
sudo ln -s /data/nltk_data /usr/share/nltk_data
```



필요하다면 `nltk` 코드에서도 명시적으로 경로를 설정할 수 있습니다:

```
import nltk
nltk.data.path.append('/usr/share/nltk_data')
```

그러면 `nltk`는 해당 경로에서 리소스를 먼저 찾습니다.

------

필요하다면 `~/.bashrc`나 `.profile`에 다음 환경변수를 설정할 수도 있습니다:

```
export NLTK_DATA=/usr/share/nltk_data
```

(그러면 `nltk`가 이 환경변수를 읽고 해당 경로를 참조합니다.)