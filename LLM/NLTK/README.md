# Confliction Points of NLTK



* NLTK_DATA들을 Proxy를 거쳐 Download 받거나 NLTK_DATA의 환경변수 설정 필요



### ✅ 심볼릭 링크 생성 방법

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