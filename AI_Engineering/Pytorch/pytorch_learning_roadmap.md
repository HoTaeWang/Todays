# 🎮 PyTorch 딥러닝 마스터 게임 로드맵

## 🎯 총 미션 개수: 12단계 (약 8-12주 소요)

당신의 선생님으로서, "Deep Learning with PyTorch"를 완벽하게 마스터하는 여정을 게임처럼 설계했습니다!

---

## 🌟 레벨 1: 튜토리얼 존 (1주차)
**주제: PyTorch 기초와 환경 설정**

### 📚 학습 내용
- PyTorch란 무엇인가?
- 설치 및 환경 설정
- Tensor 기초 개념
- PyTorch의 autograd (자동 미분)

### 🎯 미션 1: "첫 번째 텐서 마법사"
```python
# 미션: 다음 코드를 실행하고 결과를 이해하기
import torch

# 1. 3x3 랜덤 텐서 생성
tensor_a = torch.randn(3, 3)

# 2. 텐서 연산 수행
tensor_b = tensor_a * 2
tensor_c = tensor_a + tensor_b

# 3. 결과 출력 및 shape 확인
print(f"Tensor A: {tensor_a}")
print(f"Tensor C: {tensor_c}")
print(f"Shape: {tensor_c.shape}")
```

### ✅ 체크포인트
- [ ] PyTorch 설치 완료
- [ ] 텐서 생성 및 기본 연산 이해
- [ ] GPU/CPU 차이 이해
- [ ] autograd 개념 이해

### 🏆 보스 미션: "자동 미분 마스터"
`y = x² + 2x + 1`에서 x=3일 때 미분값을 autograd로 계산하기

---

## 🌟 레벨 2: 신경망 입문 (1주차)
**주제: 신경망의 기초**

### 📚 학습 내용
- 퍼셉트론과 다층 퍼셉트론
- 활성화 함수 (ReLU, Sigmoid, Tanh)
- 순전파와 역전파
- nn.Module 사용법

### 🎯 미션 2: "나의 첫 신경망"
```python
import torch.nn as nn

# 미션: 간단한 2층 신경망 만들기
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 생성 및 테스트
model = SimpleNet()
dummy_input = torch.randn(1, 10)
output = model(dummy_input)
```

### ✅ 체크포인트
- [ ] nn.Module 상속 이해
- [ ] forward() 메서드 역할 이해
- [ ] 활성화 함수 종류와 특징 이해
- [ ] 레이어 연결 방법 이해

### 🏆 보스 미션: "XOR 문제 해결하기"
XOR 문제를 풀 수 있는 신경망 설계 및 학습

---

## 🌟 레벨 3: 학습 메커니즘 (1-2주차)
**주제: 손실 함수와 최적화**

### 📚 학습 내용
- 손실 함수 (MSE, Cross-Entropy)
- 경사하강법 (SGD, Adam, RMSprop)
- 학습률(Learning Rate)의 중요성
- 배치(Batch) 개념

### 🎯 미션 3: "최적화 마법사"
```python
import torch.optim as optim

# 미션: 다양한 옵티마이저로 같은 문제 풀어보기
model = SimpleNet()
criterion = nn.MSELoss()

# 3가지 옵티마이저 비교
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01),
    'Adam': optim.Adam(model.parameters(), lr=0.001),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.001)
}

# 각 옵티마이저로 학습 후 수렴 속도 비교
```

### ✅ 체크포인트
- [ ] 손실 함수 3가지 이상 이해
- [ ] 옵티마이저 차이점 이해
- [ ] 학습률 조정의 중요성 이해
- [ ] 배치 사이즈의 영향 이해

### 🏆 보스 미션: "학습 곡선 그리기"
손실값과 정확도를 epoch마다 기록하고 matplotlib으로 시각화

---

## 🌟 레벨 4: 데이터 로딩 마스터 (1주차)
**주제: Dataset과 DataLoader**

### 📚 학습 내용
- torch.utils.data.Dataset
- torch.utils.data.DataLoader
- 데이터 전처리 및 증강
- 배치 처리

### 🎯 미션 4: "데이터 파이프라인 구축"
```python
from torch.utils.data import Dataset, DataLoader

# 미션: 커스텀 Dataset 클래스 만들기
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# DataLoader로 배치 생성
dataset = CustomDataset(your_data, your_labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### ✅ 체크포인트
- [ ] Dataset 클래스 구조 이해
- [ ] __getitem__과 __len__ 구현
- [ ] DataLoader 파라미터 이해
- [ ] 데이터 셔플링의 중요성 이해

### 🏆 보스 미션: "CSV 데이터셋 로더 만들기"
CSV 파일을 읽어서 Dataset으로 변환하는 클래스 작성

---

## 🌟 레벨 5: 이미지 분류 입문 (2주차)
**주제: CNN 기초**

### 📚 학습 내용
- 합성곱(Convolution) 연산
- 풀링(Pooling) 레이어
- CNN 아키텍처 기초
- MNIST/CIFAR-10 다루기

### 🎯 미션 5: "나의 첫 CNN"
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### ✅ 체크포인트
- [ ] Conv2d 파라미터 이해 (in_channels, out_channels, kernel_size)
- [ ] 풀링의 역할 이해
- [ ] Feature Map 크기 계산
- [ ] Flatten 연산 이해

### 🏆 보스 미션: "MNIST 손글씨 인식 90% 이상 달성"
MNIST 데이터셋으로 90% 이상의 정확도 달성

---

## 🌟 레벨 6: 전이 학습 (1-2주차)
**주제: 사전 학습 모델 활용**

### 📚 학습 내용
- ImageNet과 사전 학습 모델
- ResNet, VGG, EfficientNet
- Feature Extraction vs Fine-tuning
- torchvision.models 사용법

### 🎯 미션 6: "거인의 어깨 위에 서기"
```python
import torchvision.models as models

# 미션: ResNet18 사전 학습 모델 사용하기
model = models.resnet18(pretrained=True)

# Feature Extraction: 마지막 레이어만 교체
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10개 클래스로 변경

# 대부분의 레이어 동결
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True
```

### ✅ 체크포인트
- [ ] 사전 학습의 장점 이해
- [ ] Feature Extraction 방법 이해
- [ ] Fine-tuning 전략 이해
- [ ] 레이어 동결/해제 방법 이해

### 🏆 보스 미션: "커스텀 이미지 분류기"
자신만의 이미지 데이터셋으로 전이 학습 적용

---

## 🌟 레벨 7: 순차 데이터 처리 (2주차)
**주제: RNN과 LSTM**

### 📚 학습 내용
- RNN 구조와 원리
- LSTM과 GRU
- 시퀀스 투 시퀀스
- 텍스트 데이터 전처리

### 🎯 미션 7: "시간의 마법사"
```python
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
```

### ✅ 체크포인트
- [ ] RNN의 hidden state 개념 이해
- [ ] LSTM의 게이트 메커니즘 이해
- [ ] 시퀀스 길이 처리 방법 이해
- [ ] batch_first 파라미터 이해

### 🏆 보스 미션: "주식 가격 예측 모델"
시계열 데이터로 LSTM 모델 학습

---

## 🌟 레벨 8: 자연어 처리 입문 (2주차)
**주제: NLP와 임베딩**

### 📚 학습 내용
- Word Embedding (Word2Vec, GloVe)
- nn.Embedding 사용법
- 감성 분석
- 텍스트 분류

### 🎯 미션 8: "단어의 의미 찾기"
```python
# 미션: 영화 리뷰 감성 분석기 만들기
class SentimentAnalyzer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # 긍정/부정
        
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out
```

### ✅ 체크포인트
- [ ] 임베딩의 의미 이해
- [ ] Vocabulary 구축 방법 이해
- [ ] 패딩과 시퀀스 길이 조정 이해
- [ ] 감성 분석 파이프라인 이해

### 🏆 보스 미션: "나만의 감성 분석기"
실제 텍스트 데이터로 감성 분석 모델 구축

---

## 🌟 레벨 9: Transformer 시대 (2주차)
**주제: Attention과 Transformer**

### 📚 학습 내용
- Attention 메커니즘
- Self-Attention
- Transformer 아키텍처
- BERT, GPT 소개

### 🎯 미션 9: "주의력 마스터"
```python
# 미션: 간단한 Multi-Head Attention 구현
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v):
        # Attention 계산 구현
        pass
```

### ✅ 체크포인트
- [ ] Attention의 Query, Key, Value 이해
- [ ] Self-Attention vs Cross-Attention 이해
- [ ] Multi-Head의 의미 이해
- [ ] Positional Encoding 이해

### 🏆 보스 미션: "Transformer 블록 구현"
완전한 Transformer 인코더 블록 구현

---

## 🌟 레벨 10: 생성 모델 (2주차)
**주제: GAN과 VAE**

### 📚 학습 내용
- GAN 구조 (Generator + Discriminator)
- VAE (Variational Autoencoder)
- 이미지 생성
- 잠재 공간(Latent Space)

### 🎯 미션 10: "창조의 마법사"
```python
# 미션: 간단한 GAN 구조 만들기
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, img_shape),
            nn.Tanh()
        )
        
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_shape, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        return self.model(img)
```

### ✅ 체크포인트
- [ ] GAN의 적대적 학습 이해
- [ ] Generator와 Discriminator 역할 이해
- [ ] 모드 붕괴(Mode Collapse) 문제 이해
- [ ] VAE의 재구성 손실 이해

### 🏆 보스 미션: "MNIST 숫자 생성기"
GAN으로 MNIST 스타일의 숫자 이미지 생성

---

## 🌟 레벨 11: 객체 탐지 (2주차)
**주제: Object Detection**

### 📚 학습 내용
- R-CNN 계열
- YOLO 아키텍처
- Bounding Box와 IoU
- Non-Maximum Suppression

### 🎯 미션 11: "탐지의 달인"
```python
import torchvision

# 미션: Faster R-CNN으로 객체 탐지
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 이미지에서 객체 탐지
with torch.no_grad():
    predictions = model(images)
    
# Bounding Box 그리기
for box, label, score in zip(predictions['boxes'], 
                              predictions['labels'], 
                              predictions['scores']):
    if score > 0.5:
        # 박스 그리기
        pass
```

### ✅ 체크포인트
- [ ] Bounding Box 표현 방식 이해
- [ ] IoU 계산 방법 이해
- [ ] NMS 알고리즘 이해
- [ ] Anchor Box 개념 이해

### 🏆 보스 미션: "실시간 객체 탐지 시스템"
웹캠으로 실시간 객체 탐지 구현

---

## 🌟 레벨 12: 최종 보스전 (2-3주차)
**주제: 실전 프로젝트와 배포**

### 📚 학습 내용
- 모델 저장 및 로드
- TorchScript와 ONNX
- 모델 최적화 (양자화, 프루닝)
- 배포 전략

### 🎯 미션 12: "마스터 프로젝트"
```python
# 미션: 완전한 딥러닝 파이프라인 구축
# 1. 데이터 수집
# 2. 전처리
# 3. 모델 설계
# 4. 학습
# 5. 평가
# 6. 저장
torch.save(model.state_dict(), 'final_model.pth')

# 7. 배포용 변환
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pt')
```

### ✅ 체크포인트
- [ ] 모델 체크포인트 저장/로드
- [ ] TorchScript 변환
- [ ] ONNX 변환
- [ ] 모델 양자화 적용

### 🏆 최종 보스 미션: "나만의 AI 서비스"
선택한 도메인에서 완전한 딥러닝 서비스 구축 및 배포

---

## 📊 진행도 체크리스트

### 초급 (레벨 1-4) ⭐
- [ ] 레벨 1 완료
- [ ] 레벨 2 완료
- [ ] 레벨 3 완료
- [ ] 레벨 4 완료

### 중급 (레벨 5-8) ⭐⭐
- [ ] 레벨 5 완료
- [ ] 레벨 6 완료
- [ ] 레벨 7 완료
- [ ] 레벨 8 완료

### 고급 (레벨 9-12) ⭐⭐⭐
- [ ] 레벨 9 완료
- [ ] 레벨 10 완료
- [ ] 레벨 11 완료
- [ ] 레벨 12 완료

---

## 🎖️ 달성 배지 시스템

### 🥉 브론즈 배지 (레벨 1-4 완료)
"PyTorch 초심자"
- 기본 텐서 연산 마스터
- 간단한 신경망 구축 가능
- 데이터 로딩 파이프라인 구축 가능

### 🥈 실버 배지 (레벨 5-8 완료)
"딥러닝 실무자"
- CNN으로 이미지 분류 가능
- RNN으로 시퀀스 데이터 처리 가능
- 전이 학습 활용 가능

### 🥇 골드 배지 (레벨 9-12 완료)
"PyTorch 마스터"
- Transformer 구조 이해 및 구현
- 생성 모델 구축 가능
- 객체 탐지 시스템 개발 가능
- 실전 프로젝트 배포 경험

---

## 💡 학습 팁

1. **매일 코딩하기**: 하루 1시간씩이라도 꾸준히
2. **손으로 코드 쳐보기**: 복사-붙여넣기 금지!
3. **에러 로그 읽기**: 에러는 최고의 선생님
4. **시각화하기**: 텐서의 shape를 항상 print로 확인
5. **커뮤니티 활용**: PyTorch 포럼, Stack Overflow
6. **논문 읽기**: 각 레벨의 원본 논문 찾아 읽기

---

## 📅 권장 학습 일정

- **파트타임 (주 10시간)**: 12주 완성
- **풀타임 (주 30시간)**: 4주 완성
- **집중 캠프 (주 50시간)**: 2.5주 완성

---

## 🎯 다음 단계는?

현재 여러분은 **레벨 1: 튜토리얼 존**에 입장했습니다!

**첫 번째 미션을 시작할 준비가 되셨나요?** 😊

준비되셨다면 "미션 1 시작!"이라고 말씀해주세요!
