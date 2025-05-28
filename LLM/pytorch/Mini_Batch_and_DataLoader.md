# Mini Batch and DataLoader



이번 챕터에서는 데이터를 로드하는 방법과 미니 배치 경사 하강법(Minibatch Gradient Descent)에 대해서 학습합니다.



## 1. 미니 배치와 배치 크기(Mini Batch and Batch Size)

앞서 배운 다중 선형 회귀에서 사용했던 데이터를 상기해봅시다.

```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```

위 데이터의 샘플의 개수는 5개입니다. 전체 데이터를 하나의 행렬로 선언하여 전체 데이터에 대해서 경사 하강법을 수행하여 학습할 수 있습니다. 그런데 위 데이터는 현업에서 다루게 되는 방대한 양의 데이터에 비하면 굉장히 적은 양입니다. 만약, 데이터가 수십만개 이상이라면 전체 데이터에 대해서 경사 하강법을 수행하는 것은 매우 느릴 뿐만 아니라 많은 계산량이 필요합니다. 정말 어쩌면 메모리의 한계로 계산이 불가능한 경우도 있을 수 있습니다.

그렇기 때문에 전체 데이터를 더 작은 단위로 나누어서 해당 단위로 학습하는 개념이 나오게 되었습니다.
이 단위를 미니 배치(Mini Batch)라고 합니다.

![img](https://wikidocs.net/images/page/55580/%EB%AF%B8%EB%8B%88%EB%B0%B0%EC%B9%98.PNG)

위의 그림은 전체 데이터를 미니 배치 단위로 나누는 것을 보여줍니다. 미니 배치 학습을 하게되면 미니 배치만큼만 가져가서 미니 배치에 대한 대한 비용(cost)를 계산하고, 경사 하강법을 수행합니다. 그리고 다음 미니 배치를 가져가서 경사 하강법을 수행하고 마지막 미니 배치까지 이를 반복합니다. 이렇게 전체 데이터에 대한 학습이 1회 끝나면 1 에포크(Epoch)가 끝나게 됩니다.

- **에포크(Epoch)는 전체 훈련 데이터가 학습에 한 번 사용된 주기를 말한다고 언급한 바 있습니다.**

미니 배치 학습에서는 미니 배치의 개수만큼 경사 하강법을 수행해야 전체 데이터가 한 번 전부 사용되어 1 에포크(Epoch)가 됩니다. 미니 배치의 개수는 결국 미니 배치의 크기를 몇으로 하느냐에 따라서 달라지는데 미니 배치의 크기를 배치 크기(batch size)라고 합니다.

- **전체 데이터에 대해서 한 번에 경사 하강법을 수행하는 방법을 '배치 경사 하강법'이라고 부릅니다. 반면, 미니 배치 단위로 경사 하강법을 수행하는 방법을 '미니 배치 경사 하강법'이라고 부릅니다.**

- **배치 경사 하강법은 경사 하강법을 할 때, 전체 데이터를 사용하므로 가중치 값이 최적값에 수렴하는 과정이 매우 안정적이지만, 계산량이 너무 많이 듭니다. 미니 배치 경사 하강법은 경사 하강법을 할 때, 전체 데이터의 일부만을 보고 수행하므로 최적값으로 수렴하는 과정에서 값이 조금 헤매기도 하지만 훈련 속도가 빠릅니다.**

- 배치 크기는 보통 2의 제곱수를 사용합니다. ex) 2, 4, 8, 16, 32, 64... 그 이유는 CPU와 GPU의 메모리가 2의 배수이므로 배치크기가 2의 제곱수일 경우에 데이터 송수신의 효율을 높일 수 있다고 합니다.

  

## 2. 이터레이션(Iteration)

미니 배치와 배치 크기의 정의에 대해서 이해하였다면 이터레이션(iteration)을 정의할 수 있습니다.

![img](https://wikidocs.net/images/page/36033/batchandepochiteration.PNG)

위의 그림은 에포크와 배치 크기와 이터레이션의 관계를 보여줍니다. 위의 그림의 예제를 통해 설명해보겠습니다.

이터레이션은 한 번의 에포크 내에서 이루어지는 매개변수인 가중치 W와 b의 업데이트 횟수입니다. 전체 데이터가 2,000일 때 배치 크기를 200으로 한다면 이터레이션의 수는 총 10개입니다. 이는 한 번의 에포크 당 매개변수 업데이트가 10번 이루어짐을 의미합니다.

이제 미니 배치 학습을 할 수 있도록 도와주는 파이토치의 도구들을 알아봅시다.



## 3. 데이터 로드하기(Data Load)

파이토치에서는 데이터를 좀 더 쉽게 다룰 수 있도록 유용한 도구로서 데이터셋(Dataset)과 데이터로더(DataLoader)를 제공합니다. 이를 사용하면 **미니 배치 학습**, 데이터 셔플(shuffle), 병렬 처리까지 간단히 수행할 수 있습니다. 기본적인 사용 방법은 Dataset을 정의하고, 이를 DataLoader에 전달하는 것입니다.

Dataset을 커스텀하여 만들 수도 있지만 여기서는 텐서를 입력받아 Dataset의 형태로 변환해주는 TensorDataset을 사용해보겠습니다.

실습을 위해 기본적으로 필요한 파이토치의 도구들을 임포트합니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

TensorDataset과 DataLoader를 임포트합니다.

```python
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
```

TensorDataset은 기본적으로 텐서를 입력으로 받습니다. 텐서 형태로 데이터를 정의합니다.

```python
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
```

이제 이를 TensorDataset의 입력으로 사용하고 dataset으로 저장합니다.

```python
dataset = TensorDataset(x_train, y_train)
```

파이토치의 데이터셋을 만들었다면 데이터로더를 사용 가능합니다. 데이터로더는 기본적으로 2개의 인자를 입력받는다. 하나는 데이터셋, 미니 배치의 크기입니다. 이때 미니 배치의 크기는 통상적으로 2의 배수를 사용합니다. (ex) 64, 128, 256...) 그리고 추가적으로 많이 사용되는 인자로 shuffle이 있습니다. shuffle=True를 선택하면 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꿉니다.

사람도 같은 문제지를 계속 풀면 어느 순간 문제의 순서에 익숙해질 수 있습니다. 예를 들어 어떤 문제지의 12번 문제를 풀면서, '13번 문제가 뭔지는 기억은 안 나지만 어제 풀었던 기억으로 정답은 5번이었던 것 같은데' 하면서 문제 자체보단 순서에 익숙해질 수 있다는 것입니다. 그럴 때 문제지를 풀 때마다 문제 순서를 랜덤으로 바꾸면 도움이 될 겁니다. 마찬가지로 모델이 데이터셋의 순서에 익숙해지는 것을 방지하여 학습할 때는 이 옵션을 True를 주는 것을 권장합니다.

```python
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

이제 모델과 옵티마이저를 설계합니다.

```python
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 
```

이제 훈련을 진행합니다. 아래 코드에서는 batch_idx와 samples를 주석 처리했는데 어떤 식으로 훈련되고 있는지 궁금하다면 주석 처리를 해제하고 훈련시켜보시기 바랍니다.

```python
nb_epochs = 20
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    # print(batch_idx)
    # print(samples)
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
Epoch    0/20 Batch 1/3 Cost: 26085.919922
Epoch    0/20 Batch 2/3 Cost: 3660.022949
Epoch    0/20 Batch 3/3 Cost: 2922.390869
... 중략 ...
Epoch   20/20 Batch 1/3 Cost: 6.315856
Epoch   20/20 Batch 2/3 Cost: 13.519956
Epoch   20/20 Batch 3/3 Cost: 4.262849
```

Cost의 값이 점차 작아집니다. (사실 아직 에포크를 더 늘려서 훈련하면 Cost의 값이 더 작아질 여지가 있습니다. 에포크를 늘려서도 훈련해보세요.) 이제 모델의 입력으로 임의의 값을 넣어 예측값을 확인합니다.

```python
# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 
훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[154.3850]], grad_fn=<AddmmBackward>)
```

앞 내용을 잠깐 복습해봅시다. 파이토치에서는 데이터셋을 좀 더 쉽게 다룰 수 있도록 유용한 도구로서 torch.utils.data.Dataset과 torch.utils.data.DataLoader를 제공합니다. 이를 사용하면 미니 배치 학습, 데이터 셔플(shuffle), 병렬 처리까지 간단히 수행할 수 있습니다. 기본적인 사용 방법은 Dataset을 정의하고, 이를 DataLoader에 전달하는 것입니다.



## 4. 커스텀 데이터셋(Custom Dataset)

그런데 torch.utils.data.Dataset을 상속받아 직접 커스텀 데이터셋(Custom Dataset)을 만드는 경우도 있습니다. torch.utils.data.Dataset은 파이토치에서 데이터셋을 제공하는 추상 클래스입니다. Dataset을 상속받아 다음 메소드들을 오버라이드 하여 커스텀 데이터셋을 만들어보겠습니다.

커스텀 데이터셋을 만들 때, 일단 가장 기본적인 뼈대는 아래와 같습니다. 여기서 필요한 기본적인 define은 3개입니다.

```python
class CustomDataset(torch.utils.data.Dataset): 
  def __init__(self):

  def __len__(self):

  def __getitem__(self, idx): 
```

이를 좀 더 자세히 봅시다.

```python
class CustomDataset(torch.utils.data.Dataset): 
  def __init__(self):
  데이터셋의 전처리를 해주는 부분

  def __len__(self):
  데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분

  def __getitem__(self, idx): 
  데이터셋에서 특정 1개의 샘플을 가져오는 함수
```

- len(dataset)을 했을 때 데이터셋의 크기를 리턴할 **len**
- dataset[i]을 했을 때 i번째 샘플을 가져오도록 하는 인덱싱을 위한 **get_item**



## 5. 커스텀 데이터셋(Custom Dataset)으로 선형 회귀 구현하기

```python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# Dataset 상속
class CustomDataset(Dataset): 
  def __init__(self):
    self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
    self.y_data = [[152], [185], [180], [196], [142]]

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y
```

위 코드는 PyTorch의 Dataset 클래스를 상속받아 사용자 정의 데이터셋 클래스를 만드는 방법을 보여줍니다. CustomDataset이라는 클래스는 두 가지 데이터를 포함합니다. x_data는 입력 데이터이고, y_data는 해당 입력에 대응하는 출력 데이터입니다.

클래스가 초기화될 때, 이 두 데이터를 내부 변수로 저장합니다. 데이터셋의 길이를 반환하는 메서드가 정의되어 있으며, 이 메서드는 데이터셋에 포함된 데이터의 개수를 반환합니다. 이 클래스에서 가장 중요한 메서드는 인덱스를 입력으로 받아 해당 인덱스에 맵핑된 데이터를 반환하는 것입니다. 이 메서드는 x_data와 y_data의 특정 인덱스에 해당하는 데이터를 torch.FloatTensor 형식으로 변환하여 반환합니다.

```python
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = torch.nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 
```

위 코드는 CustomDataset 클래스를 인스턴스화하여 데이터셋 객체를 만듭니다. 이 데이터셋을 기반으로 PyTorch의 DataLoader 객체를 생성하며, 여기서 배치 크기를 설정하고 데이터를 무작위로 섞을지 여부를 결정합니다. 배치 크기는 한 번에 몇 개의 데이터를 모델에 입력할지 결정하며, 무작위 섞기는 모델이 학습하는 동안 데이터 순서에 의한 편향을 방지하기 위해 사용됩니다.

그 다음, 입력 차원이 3이고 출력 차원이 1인 선형 회귀 모델을 정의합니다. 이 모델은 torch.nn.Linear 클래스를 사용하여 구현됩니다. 모델이 데이터에서 학습할 수 있도록 옵티마이저를 설정하는데, 여기서는 확률적 경사 하강법(SGD) 옵티마이저를 사용합니다. 옵티마이저는 모델의 파라미터를 업데이트할 때 학습률을 사용하며, 학습률은 모델이 얼마나 빠르게 또는 느리게 학습할지를 결정하는 중요한 하이퍼파라미터입니다.

```python
nb_epochs = 20
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    # print(batch_idx)
    # print(samples)
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
```

전체 학습 반복 횟수는 nb_epochs로 설정되어 있습니다. 외부 반복문에서는 각 학습 주기, 즉 에포크를 제어하며, 내부 반복문에서는 데이터 로더에서 미니배치 단위로 데이터를 가져와 모델을 학습시킵니다.

데이터 로더에서 가져온 각 배치는 samples라는 변수에 저장되고, 이 배치에서 입력 데이터 x_train과 타겟 데이터 y_train을 분리합니다. 모델은 입력 데이터에 대해 예측 값을 계산하고, 그 예측 값과 실제 타겟 값 사이의 손실을 평균 제곱 오차 함수 F.mse_loss를 사용해 계산합니다.

계산된 손실 값을 기반으로 모델의 가중치를 업데이트하기 위해 기울기를 초기화한 후, 손실에 대한 역전파를 수행하고, 옵티마이저를 사용해 모델의 파라미터를 업데이트합니다.

각 에포크와 배치가 끝날 때마다 현재 에포크와 배치 번호, 그리고 해당 배치에서의 손실 값이 출력됩니다. 이를 통해 학습이 잘 진행되고 있는지를 실시간으로 모니터링할 수 있습니다.

```python
Epoch    0/20 Batch 1/3 Cost: 29410.156250
Epoch    0/20 Batch 2/3 Cost: 7150.685059
Epoch    0/20 Batch 3/3 Cost: 3482.803467
... 중략 ...
Epoch   20/20 Batch 1/3 Cost: 0.350531
Epoch   20/20 Batch 2/3 Cost: 0.653316
Epoch   20/20 Batch 3/3 Cost: 0.010318
# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 
Copy훈련 후 입력이 73, 80, 75일 때의 예측값 : tensor([[151.2319]], grad_fn=<AddmmBackward>)
```