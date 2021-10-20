import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 입력 이미지 채널 1개, 출력 채널 6개, 3x3의 정사각 컨볼루션 행렬
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 어파인 연산 : y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6*6은 이미지 차원에 해당
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # [2, 2] 크기 윈도우에 대해 맥스 풀링
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 크기가 정사각형이면 하나의 숫자만을 명시
        x = F.max_pool2d(F.relu(self.con2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # 배치 차원을 제외한 모든 차원
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
###### 신경망까지함

### 파라미터
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1의 가중치

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
# print(output)
net.zero_grad()
out.backward(torch.randn(1, 10))
### 손실 함수 - Loss Function
output = net(input) 
target = torch.randn(10) # target 덩어리
target = target.view(1, -1) # output과 동일한 형태로 만들기
criterion = nn.MESLoss()

loss = criterion(output, target)
print(loss)
#
print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

net.zero_grad() # 모든 매개변수의 변화도 버퍼를 0으로 만들기

print('conv1.bias.grad befort backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
