import math
import torch

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)
# @는 내적 연산을 나타냅니다.
# _ => inplace 처리입니다.
bs = 64 # 배치 사이즈

xb = x_train[0:bs] # x로부터 미니배치 설정
preds = model(xb) # 예측
preds[0], preds.shape
print(preds[0], preds.shape)
