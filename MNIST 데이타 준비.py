from pathlib import Path
import requests

DATA_PATH = Path('data')
PATH = DATA_PATH / 'mnist'

PATH.mkdir(parents=True, exist_ok=True)

URL = 'https://github.com/pytorch/tutorials/raw/master/_static/'
FILENAME = 'mnist.pkl.gz'

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open('wb').write(content)

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f,
                                                              encoding='latin-1')

from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap='gray')
print(x_train.shape)

import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

# 바닥부터 신경망 구현하기 (torch.nn 없이)
import math

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

# 보이는데로 preds tensor는 tensor 값 뿐만 아니라 변화고 함수도 포함합니다.
# 이것을 나중에 역전파를 하기 위해 사용할 것입니다.
# 손실 함수로서 사용하기 위해 음의 로그 우도(negative log-likelihood)를 구현하겠습니다.

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb))

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds, yb))

# 이제 훈련 루프(training loop)를 작동할 수 있습니다.
# 매 반복마다 다음을 할 것입니다.