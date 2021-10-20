import torch
import math

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")   # Uncomment this to run on GPU

# 입력과 출력을 갖기 위한 Tensor 생성하기
# 기본값으로 requires_grad=False를 합니다.
# 이는 역전파 동안 Tensor 에 관한 변화도를 계산하지 않는다는 의미입니다.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# 가중치에 대한 무작위 Tensor를 생성합니다.
# 3차 polynomia에서 4개의 가중치가 필요합니다.
# y = a + b x + c x^2 + d x^3
# 역전파 동안 Tensor에 대한 변화도를 계산하는 것을 나타내는
# requires_grad=True로 설정합니다.
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    # 순전파 : Tensor 연산을 사용해서 Y를 예측합니다.
    y_pred = a + b * x + c * x ** 2 + d * x **3

    # Tensor 연산을 사용해서 손실을 계산하고 출력합니다.
    # 이제 손실은 (1,) 형태의 Tensor 입니다.
    # loss.item()은 손실이 갖고 있는 scalar 값을 얻습니다.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # 역전파를 계산하기 위해 자동미분을 사용합니다.
    # 이는 requires_grad=True인 모든 Tensor 에 대한 변화도를 계산합니다.
    # 이것을 호출한 뒤에 a.grad, b.grad, c.grad, d.grad는 각각의 a,b,c,d에 대한
    # 손실의 변화도를 갖고 있는 Tensor 입니다.
    loss.backward()

    # 경사 하강법을 사용해서 수동으로 가중치를 갱신합니다.
    # torch.no_grad()로 감싸는 이유는 가중치는 requires_True를 갖고 있지만,
    # 자동미분은 이것을 추적할 필요가 없기 때문입니다.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # 가중치를 갱신한 후에 수동으로 변화도를 0으로 만듭니다.
        a.grad.zero_()
        b.grad.zero_()
        c.grad.zero_()
        d.grad.zero_()

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
