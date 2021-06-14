import torch
import math

class LegendrePolynomial3(torch.autograd.Function):
    '''
    torch.autograd.Function을 상속받아 사용자 정의 자동미분 함수를 구현하고
    Tensor 연산을 하는 순전파와 역전파를 구현하겠습니다.
    '''

    @staticmethod
    def forward(ctx, input):
    """
    순전파에서 입력을 지닌 Tensor를 받고 결과값을 지닌 Tensor를 반환합니다.
    ctx는 역전파를 위한 정보를 저장하기 위해 사용되는 컨텍스트 객체(context object)입니다.
    ctx.save_for_backward를 사용하여 역전파단계에서 사용되는 속성 객체를 cache에 저장할 수 있습니다.
    """

    ctx.save_for_backward(input)
    return 0.5 * (5 input ** 3 - 3 * input)


