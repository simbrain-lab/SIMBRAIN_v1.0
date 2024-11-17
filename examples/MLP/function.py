from torch.autograd import Function
from torch import Tensor
from typing import Tuple

import sys
sys.path.append('../../')
from simbrain.mapping import MLPMapping

class MemLinearFunction(Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Tensor, crossbar: MLPMapping) -> Tensor:
        # output_ref = input @ weight.T + bias[None, ...]
        output = crossbar.mapping_read_mlp(target_v=input)

        if bias is not None:
            output += bias

        ctx.save_for_backward(input, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        input, weight = ctx.saved_tensors
        grad_input = grad_output @ weight
        grad_weight = grad_output.T @ input
        grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias, None