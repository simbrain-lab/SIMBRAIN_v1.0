import torch
from torch.autograd import Function
from torch import Tensor
from typing import Tuple, Optional

import sys
sys.path.append('../../')
from simbrain.mapping import MLPMapping, CNNMapping

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


class MemConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Tensor, stride: int, padding: int,
                crossbar: CNNMapping) -> Tensor:
        batch_size, channels, height, width = input.size()
        kernel_size = weight.size(2)
        out_channels = weight.size(0)

        # Compute output dimensions
        out_height = (height - kernel_size + 2 * padding) // stride + 1
        out_width = (width - kernel_size + 2 * padding) // stride + 1

        # Add padding to the input
        input_padded = torch.nn.functional.pad(input, (padding, padding, padding, padding))

        # Unfold the input tensor to extract patches
        unfolded_input = torch.nn.functional.unfold(input_padded, (kernel_size, kernel_size), stride=stride)

        # Reshape input to the memristor array input
        input_reshape = unfolded_input.transpose(1, 2)
        s0 = input_reshape.size(0)
        s1 = input_reshape.size(1)
        s2 = input_reshape.size(2)
        input_reshape = input_reshape.reshape(-1, s2)

        # Matrix-Multiplication
        # weight_reshape = weight.reshape(out_channels, -1).t()
        # out_unfolded = input_reshape.matmul(weight_reshape)
        out_unfolded = crossbar.mapping_read_cnn(target_v=input_reshape)

        # Reshape the output
        out_unfolded = out_unfolded.reshape(s0, s1, -1)
        out_unfolded = out_unfolded.transpose(1, 2)

        # Fold the output
        output = torch.nn.functional.fold(out_unfolded, (out_height, out_width), (1, 1))

        if bias is not None:
            output += bias.unsqueeze(-1).unsqueeze(-1)

        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        # Compute gradient of loss w.r.t. weights
        grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding)

        # Compute gradient of loss w.r.t. input
        grad_input = torch.nn.grad.conv2d_input(input.shape, weight.data, grad_output, stride=stride, padding=padding)

        if bias is not None:
            grad_bias = grad_output.sum(dim=[0, 2, 3])
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None