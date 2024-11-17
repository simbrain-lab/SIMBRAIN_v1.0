import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import sys
from typing import Optional

sys.path.append('../../')
from simbrain.mapping import CNNMapping, MLPMapping
from function import MemLinearFunction, MemConv2dFunction


class Mem_Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, mem_device: dict, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

        self.crossbar = MLPMapping(sim_params=mem_device, shape=(in_features, out_features))
        self.crossbar.set_batch_size_mlp(1)


    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input: Tensor) -> Tensor:
        return MemLinearFunction.apply(input, self.weight, self.bias, self.crossbar)


    def mem_update(self):
        # Memristor crossbar program
        self.crossbar.mapping_write_mlp(target_x=self.weight.T.unsqueeze(0))


class Mem_Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        mem_device: dict = {},
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = torch.nn.Parameter(torch.empty((out_channels, in_channels // groups, kernel_size, kernel_size),
                                                     **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.crossbar = CNNMapping(sim_params=mem_device,
                                       shape=((in_channels // groups * kernel_size * kernel_size), out_channels))
        self.crossbar.set_batch_size_cnn(1)


    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input: Tensor) -> Tensor:
        return MemConv2dFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.crossbar)


    def mem_update(self):
        # Reshape weight
        out_channels = self.weight.size(0)
        weight_reshape = self.weight.reshape(out_channels, -1)

        # Memristor crossbar program
        self.crossbar.mapping_write_cnn(target_x=weight_reshape.T.unsqueeze(0))
