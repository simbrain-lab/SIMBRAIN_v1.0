import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import sys

sys.path.append('../../')
from simbrain.mapping import MLPMapping
from function import MemLinearFunction


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
