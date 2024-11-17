import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from module import *


model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
}

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        self.layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            self.layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            self.layers['relu{}'.format(i+1)] = nn.ReLU()
            self.layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        self.layers['out'] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(self.layers)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)


class Mem_MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class, mem_device):
        super(Mem_MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        self.layers = OrderedDict()

        if isinstance(n_hiddens, int):
            self.n_hiddens = [n_hiddens]
        else:
            self.n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(self.n_hiddens):
            self.layers['fc{}'.format(i+1)] = Mem_Linear(current_dims, n_hidden, mem_device)
            self.layers['relu{}'.format(i+1)] = nn.ReLU()
            self.layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        self.layers['out'] = Mem_Linear(current_dims, n_class, mem_device)

        self.model = nn.Sequential(self.layers)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)


def mlp_mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=None):
    model = MLP(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['mnist'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

def mem_mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, mem_device={}, pretrained=None):
    model = Mem_MLP(input_dims, n_hiddens, n_class, mem_device)
    if pretrained is True:
        m = model_zoo.load_url(model_urls['mnist'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        # State_dict adaption
        adapt_state_dict = OrderedDict()
        for k, v in model.state_dict().items():
            if k in state_dict.keys():
                adapt_state_dict[k] = state_dict[k]
            else:
                adapt_state_dict[k] = v
        model.load_state_dict(adapt_state_dict)
    return model