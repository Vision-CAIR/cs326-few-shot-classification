"""
For both MAML we need pure layers to be implemented,
i.e. layers that take as input both the activations from a previous layer and
parameters to run the transformation.
"""

from typing import List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PureModule:
    def __init__(self):
        self.size = 0
        self.training = True

    def train(self, enable_train_mode: bool=True):
        self.training = enable_train_mode

    def eval(self):
        self.train(False)

    def get_initial_params(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        raise NotImplementedError


class PureSequential(PureModule):
    def __init__(self, *submodules):
        self.submodules = submodules
        self.size = sum([s.size for s in submodules])

    def forward(self, x: Tensor, params: Tensor):
        assert len(params) == self.size, f"Wrong size: {params.shape}, expected: {self.size}"

        params_remaining = params

        for m in self.submodules:
            x = m(x, params_remaining[:m.size])
            params_remaining = params_remaining[m.size:]

        assert len(params_remaining) == 0, f"Not all params have been used: {params_remaining.shape}"

        return x

    def get_initial_params(self):
        return torch.cat([m.get_initial_params() for m in self.submodules])


class PureLinear(PureModule):
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_shape = (out_features, in_features)
        self.weight_size = np.prod(self.weight_shape)
        self.size = in_features * out_features + out_features # W_size + b_size

    def get_initial_params(self):
        return torch.cat([p.view(-1) for p in nn.Linear(self.in_features, self.out_features).parameters()])

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        weight = params[:self.weight_size].view(*self.weight_shape)
        bias = params[self.weight_size:]

        return F.linear(x, weight, bias)


class PureConv2d(PureModule):
    def __init__(self, in_features: int, out_features: int, kernel_size: int):
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.weight_shape = (out_features, in_features, kernel_size, kernel_size)
        self.weight_size = np.prod(self.weight_shape)
        self.size = self.weight_size + out_features # W_size + b_size

    def get_initial_params(self):
        return torch.cat([p.view(-1) for p in nn.Conv2d(self.in_features, self.out_features, self.kernel_size).parameters()])

    def compute_weight_scale(self) -> float:
        return 1 / np.sqrt(3 * self.in_features * self.kernel_size ** 2)

    def compute_bias_scale(self) -> float:
        return 1 / np.sqrt(self.in_features)

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        weight = params[:self.weight_size].view(*self.weight_shape)
        bias = params[-self.out_features:]

        return F.conv2d(x, weight, bias)


class PureProxy(PureModule):
    def __init__(self, fn: Callable):
        self.fn = fn
        self.size = 0

    def forward(self, x: Tensor, weights: Tensor) -> Tensor:
        return self.fn(x)

    def get_initial_params(self):
        return torch.tensor([])
