from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from .pure_layers import (
    PureModule,
    PureSequential,
    PureLinear,
    PureConv2d,
    PureProxy,
)


class LeNet(nn.Sequential):
    """
    Very similar to LeNet, but:
        - we use ReLU instead of Tanh
        - we add AdaptiveAvgPool2d since our image sizes are different from 28x28
    """
    def __init__(self, config: Dict):
        layers = [
            # Body
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5),
            nn.ReLU(),

            # Neck
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            # Head
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, config['training']['num_classes_per_task']),
        ]

        super().__init__(*layers)


class PureLeNet(PureModule):
    """
    LeNet architecture above, but rewritten as a pure module
    """
    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.model = PureSequential(
            # Body
            PureConv2d(1, 6, 5),
            PureProxy(nn.ReLU()),
            PureProxy(nn.AvgPool2d(2)),
            PureConv2d(6, 16, 5),
            PureProxy(nn.ReLU()),
            PureProxy(nn.AvgPool2d(2)),
            PureConv2d(16, 120, 5),
            PureProxy(nn.ReLU()),

            # Neck
            PureProxy(nn.AdaptiveAvgPool2d((1, 1))),
            PureProxy(nn.Flatten()),

            # Head
            PureLinear(120, 84),
            PureProxy(nn.ReLU()),
            PureLinear(84, config['training']['num_classes_per_task']),
        )

        self.size = self.model.size
        self.params = None

    def get_initial_params(self):
        return self.model.get_initial_params()

    def bind_params(self, params: Tensor):
        self.params = params

    def unbind_params(self):
        self.params = None

    def forward(self, x: Tensor, params: Tensor=None):
        params = params if not params is None else self.params

        return self.model(x, params)

