from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .pure_layers import PureModule


class MAMLModel:
    def __init__(self, config: Dict, target_model: PureModule):
        self.config = config
        self.target_model = target_model

        # TODO(maml): initialize parameters.
        # Hint: check .get_initial_params() method
        self.params = "TODO"

    def train(self, mode: bool=True):
        self.target_model.train(mode)

    def eval(self):
        self.target_model.eval()

    def to(self, *args, **kwargs):
        self.params = nn.Parameter(self.params.to(*args, **kwargs))

        return self

    def parameters(self) -> List[nn.Parameter]:
        return [self.params]

    def __call__(self, x: Tensor) -> Tensor:
        # TODO: perform a forward pass
        # Hint: you should call target_model here with your params
        return "TODO"
