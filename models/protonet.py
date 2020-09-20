from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class ProtoNet(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        self.config = config

        # TODO(protonet): your code here
        # Use the same embedder as in LeNet
        self.embedder = "TODO"

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            - x: images of shape [num_classes * batch_size, c, h, w]
        """
        # Aggregating across batch-size
        num_classes = self.config['training']['num_classes_per_task']
        batch_size = len(x) // num_classes
        c, h, w = x.shape[1:]

        embeddings = self.embedder(x) # [num_classes * batch_size, dim]

        # TODO(protonet): compute prototypes given the embeddings
        prototypes = "TODO"

        # TODO(protonet): copmute the logits based on embeddings and prototypes similarity
        # You can use either L2-distance or cosine similarity
        logits = "TODO"

        return logits
