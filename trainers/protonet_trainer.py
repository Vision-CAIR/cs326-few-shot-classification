from typing import Tuple
import numpy as np
import torch
from .trainer import Trainer


class ProtoNetTrainer(Trainer):
    def sample_batch(self, dataset):
        """
        In ProtoNet we require that the batch contains equal number of examples
        per each class. We do this so it is simpler to compute prototypes
        """
        k = self.config['training']['num_classes_per_task']
        num_shots = len(dataset) // k
        batch_size = min(self.config['training']['batch_size'], num_shots)

        idx = [(c * num_shots + i) for c in range(k) for i in self.rnd.permutation(num_shots)[:batch_size]]
        x = torch.stack([dataset[i][0] for i in idx])
        y = torch.stack([dataset[i][1] for i in idx])

        return x, y

    def fine_tune(self, ds_train, ds_test) -> Tuple[float, float]:
        # TODO(protonet): your code goes here
        # How does ProtoNet operate in the inference stage?
        train_scores = "TODO"
        test_scores = "TODO"

        return train_scores, test_scores