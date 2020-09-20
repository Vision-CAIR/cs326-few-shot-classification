from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .trainer import Trainer
from models import init_model


class MAMLTrainer(Trainer):
    def train_on_episode(self, model, optim, ds_train, ds_test):
        losses = []
        accs = []

        model.train()

        fast_w = model.params

        for it in range(self.config['model']['num_inner_steps']):
            x, y = self.sample_batch(ds_train)

            x = x.to(self.config['device'])
            y = y.to(self.config['device'])

            # TODO(maml): perform forward pass, compute logits, loss and accuracy
            logits = "TODO"
            loss = "TODO"
            acc = "TODO"

            # TODO(maml): compute the gradient and update the fast weights
            # Hint: think hard about it. This is maybe the hardest part of the assignment
            # You will likely need to check open-source implementations to get the idea of how things work
            grad = "TODO"
            fast_w = "TODO"

            losses.append(loss.item())
            accs.append(acc.item())

        x = torch.stack([s[0] for s in ds_test]).to(self.config['device'])
        y = torch.stack([s[1] for s in ds_test]).to(self.config['device'])

        # TODO(maml): compute the logits, outer-loop loss and outer-loop accuracy
        logits = "TODO"
        outer_loss = "TODO"
        outer_acc = "TODO"

        optim.zero_grad()
        outer_loss.backward()
        # TODO(maml): you may like to add gradient clipping here
        optim.step()

        return losses[-1], accs[-1]

    def fine_tune(self, ds_train, ds_test) -> Tuple[float, float]:
        curr_model = init_model(self.config).to(self.config['device'])
        curr_model.params.data.copy_(self.model.params.data)
        curr_optim = torch.optim.Adam(curr_model.parameters(), **self.config['model']['ft_optim_kwargs'])

        train_scores = Trainer.train_on_episode(self, curr_model, curr_optim, ds_train)
        test_scores = self.compute_scores(curr_model, ds_test)

        return train_scores, test_scores
