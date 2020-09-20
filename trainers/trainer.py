from typing import Dict, Tuple

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from models import init_model
from utils.data import FSLDataLoader


class Trainer:
    def __init__(self, config: Dict, source_dl: FSLDataLoader, target_dl: FSLDataLoader):
        self.config = config
        self.rnd = np.random.RandomState(self.config['training']['random_seed'])
        self.model = init_model(config).to(config['device'])

        self.optim = torch.optim.Adam(self.model.parameters(), **self.config['training']['optim_kwargs'])

        self.source_dataloader = source_dl
        self.target_dataloader = target_dl

    def train_on_episode(self, model, optim, ds_train, ds_test=None):
        losses = []
        accs = []

        model.train()

        for it in range(self.config['training']['num_train_steps_per_episode']):
            x, y = self.sample_batch(ds_train)

            x = x.to(self.config['device'])
            y = y.to(self.config['device'])

            logits = model(x) # [batch_size, num_classes_per_task]
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            accs.append(acc.item())

        return losses[-1], accs[-1]

    def sample_batch(self, dataset):
        batch_size = min(self.config['training']['batch_size'], len(dataset))
        idx = self.rnd.choice(np.arange(len(dataset)), size=batch_size, replace=False)
        x = torch.stack([dataset[i][0] for i in idx])
        y = torch.stack([dataset[i][1] for i in idx])

        return x, y

    @torch.no_grad()
    def compute_scores(self, model, dataset) -> Tuple[np.float, np.float]:
        """
        Computes loss/acc for the dataloader
        """
        model.eval()

        x = torch.stack([s[0] for s in dataset]).to(self.config['device'])
        y = torch.stack([s[1] for s in dataset]).to(self.config['device'])
        logits = model(x)
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()

        return loss, acc

    def fine_tune(self, ds_train, ds_test) -> Tuple[float, float]:
        curr_model = init_model(self.config).to(self.config['device'])
        curr_model.load_state_dict(self.model.state_dict())
        curr_optim = torch.optim.Adam(curr_model.parameters(), **self.config['training']['optim_kwargs'])

        train_scores = self.train_on_episode(curr_model, curr_optim, ds_train, ds_test)
        test_scores = self.compute_scores(curr_model, ds_test)

        return train_scores, test_scores

    def train(self):
        episodes = tqdm(range(self.config['training']['num_train_episodes']))

        for ep in episodes:
            ds_train, ds_test = self.source_dataloader.sample_random_task()
            ep_train_loss, ep_train_acc = self.train_on_episode(self.model, self.optim, ds_train, ds_test)

            episodes.set_description(f'[Episode {ep: 03d}] Loss: {ep_train_loss: .3f}. Acc: {ep_train_acc: .03f}')

    def evaluate(self):
        """
        For evaluation, we should
        """
        scores = {'train': [], 'test': []}
        episodes = tqdm(self.target_dataloader, desc='Evaluating')

        for ep, (ds_train, ds_test) in enumerate(episodes):
            train_scores, test_scores = self.fine_tune(ds_train, ds_test)
            scores['train'].append(train_scores)
            scores['test'].append(test_scores)

            episodes.set_description(f'[Test episode {ep: 03d}] Loss: {test_scores[0]: .3f}. Acc: {test_scores[1]: .03f}')

        for split in scores:
            split_scores = np.array(scores[split])
            print(f'[EVAL] Mean {split} loss: {split_scores[:, 0].mean(): .03f}.')
            print(f'[EVAL] Mean {split} acc: {split_scores[:, 1].mean(): .03f}.')
