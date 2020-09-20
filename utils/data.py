import random
from typing import List, Tuple, Dict, Callable, Iterable
from tqdm import tqdm

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Omniglot


class FSLDataLoader:
    def __init__(self, config: Dict, dataset: Omniglot):
        self.config = config
        self.rnd = np.random.RandomState(self.config['training']['random_seed'])
        images, labels = zip(*[(x, y) for (x, y) in tqdm(dataset, desc='Loading images in memory')])

        num_unique_classes = len(set(labels))
        sorted_images = [[] for _ in range(num_unique_classes)]

        for img, c in zip(images, labels):
            sorted_images[c].append(img.numpy())

        self.images = torch.from_numpy(np.array(sorted_images)) # [num_unique_classes, num_imgs_per_class, c, h, w]

    def sample_random_task(self) -> Tuple[DataLoader, DataLoader]:
        k = self.config['training']['num_classes_per_task']
        task_classes = self.rnd.choice(range(len(self.images)), replace=False, size=k)
        ds_train, ds_test = self.construct_datasets_for_classes(task_classes, shuffle_splits=True)

        return ds_train, ds_test

    def construct_datasets_for_classes(self, classes_to_use: List[int], shuffle_splits: bool=False) -> Tuple[Dataset, Dataset]:
        """
        It is guaranteed that examples are sorted by class order
        """
        k = self.config['training']['num_classes_per_task']
        n = self.config['training']['num_shots']
        num_imgs_per_class = len(self.images[0])
        c, h, w = self.images[0][0].shape

        task_imgs = self.images[classes_to_use] # [num_classes_per_task, num_imgs_per_class, c, h, w]
        task_labels = torch.arange(k).unsqueeze(1).repeat(1, num_imgs_per_class) # [num_classes_per_task, num_imgs_per_class]

        if shuffle_splits:
            task_imgs = task_imgs[:, self.rnd.permutation(task_imgs.shape[1])]

        task_imgs_train = task_imgs[:, :n].reshape(-1, c, h, w) # [num_classes_per_task * num_shots, c, h, w]
        task_imgs_test = task_imgs[:, n:].reshape(-1, c, h, w) # [num_classes_per_task * (num_imgs_per_class - num_shots), c, h, w]
        task_labels_train = task_labels[:, :n].reshape(-1) # [num_classes_per_task * num_shots]
        task_labels_test = task_labels[:, n:].reshape(-1) # [num_classes_per_task * (num_imgs_per_class - num_shots)]

        assert len(task_imgs_train) == len(task_labels_train), f"Wrong sizes: {len(task_imgs_train)} != {len(task_labels_train)}"
        assert len(task_imgs_test) == len(task_labels_test), f"Wrong sizes: {len(task_imgs_test)} != {len(task_labels_test)}"

        task_dataset_train = list(zip(task_imgs_train, task_labels_train))
        task_dataset_test = list(zip(task_imgs_test, task_labels_test))

        # task_dataset_train = Dataset(task_dataset_train)
        # task_dataset_test = Dataset(task_dataset_test)

        return task_dataset_train, task_dataset_test

    def __iter__(self) -> Iterable[Tuple[DataLoader, DataLoader]]:
        k = self.config['training']['num_classes_per_task']
        classes_order = self.rnd.permutation(len(self.images)) # [num_classes_total]
        classes_order = classes_order[:-(len(classes_order) % k)] # Drop last classes
        assert len(classes_order) % k == 0
        classes_order = classes_order.reshape(len(classes_order) // k, k) # [num_tasks, num_classes_per_task]

        return iter([self.construct_datasets_for_classes(cs) for cs in classes_order])

    def __len__(self) -> int:
        return len(self.images) // self.config['training']['num_classes_per_task']


def get_datasets(config: Dict) -> Tuple[Dataset, Dataset]:
    transform = get_transform(config)
    ds_train = Omniglot(config['data']['root_dir'], background=True, download=True, transform=transform)
    ds_test = Omniglot(config['data']['root_dir'], background=False, download=True, transform=transform)

    return ds_train, ds_test


def get_transform(config: Dict) -> Callable:
    return transforms.Compose([
        transforms.Resize(config['data']['target_img_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
