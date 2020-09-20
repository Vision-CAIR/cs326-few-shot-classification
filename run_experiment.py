#!/usr/bin/env python

import argparse

from trainers.trainer import Trainer
from trainers.protonet_trainer import ProtoNetTrainer
from trainers.maml_trainer import MAMLTrainer
from utils.config import construct_config
from utils.data import get_datasets, FSLDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('-m', '--method', default='unpretrained_baseline', type=str, help='Which method to run?')

    return parser.parse_args()


def fix_random_seed(seed: int):
    import random
    import torch
    import numpy

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(method: str):
    config = construct_config(method)
    fix_random_seed(config['training']['random_seed'])
    ds_train, ds_test = get_datasets(config)
    source_dl = FSLDataLoader(config, ds_train)
    target_dl = FSLDataLoader(config, ds_test)
    # source_dl = target_dl

    if method in ['unpretrained_baseline', 'pretrained_baseline']:
        trainer = Trainer(config, source_dl, target_dl)
    elif method == 'protonet':
        trainer = ProtoNetTrainer(config, source_dl, target_dl)
    elif method == 'maml':
        trainer = MAMLTrainer(config, source_dl, target_dl)
    else:
        raise NotImplementedError(f'Unknown method: {method}')

    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args.method)
