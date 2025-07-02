from cmath import log
import os
import csv
import yaml
import shutil
import sys
import time
import warnings
import numpy as np
import pandas as pd
from random import sample
from sklearn import metrics
from datetime import datetime
import argparse
from model.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset_finetune_cgcnn import CIFData, CGData
from dataset.dataset_finetune_cgcnn import collate_pool, get_train_val_test_loader, subset_train_val_test_loader
from model.cgcnn_finetune import CrystalGraphConvNet

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_ft_kcv_cgcnn.yaml', os.path.join(model_checkpoints_folder, 'config_ft_kcv_cgcnn.yaml'))


class KCVCGCNN(object):
    def __init__(self, config, log_dir):
        self.config = config
        self.device = self._get_device()

        self.writer = SummaryWriter(log_dir=log_dir)

        self.criterion = nn.MSELoss()

        # apply property filter 
        label_dir_template = self.config['dataset']['label_dir_template']
        target_property = self.config['target_property']
        new_label_dir = label_dir_template.format(target_property=target_property)
        new_config = self.config['dataset'].copy()
        new_config['label_dir'] = new_label_dir
        new_config.pop('label_dir_template', None)

        self.dataset = CIFData(self.config['task'], **new_config) # use this if you dont have .npz files of preloaded crystal graphs
        # self.dataset = CGData(self.config['task'], **self.config['dataset'], shuffle=False)
        self.random_seed = self.config['random_seed']
        collate_fn = collate_pool
        
        self.train_loader, self.valid_loader, self.test_loader = get_train_val_test_loader(
            dataset = self.dataset,
            random_seed = self.random_seed,
            collate_fn = collate_fn,
            pin_memory = self.config['gpu'] != 'cpu',
            batch_size = self.config['batch_size'], 
            return_test = True,
            **self.config['dataloader']
        )

        # self.train_loader, self.valid_loader, self.test_loader = subset_train_val_test_loader(
        #     dataset = self.dataset,
        #     random_seed = self.random_seed,
        #     collate_fn = collate_fn,
        #     pin_memory = self.config['gpu'] != 'cpu',
        #     batch_size = self.config['batch_size'], 
        #     return_test = True,
        #     **self.config['dataloader']
        # )

        # obtain target value normalizer
        if len(self.dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                        'Lower accuracy is expected. ')
            sample_data_list = [self.dataset[i] for i in range(len(self.dataset))]
        else:
            sample_data_list = [self.dataset[i] for i in
                                sample(range(len(self.dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        self.normalizer = Normalizer(sample_target)

    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        print("Running on:", device)

        return device