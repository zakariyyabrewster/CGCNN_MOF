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

from dataset.dataset_finetune_cgcnn import CIFData
from dataset.dataset_finetune_cgcnn import collate_pool, kcv_loader
from model.cgcnn_finetune import CrystalGraphConvNet

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_kcv_cgcnn.yaml', os.path.join(model_checkpoints_folder, 'config_kcv_cgcnn.yaml'))


class KCV_CGCNN(object):
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
        new_config.pop('fold_dir', None)

        self.dataset = CIFData(self.config['task'], **new_config) # use this if you dont have .npz files of preloaded crystal graphs
        # self.dataset = CGData(self.config['task'], **self.config['dataset'], shuffle=False)
        self.random_seed = self.config['random_seed']
        collate_fn = collate_pool
        
        self.train_loader, self.valid_loader, self.test_loader = kcv_loader(
            dataset = self.dataset,
            random_seed = self.random_seed,
            collate_fn = collate_fn,
            pin_memory = self.config['gpu'] != 'cpu',
            batch_size = self.config['batch_size'], 
            return_test = True,
            **self.config['dataloader']
        )

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

    def train(self):
        print("Training CGCNN on {} for {}...".format(self.config['data_name'], self.config['target_property']))
        structures, _, _ = self.dataset[0] # dataset[0] = (atom_fea, nbr_fea, nbr_fea_idx)
        orig_atom_fea_len = structures[0].shape[-1] # number of atom features used in embedding (92 in atom_init)
        nbr_fea_len = structures[1].shape[-1] # number of neighbor features used (depends on filter --> np.arange(dmin, dmax+step, step))
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    classification=(self.config['task']=='classification'), 
                                    **self.config['model']
        )

        # model = self._load_pre_trained_weights(model)

        if self.config['cuda']:
            model = model.to(self.device)
        #print(len(model))
        #pytorch_total_params = sum(p.numel() for p in model.parameters if p.requires_grad)
        #print(pytorch_total_params)
        layer_list = []
        for name, param in model.named_parameters():
            if 'fc_out' in name:
                # print(name, 'new layer')
                layer_list.append(name) # grab fc_out layers --> transfer learning (finetune) if using pretrained model
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters())))) # params for fc_out layers 
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters())))) # params for conv layers + hidden fc layers


        if self.config['optim']['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                [{'params': base_params, 'lr': self.config['optim']['lr']*0.2}, {'params': params}], # separate base params and fc_out params to have different learning rates
                 self.config['optim']['lr'], momentum=self.config['optim']['momentum'], 
                weight_decay=eval(self.config['optim']['weight_decay'])
            )
        elif self.config['optim']['optimizer'] == 'Adam':
            lr_multiplier = 0.2
            if 'scratch' in self.config['fine_tune_from']: # no difference in lr when ftf scratch --> more training required, larger lr for efficiency
                lr_multiplier = 1
            optimizer = optim.Adam(
                [{'params': base_params, 'lr': self.config['optim']['lr']*lr_multiplier}, {'params': params}],
                self.config['optim']['lr'], weight_decay=eval(self.config['optim']['weight_decay'])
            )
        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')        
        
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_mae = np.inf
        best_valid_roc_auc = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, (input, target, _) in enumerate(self.train_loader):
                if self.config['cuda']:
                    input_var = (Variable(input[0].to(self.device, non_blocking=True)),
                                Variable(input[1].to(self.device, non_blocking=True)),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                Variable(input[1]),
                                input[2],
                                input[3])
                
                target_normed = self.normalizer.norm(target)

                
                if self.config['cuda']:
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output, _ = model(*input_var)

                # print(output.shape, target_var.shape)
                loss = self.criterion(output, target_var)

                if bn % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    # self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print('Epoch: %d, Batch: %d, Loss:'%(epoch_counter+1, bn), loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss, valid_mae = self._validate(model, self.valid_loader, epoch_counter)
                if valid_mae < best_valid_mae:
                    # save the model weights
                    best_valid_mae = valid_mae
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, f"model_fold{self.config['dataloader']['fold']}.pth"))

                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
        self.model = model
           
    def _load_pre_trained_weights(self, model):
        try:
            # checkpoints_folder = os.path.join(self.config['fine_tune_from'], 'checkpoints')
            checkpoints_folder = self.config['fine_tune_from']
            print(os.path.join(checkpoints_folder, 'model_graph_3.pth'))
            load_state = torch.load(os.path.join(checkpoints_folder, 'model_graph_3.pth'),  map_location=self.config['gpu']) 
 
            # checkpoint = torch.load('model_best.pth.tar', map_location=args.gpu)
            # load_state = checkpoint['state_dict']
            model_state = model.state_dict()

            #pytorch_total_params = sum(p.numel() for p in model_state.parameters if p.requires_grad)
            #print(pytorch_total_params)
            for name, param in load_state.items():
                if name not in model_state:
                    print('NOT loaded:', name)
                    continue
                else:
                    print('loaded:', name)
                if isinstance(param, nn.parameter.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                model_state[name].copy_(param)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, n_epoch):
        losses = AverageMeter()
        mae_errors = AverageMeter()
        print('Validating model at epoch {0}...'.format(n_epoch+1))

        with torch.no_grad():
            model.eval()
            for bn, (input, target, _) in enumerate(valid_loader):
                if self.config['cuda']:
                    input_var = (Variable(input[0].to(self.device, non_blocking=True)),
                                Variable(input[1].to(self.device, non_blocking=True)),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                Variable(input[1]),
                                input[2],
                                input[3])
                
                target_normed = self.normalizer.norm(target)
                
                if self.config['cuda']:
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output, _ = model(*input_var)
        
                loss = self.criterion(output, target_var)

                mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))

            
            print('Epoch [{0}] Validate: [{1}/{2}], '
                    'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                n_epoch+1, bn+1, len(self.valid_loader), loss=losses,
                mae_errors=mae_errors))

        
        model.train()

        print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg


    
    def test(self):
        # test steps
        print("Testing CGCNN on {} for {}...".format(self.config['data_name'], self.config['target_property']))
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', "model_fold{}.pth".format(self.config['dataloader']['fold']))
        print(model_path)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        losses = AverageMeter()
        mae_errors = AverageMeter()

        
        test_targets = []
        test_preds = []
        test_cif_ids = []

        with torch.no_grad():
            self.model.eval()
            for bn, (input, target, batch_cif_ids) in enumerate(self.test_loader):
                if self.config['cuda']:
                    input_var = (Variable(input[0].to(self.device, non_blocking=True)),
                                Variable(input[1].to(self.device, non_blocking=True)),
                                input[2].to(self.device, non_blocking=True),
                                [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                Variable(input[1]),
                                input[2],
                                input[3])
                
                target_normed = self.normalizer.norm(target)

                
                if self.config['cuda']:
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output, _ = self.model(*input_var)
        
                loss = self.criterion(output, target_var) # MSE Loss

                mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
                
                test_pred = self.normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

            
            print('Test: [{0}/{1}], '
                    'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                bn+1, len(self.valid_loader), loss=losses,
                mae_errors=mae_errors))


        with open(os.path.join(self.writer.log_dir, 'test_results_{}.csv'.format(target_property)), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['cif_id', 'target', 'pred'])
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
        
        self.model.train()

        print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
    parser.add_argument('--seed', default=1, type=int,
                        metavar='Seed', help='random seed for splitting data (default: 1)')
    parser.add_argument('--target_property', type=str, help="Target property to override in config", default='Di')

    args = parser.parse_args(sys.argv[1:])

    config = yaml.load(open("config_kcv_cgcnn.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    config['random_seed'] = args.seed
    config['target_property'] = args.target_property

    task_name = config['data_name']
    # ftf: finetuning from
    if 'scratch' not in config['fine_tune_from']:
        # ftf = config['fine_tune_from'].split('/')[-1]
        ftf = 'pretrained'
    else:
        ftf = 'scratch'

    seed = config['random_seed']
    target_property = config['target_property']
    fold_dir = config['dataset']['fold_dir']


    fold_results = []

    num_folds = config['dataloader']['num_folds']
    for fold in range(num_folds):
        print("Fold {}".format(fold))
        config['dataloader']['fold'] = fold

        train_names = pd.read_csv(os.path.join(fold_dir, 'train_val', "fold_{}_train.csv".format(fold)))['MOFname'].tolist()
        valid_names = pd.read_csv(os.path.join(fold_dir, 'train_val', "fold_{}_val.csv".format(fold)))['MOFname'].tolist()
        test_names = pd.read_csv(os.path.join(fold_dir, 'test_holdout.csv'))['MOFname'].tolist()

        config['dataloader']['train_mofnames'] = train_names
        config['dataloader']['val_mofnames'] = valid_names
        config['dataloader']['test_mofnames'] = test_names

        log_dir = os.path.join(
            'training_results/finetuning/CGCNN_CV',
            "CGCNN_fold_{}_{}".format(fold, target_property)
        )
        os.makedirs(log_dir, exist_ok=True)

        fine_tune = KCV_CGCNN(config, log_dir)
        fine_tune.train()
        loss, metric = fine_tune.test()

        result_df = pd.DataFrame([[fold, loss, metric]], columns=['Fold', 'MSE Loss', 'MAE Loss'])
        result_df.to_csv(
            os.path.join(log_dir, 'fold_results.csv'),
            mode='a', index=False, header=not os.path.exists(os.path.join(log_dir, 'fold_results.csv'))
        )
        fold_results.append((fold, loss, metric))
    
    all_results_df = pd.DataFrame(fold_results, columns=['Fold', 'MSE Loss', 'MAE Loss'])
    all_results_df.to_csv(
        "training_results/finetuning/CGCNN_CV/cv_results_{}.csv".format(target_property),
        mode='a', index=False, header=True
    )