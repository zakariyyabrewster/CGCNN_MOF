import os
import shutil
import csv
import yaml
from model.utils import *
from random import sample
import numpy as np
import pandas as pd
import argparse
import sys
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_pcd import PointCloudData, collate_pcd_padded, get_train_val_test_loader_pcd
from model.pointcloud import PointNetLite
from dataset.transforms_pcd import *

warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)



def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_ft_pcd.yaml', os.path.join(model_checkpoints_folder, 'config_ft_pcd.yaml'))

class FineTunePCD(object):
    def __init__(self, config, log_dir):
        self.config = config
        self.device = self._get_device()

        self.writer = SummaryWriter(log_dir=log_dir)

        self.criterion = nn.MSELoss()

        label_dir_template = self.config['dataset']['label_dir_template']
        target_property = self.config['target_property']
        new_label_dir = label_dir_template.format(target_property=target_property)
        new_config = self.config['dataset'].copy()
        new_config['label_dir'] = new_label_dir
        new_config.pop('label_dir_template', None)

        transforms = PointCloudTransform(
            center=self.config['dataset']['center'],
            normalize=self.config['dataset']['normalize'],
        )

        self.dataset = PointCloudData(**new_config, transform=transforms)
        self.train_loader, self.valid_loader, self.test_loader = get_train_val_test_loader_pcd(
            dataset=self.dataset,
            collate_fn=collate_pcd_padded,
            random_seed=self.config['random_seed'],
            pin_memory=self.config['gpu'] != 'cpu',
            batch_size=self.config['batch_size'],
            **self.config['dataloader'],
            return_test=True
        )

        # obtain target value normalizer
        if len(self.dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                        'Lower accuracy is expected. ')
            sample_data_list = [self.dataset[i] for i in range(len(self.dataset))]
        else:
            sample_data_list = [self.dataset[i] for i in
                                sample(range(len(self.dataset)), 500)]
        _, sample_target, _ = collate_pcd_padded(sample_data_list)
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
        print("Training PointNet on {} for {}...".format(self.config['data_name'], self.config['target_property']))
        pcd = self.dataset[0]
        atom_feats = pcd[0].shape[0]  # Number of features per atom
        model = PointNetLite(atom_feats=atom_feats, output_dims=1)
        
        if self.config['cuda']:
            model = model.to(self.device)
            
        # Setup optimizer similar to CGCNN
        if self.config['optim']['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), 
                                lr=self.config['optim']['lr'], 
                                momentum=self.config['optim']['momentum'], 
                                weight_decay=eval(self.config['optim']['weight_decay']))
        elif self.config['optim']['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), 
                                 lr=self.config['optim']['lr'], 
                                 weight_decay=eval(self.config['optim']['weight_decay']),
                                 momentum=self.config['optim']['momentum'])
        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')
        
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        
        # save config file
        _save_config_file(model_checkpoints_folder)
        
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_mae = np.inf
        
        for epoch_counter in range(self.config['epochs']):
            for bn, (batch_pcd, batch_targets, batch_cif_ids) in enumerate(self.train_loader):
                if self.config['cuda']:
                    batch_pcd = Variable(batch_pcd.to(self.device, non_blocking=True))
                    target_normed = self.normalizer.norm(batch_targets)
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    batch_pcd = Variable(batch_pcd)
                    target_normed = self.normalizer.norm(batch_targets)
                    target_var = Variable(target_normed)
                
                # compute output
                output = model(batch_pcd)
                loss = self.criterion(output.squeeze(), target_var)
                
                if bn % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
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
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
        
        self.model = model

    def _validate(self, model, valid_loader, n_epoch):
        losses = AverageMeter()
        mae_errors = AverageMeter()
        print('Validating model at epoch {0}...'.format(n_epoch+1))

        with torch.no_grad():
            model.eval()
            for bn, (batch_pcd, batch_targets, batch_cif_ids) in enumerate(valid_loader):
                if self.config['cuda']:
                    batch_pcd = Variable(batch_pcd.to(self.device, non_blocking=True))
                    target_normed = self.normalizer.norm(batch_targets)
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    batch_pcd = Variable(batch_pcd)
                    target_normed = self.normalizer.norm(batch_targets)
                    target_var = Variable(target_normed)

                # compute output
                output = model(batch_pcd)
                loss = self.criterion(output.squeeze(), target_var)

                mae_error = mae(self.normalizer.denorm(output.data.cpu()), batch_targets)
                losses.update(loss.data.cpu().item(), batch_targets.size(0))
                mae_errors.update(mae_error, batch_targets.size(0))

            print('Epoch [{0}] Validate: [{1}/{2}], '
                    'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                n_epoch+1, bn+1, len(valid_loader), loss=losses,
                mae_errors=mae_errors))

        model.train()
        print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg

    def test(self):
        # test steps
        print("Testing PointNet on {} for {}...".format(self.config['data_name'], self.config['target_property']))
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
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
            for bn, (batch_pcd, batch_targets, batch_cif_ids) in enumerate(self.test_loader):
                if self.config['cuda']:
                    batch_pcd = Variable(batch_pcd.to(self.device, non_blocking=True))
                    target_normed = self.normalizer.norm(batch_targets)
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    batch_pcd = Variable(batch_pcd)
                    target_normed = self.normalizer.norm(batch_targets)
                    target_var = Variable(target_normed)

                # compute output
                output = self.model(batch_pcd)
                loss = self.criterion(output.squeeze(), target_var)  # MSE Loss

                mae_error = mae(self.normalizer.denorm(output.data.cpu()), batch_targets)
                losses.update(loss.data.cpu().item(), batch_targets.size(0))
                mae_errors.update(mae_error, batch_targets.size(0))
                
                test_pred = self.normalizer.denorm(output.data.cpu())
                test_target = batch_targets
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

            print('Test: [{0}/{1}], '
                    'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                bn+1, len(self.test_loader), loss=losses,
                mae_errors=mae_errors))

        # Save test results
        with open(os.path.join(self.writer.log_dir, 'test_results_{}.csv'.format(self.config['target_property'])), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['cif_id', 'target', 'pred'])
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
        
        self.model.train()
        print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PointNet for MOF Property Prediction')
    parser.add_argument('--seed', default=1, type=int,
                        metavar='Seed', help='random seed for splitting data (default: 1)')
    parser.add_argument('--target_property', type=str, help="Target property to override in config", default='Di')

    args = parser.parse_args(sys.argv[1:])

    config = yaml.load(open("config_ft_pcd.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    config['random_seed'] = args.seed
    config['target_property'] = args.target_property
    
    task_name = config['data_name']
    seed = config['random_seed']
    target_property = config['target_property']

    log_dir = os.path.join(
        'training_results/finetuning/PointNet',
        'PointNet_{}_{}_{}_{}'.format('scratch', task_name, seed, target_property)
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fine_tune = FineTunePCD(config, log_dir)
    fine_tune.train()
    loss, metric = fine_tune.test()

    fn = 'PointNet_{}_{}_{}_{}.csv'.format('scratch', task_name, seed, target_property)
    df = pd.DataFrame([[loss, metric.item()]], 
                      columns=['MSE Loss', 'MAE Loss'])
    df.to_csv(
        os.path.join(log_dir, fn),
        mode='a', index=False, header=True
    )



                





                    
