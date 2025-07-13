from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import TransformerRegressor, Transformer, RegressionHead
from model.utils import *
from datetime import datetime, timedelta
from time import time
from torch.utils.data import dataset, DataLoader

import os
import csv
import yaml
import shutil
import argparse
import sys
import time
import warnings
import numpy as np
import pandas as pd
from random import sample
from sklearn import metrics
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_finetune_transformer import MOF_ID_Dataset
# from dataset.dataset_finetune import collate_pool, get_train_val_test_loader
#from model.cgcnn_finetune import CrystalGraphConvNet

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_kcv_transformer.yaml', os.path.join(model_checkpoints_folder, 'config_kcv_transformer.yaml'))


class KCV_Transformer(object):
    def __init__(self, config, log_dir):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(log_dir=log_dir)

        self.random_seed = self.config['dataloader']['randomSeed']

        # self.mofdata = np.load(self.config['dataset']['dataPath'], allow_pickle=True)
        dataPathTemp = self.config['dataset']['dataPath']
        new_dataPath = dataPathTemp.format(target_property=self.config['target_property'])
        self.config['dataset']['dataPath'] = new_dataPath
        with open(self.config['dataset']['dataPath']) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            self.mofdata = [row for row in reader]
        self.mofdata = np.array(self.mofdata)
        self.vocab_path = self.config['vocab_path']
        self.tokenizer = MOFTokenizer(self.vocab_path, model_max_length = 512, padding_side='right')

        # Lookup table for MOFID and MOFname
        df = pd.read_csv(self.config['dataset']['dataPath'])
        df["MOFID"] = df["MOFID"].astype(str)
        df["MOFname"] = df["MOFname"].astype(str)

        self.mofid_to_mofname = dict(zip(df["MOFID"], df["MOFname"]))

        # Filter data based on fold assignments
        train_names = self.config['dataloader']['train_mofnames']
        val_names = self.config['dataloader']['val_mofnames']
        test_names = self.config['dataloader']['test_mofnames']

        # Filter mofdata based on MOFname assignments
        train_mask = np.isin(df["MOFname"], train_names)
        val_mask = np.isin(df["MOFname"], val_names)
        test_mask = np.isin(df["MOFname"], test_names)

        self.train_data = self.mofdata[train_mask]
        self.valid_data = self.mofdata[val_mask]
        self.test_data = self.mofdata[test_mask]
        
        self.train_dataset = MOF_ID_Dataset(data = self.train_data, tokenizer = self.tokenizer)
        self.valid_dataset = MOF_ID_Dataset(data = self.valid_data, tokenizer = self.tokenizer)
        self.test_dataset = MOF_ID_Dataset(data = self.test_data, tokenizer = self.tokenizer)
        
        self.train_loader = DataLoader(
        self.train_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], drop_last=False, 
        shuffle=True, pin_memory = False
        )

        self.valid_loader = DataLoader(
        self.valid_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], drop_last=False, 
        shuffle=False, pin_memory = False
        )

        self.test_loader = DataLoader(
        self.test_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], drop_last=False, 
        shuffle=False, pin_memory = False
        )

        self.criterion = nn.MSELoss()

        self.normalizer = Normalizer(torch.from_numpy(self.train_dataset.label))


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
        print("Training Transformer on {} for {}...".format(self.config['dataset']['data_name'], self.config['target_property']))
        self.transformer = Transformer(**self.config['Transformer'])
        # Load state dict
        if self.config['cuda']:
            self.transformer = self.transformer.to(self.device)

        model_transformer = self._load_pre_trained_weights(self.transformer)


        model = TransformerRegressor(transformer=model_transformer, d_model=512).to(self.device)
        
        if self.config['cuda']:
            model = model.to(self.device)

        #model = self._load_pre_trained_weights(model)
        #print(len(model))
        #pytorch_total_params = sum(p.numel() for p in model.parameters if p.requires_grad)
        #print(pytorch_total_params)
        layer_list = []
        for name, param in model.named_parameters():
            if 'fc_out' in name:
                print(name, 'new layer')
                layer_list.append(name)
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        if self.config['optim']['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                [{'params': base_params, 'lr': self.config['optim']['lr']*0.2}, {'params': params}],
                 self.config['optim']['init_lr'], momentum=self.config['optim']['momentum'], 
                weight_decay=eval(self.config['optim']['weight_decay'])
            )
        elif self.config['optim']['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                [{'params': base_params, 'lr': self.config['optim']['init_lr']*1}, {'params': params}],
                self.config['optim']['init_lr']*200, weight_decay=eval(self.config['optim']['weight_decay'])
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

        model.train()

        for epoch_counter in range(self.config['epochs']):
            for bn, (inputs, target) in enumerate(self.train_loader):
                if self.config['cuda']:
                    input_var = inputs.to(self.device)
                else:
                    input_var = inputs.to(self.device)
                
                target_normed = self.normalizer.norm(target)
                
                if self.config['cuda']:
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output = model(input_var)

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
            load_state = torch.load(os.path.join(checkpoints_folder, 'model_transformer_3.pth'),  map_location=self.config['gpu']) 
 
            model_state = model.state_dict()

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
            for bn, (inputs, target) in enumerate(valid_loader):
                if self.config['cuda']:
                    input_var = inputs.to(self.device)
                else:
                    input_var = inputs.to(self.device)
                
                target_normed = self.normalizer.norm(target)
                
                if self.config['cuda']:
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output = model(input_var)
        
                loss = self.criterion(output, target_var)

                mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))

            print('Epoch [%d] Validate: [{1}/{2}], '
                    'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                n_epoch+1, bn+1, len(self.valid_loader), loss=losses,
                mae_errors=mae_errors))
        
        model.train()

        print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg

    
    def test(self):
        # test steps
        print("Testing Transformer on {} for {}...".format(self.config['dataset']['data_name'], self.config['target_property']))
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', f"model_fold{self.config['dataloader']['fold']}.pth")
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
            for bn, (inputs, target) in enumerate(self.test_loader):

                input_var = inputs.to(self.device)
                
                target_normed = self.normalizer.norm(target)
                
                if self.config['cuda']:
                    target_var = Variable(target_normed.to(self.device, non_blocking=True))
                else:
                    target_var = Variable(target_normed)

                # compute output
                output = self.model(input_var)
        
                loss = self.criterion(output, target_var)

                mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
                
                test_pred = self.normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                
                batch_size = inputs.size(0)
                start_idx = bn * self.test_loader.batch_size
                end_idx = start_idx + batch_size

                batch_mofids = self.test_loader.dataset.mofid[start_idx:end_idx]
                batch_mofnames = [self.mofid_to_mofname.get(mid, "UNKNOWN") for mid in batch_mofids]
                test_cif_ids.extend(batch_mofnames)

            print('Test: [{0}/{1}], '
                    'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                bn, len(self.test_loader), loss=losses,
                mae_errors=mae_errors))
            

        with open(os.path.join(self.writer.log_dir, 'test_results_{}.csv'.format(self.config['target_property'])), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['cif_id', 'target', 'pred'])
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
        
        self.model.train()

        print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer k-fold cross validation')
    parser.add_argument('--seed', default=1, type=int,
                        metavar='Seed', help='random seed for splitting data (default: 1)')
    parser.add_argument('--target_property', type=str, help="Target property to override in config", default='Di')

    args = parser.parse_args(sys.argv[1:])

    config = yaml.load(open("config_kcv_transformer.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    config['dataloader']['randomSeed'] = args.seed
    config['target_property'] = args.target_property

    task_name = config['dataset']['data_name']
    
    # ftf: finetuning from
    # ptw: pre-trained with
    if config['fine_tune_from'] == 'scratch':
        ftf = 'scratch'
        ptw = 'scratch'
    else:
        ftf = config['fine_tune_from'].split('/')[-1]
        ptw = config['trained_with']

    seed = config['dataloader']['randomSeed']
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
            'training_results/finetuning/Transformer_CV',
            "Trans_fold_{}_{}".format(fold, target_property)
        )
        os.makedirs(log_dir, exist_ok=True)

        fine_tune = KCV_Transformer(config, log_dir)
        fine_tune.train()
        loss, metric = fine_tune.test()

        result_df = pd.DataFrame([[fold, loss, metric.item()]], columns=['Fold', 'MSE Loss', 'MAE Loss'])
        result_df.to_csv(
            os.path.join(log_dir, 'fold_results.csv'),
            mode='a', index=False, header=not os.path.exists(os.path.join(log_dir, 'fold_results.csv'))
        )
        fold_results.append((fold, loss, metric.item()))
    
    all_results_df = pd.DataFrame(fold_results, columns=['Fold', 'MSE Loss', 'MAE Loss'])
    all_results_df.to_csv(
        "training_results/finetuning/Transformer_CV/cv_results_{}.csv".format(target_property),
        mode='a', index=False, header=True
    )