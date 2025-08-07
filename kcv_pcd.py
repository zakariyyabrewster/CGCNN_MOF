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
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_pcd import PointCloudData, collate_pcd_upsample, kcv_loader
from model.pointcloud import PointNetLite
from dataset.transforms_pcd import *

warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_kcv_pcd.yaml', os.path.join(model_checkpoints_folder, 'config_kcv_pcd.yaml'))


class KCV_PCD(object):
    def __init__(self, config, log_dir):
        self.config = config
        self.device = self._get_device()

        self.writer = SummaryWriter(log_dir=log_dir)

        self.criterion = nn.MSELoss()

        label_dir_template = self.config['dataset']['label_dir_template']
        target_property = self.config['target_property']
        new_label_dir = label_dir_template.format(target_property=target_property)
        
        transforms = PointCloudTransform()

        # Only pass parameters that PointCloudData expects
        dataset_config = {
            'root_dir': self.config['dataset']['root_dir'],
            'label_dir': new_label_dir,
            'random_seed': self.config['dataset']['random_seed'],
            'transform': transforms
        }

        self.dataset = PointCloudData(**dataset_config)
        self.train_loader, self.valid_loader, self.test_loader = kcv_loader(
            dataset=self.dataset,
            collate_fn=collate_pcd_upsample,
            random_seed=self.config['random_seed'],
            pin_memory=self.config['gpu'] != 'cpu',
            batch_size=self.config['batch_size'],
            **self.config['dataloader'],
            return_test=True
        )
            
        train_indices = list(self.train_loader.sampler)

        # Randomly sample up to 500
        sample_indices = sample(train_indices, min(500, len(train_indices)))

        # Use dataset to get the samples
        sample_data_list = [self.dataset[i] for i in sample_indices]

        # Use your collate function to extract target tensor
        _, sample_target, _ = collate_pcd_upsample(sample_data_list)

        # Fit normalizer only on training targets
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
        print("Training PointNet on {} for {} (Fold {})...".format(
            self.config['data_name'], 
            self.config['target_property'],
            self.config['dataloader'].get('fold', 'Unknown')
        ))
        pcd = self.dataset[0]
        atom_feats = pcd[0].shape[0]  # Number of features per atom

        model = PointNetLite(atom_feats=atom_feats, output_dims=1)
        
        if self.config['cuda']:
            model = model.to(self.device)
            
        # Setup optimizer similar to CGCNN
        optimizer = optim.Adam(model.parameters(), 
                                 lr=self.config['optim']['lr'], 
                                 weight_decay=eval(self.config['optim']['weight_decay']))
        
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        
        # save config file
        _save_config_file(model_checkpoints_folder)
        
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_mae = np.inf
        
        # Lists to track losses for plotting
        train_losses = []
        valid_losses = []
        valid_epochs = []
        
        for epoch_counter in range(self.config['epochs']):
            epoch_train_losses = []
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
                
                # Track training loss for this batch
                epoch_train_losses.append(loss.item())
                
                if bn % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    print('Epoch: %d, Batch: %d, Loss:'%(epoch_counter+1, bn), loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1
            
            # Store average training loss for this epoch
            train_losses.append(np.mean(epoch_train_losses))

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss, valid_mae = self._validate(model, self.valid_loader, epoch_counter)
                
                # Store validation loss and epoch
                valid_losses.append(valid_loss)
                valid_epochs.append(epoch_counter)
                
                if valid_mae < best_valid_mae:
                    # save the model weights
                    best_valid_mae = valid_mae
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
        
        # Create and save loss plot
        self._plot_losses(train_losses, valid_losses, valid_epochs, model_checkpoints_folder)
        
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
        print("Testing PointNet on {} for {} (Fold {})...".format(
            self.config['data_name'], 
            self.config['target_property'],
            self.config['dataloader'].get('fold', 'Unknown')
        ))
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
        fold_num = self.config['dataloader'].get('fold', 'Unknown')
        with open(os.path.join(self.writer.log_dir, 'test_results_{}.csv'.format(fold_num, self.config['target_property'])), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['cif_id', 'target', 'pred'])
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
        
        self.model.train()
        print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return losses.avg, mae_errors.avg

    def _plot_losses(self, train_losses, valid_losses, valid_epochs, save_dir):
        """
        Plot training and validation losses and save the plot.
        
        Args:
            train_losses: List of training losses (one per epoch)
            valid_losses: List of validation losses
            valid_epochs: List of epochs where validation was performed
            save_dir: Directory to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        epochs = list(range(len(train_losses)))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        # Plot validation loss
        plt.plot(valid_epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2, marker='o')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        fold_num = self.config['dataloader'].get('fold', 'Unknown')
        plt.title(f'Training and Validation Loss - {self.config["target_property"]} (Fold {fold_num})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(save_dir, f'loss_plot_fold_{fold_num}_{self.config["target_property"]}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss plot saved to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='5-Fold Cross-Validation PointNet for MOF Property Prediction')
    parser.add_argument('--seed', default=1, type=int,
                        metavar='Seed', help='random seed for splitting data (default: 1)')
    parser.add_argument('--target_property', type=str, help="Target property to override in config", default='Di')

    args = parser.parse_args(sys.argv[1:])

    config = yaml.load(open("config_kcv_pcd.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    config['random_seed'] = args.seed
    config['target_property'] = args.target_property

    task_name = config['data_name']
    seed = config['random_seed']
    target_property = config['target_property']
    fold_dir = config['dataset']['fold_dir']

    fold_results = []

    num_folds = config['dataloader']['num_folds']
    for fold in range(num_folds):
        print("=" * 60)
        print(f"Starting Fold {fold}")
        print("=" * 60)
        config['dataloader']['fold'] = fold

        # Load fold-specific train/val/test splits
        train_names = pd.read_csv(os.path.join(fold_dir, 'train_val', "fold_{}_train.csv".format(fold)))['MOFname'].tolist()
        valid_names = pd.read_csv(os.path.join(fold_dir, 'train_val', "fold_{}_val.csv".format(fold)))['MOFname'].tolist()
        test_names = pd.read_csv(os.path.join(fold_dir, 'test_holdout.csv'))['MOFname'].tolist()

        config['dataloader']['train_mofnames'] = train_names
        config['dataloader']['val_mofnames'] = valid_names
        config['dataloader']['test_mofnames'] = test_names

        print(f"Train samples: {len(train_names)}")
        print(f"Validation samples: {len(valid_names)}")
        print(f"Test samples: {len(test_names)}")

        log_dir = os.path.join(
            'training_results/finetuning/PointNet_CV_V2',
            "PointNet_fold_{}_{}".format(fold, target_property)
        )
        os.makedirs(log_dir, exist_ok=True)

        fine_tune = KCV_PCD(config, log_dir)
        fine_tune.train()
        loss, metric = fine_tune.test()

        # Save fold results
        result_df = pd.DataFrame([[fold, loss, metric.item()]], columns=['Fold', 'MSE Loss', 'MAE Loss'])
        result_df.to_csv(
            os.path.join(log_dir, 'fold_results.csv'),
            mode='a', index=False, header=not os.path.exists(os.path.join(log_dir, 'fold_results.csv'))
        )
        fold_results.append((fold, loss, metric.item()))
            
    
    # Save overall cross-validation results
    cv_results_dir = "training_results/finetuning/PointNet_CV"
    os.makedirs(cv_results_dir, exist_ok=True)
    
    all_results_df = pd.DataFrame(fold_results, columns=['Fold', 'MSE Loss', 'MAE Loss'])
    all_results_df.to_csv(
        os.path.join(cv_results_dir, "cv_results_{}.csv".format(target_property)),
        index=False
    )