import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
    


class PointNetLite(nn.Module):
    def __init__(self, atom_feats: int, output_dims: int):
        '''
        input_features: 3 (x, y, z) + atom features
        output_features: global features (1024)

        n * (3 + C) -> MLP(64, 64) 
        -> n * (64) -> MLP(64, 128, 1024) -
        > n * (1024) -> MaxPool -> 1*1024 
        -> MLP(512, 256) -> fc_out -> prediction

        '''
        super(PointNetLite, self).__init__()

        input_dims = atom_feats  # x, y, z coordinates + atom features
        # Shared MLP1: input_dims (95) -> 64 -> 64
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dims, 64, kernel_size=1), 
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )

        # Global max pooling - adaptive to any number of atoms
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)

        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        self.fc_out = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dims, n_atoms)
        x = self.mlp1(x)  # (batch_size, 64, n_atoms)
        x = self.mlp2(x)  # (batch_size, 1024, n_atoms)
        x = self.global_maxpool(x)  # (batch_size, 1024, 1)
        x = x.squeeze(-1)  # (batch_size, 1024)
        x = self.mlp3(x)  # (batch_size, 256)
        x = self.fc_out(x)  # (batch_size, output_dims=1)
        return x
