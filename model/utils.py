import numpy as np
import torch
import re
import pandas as pd
from typing import Optional, Dict, Tuple, List

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))

def mse(prediction, target):
    return torch.mean(torch.square(target - prediction))

# def split_data(data, train_ratio, valid_ratio, use_ratio=1, randomSeed = None):
#       num_train = len(data)
#       indices = list(range(num_train))

#       random_state = np.random.RandomState(randomSeed)
#       random_state.shuffle(indices)

#       indices = indices[:int(len(indices)*use_ratio)]

#       split = int(np.floor(valid_ratio * len(indices)))
#       train_idx, valid_idx = indices[split:], indices[:split]

#       return data[train_idx], data[valid_idx]
def split_data(data, test_ratio, valid_ratio, use_ratio=1, randomSeed = None):
    total_size = len(data)
    train_ratio = 1 - valid_ratio - test_ratio
    indices = list(range(total_size))
    print("The random seed is: ", randomSeed)
    np.random.seed(randomSeed)
    np.random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = int(test_ratio * total_size)
    print('Train size: {}, Validation size: {}, Test size: {}'.format(
    train_size, valid_size, test_size
    ))

    train_idx, valid_idx, test_idx = indices[:train_size], indices[-(valid_size + test_size):-test_size], indices[-test_size:]



    return data[train_idx], data[valid_idx], data[test_idx]

def split_data_subset(data, test_ratio, valid_ratio, subset_size=500, randomSeed = None):
    total_size = len(data)
    train_ratio = 1 - valid_ratio - test_ratio
    indices = list(range(total_size))
    print("The random seed is: ", randomSeed)
    np.random.seed(randomSeed)
    np.random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = int(test_ratio * total_size)
    print('Train size: {}, Validation size: {}, Test size: {}'.format(
    subset_size, valid_size, test_size
    ))
    train_idx = np.random.choice(indices[:train_size], size=subset_size, replace=False)
    valid_idx, test_idx = indices[-(valid_size + test_size):-test_size], indices[-test_size:]



    return data[train_idx], data[valid_idx], data[test_idx]

def split_data_df(df, test_ratio, valid_ratio, random_seed=None):
    """
    Randomly split DataFrame into train/val/test by row position.
    - Handles 0% splits without -0 slicing issues
    - Rounds sizes, and gives any remainder to train
    - Uses modern numpy RNG for better reproducibility
    
    Args:
        df: DataFrame to split
        test_ratio: Fraction for test set (0.0 to 1.0)
        valid_ratio: Fraction for validation set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    # Input validation - ensure ratios are valid
    assert 0 <= test_ratio <= 1 and 0 <= valid_ratio <= 1, "ratios must be in [0,1]"
    assert test_ratio + valid_ratio <= 1, "valid + test must be <= 1"

    n = len(df)
    
    # Use modern numpy RNG for better random state management
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(n)  # Create random permutation of indices

    print("The random seed is:", random_seed)

    # Calculate split sizes using rounding (prevents data loss from truncation)
    n_test = int(round(test_ratio * n))
    n_valid = int(round(valid_ratio * n))
    n_train = n - n_valid - n_test  # Give any remainder to training set
    
    # Safety check - ensure splits don't exceed dataset size
    if n_train < 0:
        raise ValueError("Ratios too large for dataset size.")
    
    print('Train size: {}, Validation size: {}, Test size: {}'.format(
        n_train, n_valid, n_test
    ))

    # Create contiguous, non-overlapping slices over the single permutation
    # This avoids the -0 slicing bug and ensures proper splits
    train_idx = perm[:n_train]                      # First n_train indices
    valid_idx = perm[n_train:n_train + n_valid]     # Next n_valid indices  
    test_idx = perm[n_train + n_valid:]             # Remaining n_test indices

    # Return copies to prevent unintended mutations of the original DataFrame
    return df.iloc[train_idx].copy(), df.iloc[valid_idx].copy(), df.iloc[test_idx].copy()
