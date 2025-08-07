import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import csv
import os
import random
import json
import functools

class PointCloudData(Dataset):
    def __init__(self, root_dir, label_dir, random_seed, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.random_seed = random_seed
        id_prop_file = label_dir
        assert os.path.exists(id_prop_file), 'id_prop_hmof.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            self.id_prop_data = [row for row in reader]
        self.id_prop_data = self.id_prop_data
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join('benchmark_datasets/atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.transform = transform

    def __len__(self):
        return len(self.id_prop_data)
    
    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, index):
        cif_id, target = self.id_prop_data[index]
        crystal = Structure.from_file(os.path.join(self.root_dir, f'{cif_id}.cif'))
        
        # Get coordinates and features
        atom_coords = np.vstack([crystal[i].coords for i in range(len(crystal))])
        atom_coords = torch.tensor(atom_coords, dtype=torch.float32)
        
        atom_features = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
        atom_features = torch.tensor(atom_features, dtype=torch.float32)
        
        # Apply transforms if provided
        if self.transform:
            atom_coords = self.transform(atom_coords)
        pcd = torch.concat((atom_coords, atom_features), dim=1) # (n_atoms, 3 + 92) = (n_atoms, 95)

        pcd = pcd.T # Transpose to (95, n_atoms) for Conv1d input

        target = torch.tensor(float(target), dtype=torch.float32)
        return pcd, target, cif_id

class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def collate_pcd_upsample(batch):
    """
    Collate function with upsampling - upsamples smaller point clouds to match the largest one.
    Uses sampling with replacement to maintain consistent batch size for PointNet.
    
    Args:
        batch: List of (pcd, target, cif_id) tuples where pcd is (features, n_atoms)
        
    Returns:
        batch_pcd: (batch_size, features, max_n_atoms) with upsampled data
        batch_targets: (batch_size,) 
        batch_cif_ids: List of cif_ids
    """
    pcds, targets, cif_ids = zip(*batch)
    
    # Find maximum number of atoms in this batch
    max_n_atoms = max(pcd.shape[1] for pcd in pcds)
    features_dim = pcds[0].shape[0]  # Should be 95 (3 + 92)
    
    # Create upsampled tensor
    batch_pcd = torch.zeros(len(pcds), features_dim, max_n_atoms)
    
    # Fill in the data with upsampling
    for i, pcd in enumerate(pcds):
        n_atoms = pcd.shape[1]
        if n_atoms < max_n_atoms:
            # Upsample by sampling with replacement
            indices = torch.randint(0, n_atoms, (max_n_atoms,))
            batch_pcd[i, :, :] = pcd[:, indices]
        else:
            # Already at max size
            batch_pcd[i, :, :] = pcd
    
    batch_targets = torch.stack(targets)
    
    return batch_pcd, batch_targets, cif_ids

def get_train_val_test_loader_pcd(dataset, collate_fn=default_collate,
                              batch_size=64,random_seed = 2, val_ratio=0.1, test_ratio=0.1, 
                              return_test=False, num_workers=1, pin_memory=False, 
                              **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    train_ratio = 1 - val_ratio - test_ratio
    indices = list(range(total_size))
    print("The random seed is: ", random_seed)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)
    print('Train size: {}, Validation size: {}, Test size: {}'.format(
        train_size, valid_size, test_size
    ))
    
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

def kcv_loader(dataset, collate_fn=default_collate,
                                 batch_size=64, random_seed=1,
                                 return_test=False, num_workers=1, pin_memory=False,
                                 **kwargs):
    """
    Create data loaders using pre-defined train/val/test MOFname splits.

    Parameters
    ----------
    dataset : PointCloudData(Dataset)
        Full dataset (with dataset.df containing 'MOFname' column).
    collate_fn : function
        Function to collate data batches.
    batch_size : int
        Batch size for DataLoader.
    random_seed : int
        Not used here, but retained for logging/debugging.
    return_test : bool
        Whether to return a test_loader.
    num_workers : int
        Number of subprocesses to use for data loading.
    pin_memory : bool
        Whether to copy tensors into CUDA pinned memory.
    kwargs : dict
        Should contain 'train_mofnames', 'val_mofnames', and optionally 'test_mofnames'.

    Returns
    -------
    train_loader, val_loader [, test_loader]
    """

    train_mofnames = kwargs.get('train_mofnames')
    val_mofnames = kwargs.get('val_mofnames')
    test_mofnames = kwargs.get('test_mofnames')

    assert train_mofnames is not None and val_mofnames is not None, "Train and Val MOFname lists must be provided."

    print("Using pre-defined MOFname splits.")
    print(f"Random seed: {random_seed}")

    # Map MOFname → dataset index
    name_to_idx = {dataset.id_prop_data[i][0]: i for i in range(len(dataset))}

    train_idx = [name_to_idx[name] for name in train_mofnames if name in name_to_idx]
    val_idx = [name_to_idx[name] for name in val_mofnames if name in name_to_idx]
    test_idx = [name_to_idx[name] for name in test_mofnames if return_test and test_mofnames is not None and name in name_to_idx]

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)

    if return_test and test_mofnames is not None:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=SubsetRandomSampler(test_idx),
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
        return train_loader, val_loader, test_loader

    return train_loader, val_loader