import numpy as np
import torch
from torch import nn


def center_pcd(pcd: torch.Tensor) -> torch.Tensor:
    """
    Center the point cloud data around the origin.

    Args:
        pcd (torch.Tensor): Point cloud data of shape (N, 3), where N is the number of points.

    Returns:
        torch.Tensor: Centered point cloud data.
    """
    centroid = pcd.mean(dim=0, keepdim=True)
    centered_pcd = pcd - centroid
    return centered_pcd

def normalize_pcd(pcd: torch.Tensor) -> torch.Tensor:
    """
    Normalize the point cloud data to have zero mean and unit variance.

    Args:
        pcd (torch.Tensor): Point cloud data of shape (N, 3), where N is the number of points.

    Returns:
        torch.Tensor: Normalized point cloud data.
    """
    mean = pcd.mean(dim=0, keepdim=True)
    std = pcd.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero
    normalized_pcd = (pcd - mean) / std
    return normalized_pcd

def random_rotation(pcd: torch.Tensor) -> torch.Tensor:
    """
    Apply random rotation to point cloud (data augmentation).
    
    Args:
        pcd (torch.Tensor): Point cloud coordinates of shape (N, 3)
        features (torch.Tensor, optional): Atom features of shape (N, F)
    
    Returns:
        tuple: (rotated_pcd, features) - features unchanged since rotation doesn't affect atom properties
    """
    # Random rotation matrix around each axis
    angles = torch.rand(3) * 2 * np.pi
    
    # Rotation matrices
    cos_x, sin_x = torch.cos(angles[0]), torch.sin(angles[0])
    cos_y, sin_y = torch.cos(angles[1]), torch.sin(angles[1])
    cos_z, sin_z = torch.cos(angles[2]), torch.sin(angles[2])
    
    Rx = torch.tensor([[1, 0, 0],
                       [0, cos_x, -sin_x],
                       [0, sin_x, cos_x]], dtype=pcd.dtype)
    
    Ry = torch.tensor([[cos_y, 0, sin_y],
                       [0, 1, 0],
                       [-sin_y, 0, cos_y]], dtype=pcd.dtype)
    
    Rz = torch.tensor([[cos_z, -sin_z, 0],
                       [sin_z, cos_z, 0],
                       [0, 0, 1]], dtype=pcd.dtype)
    
    # Combined rotation
    R = Rz @ Ry @ Rx
    rotated_pcd = pcd @ R.T
    
    return rotated_pcd


def random_jitter(pcd: torch.Tensor, noise_std: float = 0.01) -> tuple:
    """
    Add small random noise to atomic positions (data augmentation).
    
    Args:
        pcd (torch.Tensor): Point cloud coordinates of shape (N, 3)
        noise_std (float): Standard deviation of Gaussian noise
    
    Returns:
        torch.Tensor: jittered_pcd of shape (N, 3) - pcd w/ added noise
    """
    noise = torch.randn_like(pcd) * noise_std
    jittered_pcd = pcd + noise
    return jittered_pcd


class PointCloudTransform:
    """
    Composable transforms for MOF point clouds.
    """
    def __init__(self, 
                 center: bool = True,
                 rotation: bool = False,  # Training augmentation
                 jitter: bool = False,   # Training augmentation
                 jitter_std: float = 0.01):
        
        self.center = center

        self.rotation = rotation
        self.jitter = jitter
        self.jitter_std = jitter_std
    
    def __call__(self, pcd: torch.Tensor) -> torch.Tensor:
        """
        Apply transforms to point cloud and features.
        
        Args:
            pcd (torch.Tensor): Coordinates of shape (N, 3)
        
        Returns:
            torch.Tensor: Transformed point cloud.
        """
        # Center the point cloud
        if self.center:
            pcd = center_pcd(pcd)

        if self.normalize:
            pcd = normalize_pcd(pcd)
        
        # Data augmentation (training only)
        if self.rotation:
            pcd = random_rotation(pcd)
        
        if self.jitter:
            pcd = random_jitter(pcd, self.jitter_std)
        
        return pcd