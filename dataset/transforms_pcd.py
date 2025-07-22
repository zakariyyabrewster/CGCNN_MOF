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


class PointCloudTransform:
    """
    Composable transforms for MOF point clouds.
    """
    def __init__(self, 
                 center: bool = True,
                 normalize: bool = True):
        
        self.center = center
        self.normalize = normalize
    
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
        
        return pcd