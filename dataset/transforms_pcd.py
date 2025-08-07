import numpy as np
import torch
from torch import nn
import random


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

class IdentityTransform:
    def __call__(self, pcd: torch.Tensor) -> torch.Tensor:
        return pcd

class RandomRotation:
    def __init__(self, axis=None):
        self.axis = axis if axis is not None else np.random.choice(['x', 'y', 'z'])
    
    def __call__(self, pcd: torch.Tensor) -> torch.Tensor:
        # input pcd shape: (N, 3))
        pc = pcd.clone()
        theta = np.random.uniform(0, 2 * np.pi)
        if self.axis == 'x':
            R = torch.tensor([[1, 0, 0],
                                            [0, np.cos(theta), -np.sin(theta)],
                                            [0, np.sin(theta), np.cos(theta)]], dtype=pc.dtype)
        elif self.axis == 'y':
            R = torch.tensor([[np.cos(theta), 0, np.sin(theta)],
                                            [0, 1, 0],
                                            [-np.sin(theta), 0, np.cos(theta)]], dtype=pc.dtype)
        else:
            R = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]], dtype=pc.dtype)

        pcd_rotated = (R @ pc.T).T

        return pcd_rotated

class PointCloudTransform:
    """
    Composable transforms for MOF point clouds.
    """
    def __init__(self):
        self.transforms = [
            IdentityTransform(),
            RandomRotation()
        ]
    
    def __call__(self, pcd: torch.Tensor) -> torch.Tensor:
        """
        Apply transforms to point cloud and features.
        
        Args:
            pcd (torch.Tensor): Coordinates of shape (N, 3)
        
        Returns:
            torch.Tensor: Transformed point cloud.
        """
        # Center the point cloud
        pcd = center_pcd(pcd)
        t = random.choice(self.transforms)
        pcd = t(pcd)
        return pcd