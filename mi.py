import os
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from sklearn.feature_selection import mutual_info_classif


def calc_MI_hy(features: Tensor, labels: Tensor, n_neighbors: int = 12) -> Tuple[float, float, float]:
    """
    calculate MI(feature, label), i.e., MI(h, y).
    
    :param features: 2D Tensor with the shape of (num_images, feature_dim)
    :param labels: 1D Tensor with the shape of (num_images,)
    :param n_neighbors: a integer value (a hyperparameter);
                        a higher value may reduce variance of the estimation, but could introduce a bias;
                        n_neighbors=12 is good enough by default.

    :return: (hy_mean, hy_max, hy_top): three float values
    
    recommended length of `features': n_images ~= 5000 (for saving time)
    Example:
    >>> backbone = ...          # e.g., a Resnet50 backbone that returns 2D features with the shape of (batch_size, feature_dim)
    >>> dataset_loader = ...    # e.g., a DataLoader of ImageNet
    >>>                         # **PLEASE NOTE**: JUST USE THE SIMPLEST transform (e.g., Resize + CenterCrop + ToTensor + Normalize)
    >>>
    >>> tot_num, features, labels = 0, [], []
    >>> with torch.no_grad():
    >>>     for x, y in dataset_loader:
    >>>         tot_num += x.shape[0]
    >>>         if tot_num > 5000:   # collect a subset of the original dataset for saving time
    >>>             break
    >>>         h = backbone(x.cuda()).cpu()
    >>>         y = y.view(-1).int()
    >>>         features.append(h), labels.append(y)
    >>> features, labels = torch.cat(features, dim=0), torch.cat(labels, dim=0)
    >>> assert features.ndim == 2 and labels.ndim == 1
    >>>
    >>> hy_mean, hy_max, hy_top = calc_MI_hy(features, labels)
    >>> print(hy_mean, hy_max, hy_top)
    """
    hy_values = __calc_MI_features_labels(features, labels, n_neighbors)
    hy_random = __get_random_MI_features_labels_mean(features, labels, n_neighbors)
    
    normalized_hy = [hy / hy_random for hy in hy_values]
    
    topk = max(2, round(len(hy_values) * 0.1))
    return (
        np.mean(normalized_hy).item(),
        np.max(normalized_hy).item(),
        np.mean(sorted(normalized_hy, reverse=True)[:topk]).item()
    )


def __calc_MI_features_labels(features: torch.Tensor, labels: torch.Tensor, n_neighbors: int):
    return mutual_info_classif(features, labels, n_neighbors=n_neighbors)


def __get_random_MI_features_labels_mean(features: torch.Tensor, labels: torch.Tensor, n_neighbors: int):
    shuffled_labels = labels
    shuffled_labels = shuffled_labels[torch.randperm(labels.shape[0])]
    shuffled_labels = shuffled_labels[torch.randperm(labels.shape[0])]
    baseline_values = mutual_info_classif(features, shuffled_labels, n_neighbors=n_neighbors)
    top_10_percent = max(2, round(len(baseline_values) * 0.1))
    return np.mean(sorted(baseline_values, reverse=True)[:top_10_percent]).item()

