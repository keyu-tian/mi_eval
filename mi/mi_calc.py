from collections import OrderedDict
from multiprocessing import cpu_count
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


def __calc_MI_h_x(arg):
    features, targets, n_neighbors = arg
    mi_values = mutual_info_regression(features, targets, n_neighbors=n_neighbors)
    top_10_percent = max(1, round(len(mi_values) * 0.1))
    return np.mean(sorted(mi_values, reverse=True)[:top_10_percent]).item()


def get_random_MI_features_inputs_mean(features: torch.Tensor, inputs: torch.Tensor, n_neighbors: int):
    return np.mean(mutual_info_regression(features, torch.randn(features.shape[0]), n_neighbors=n_neighbors))
    

def calc_MI_features_inputs(verbose: bool, features: torch.Tensor, inputs: torch.Tensor, n_neighbors: int):
    num_data, num_inp_dim = inputs.shape
    num_targets = 23
    regression_targets = inputs[:, torch.linspace(0, num_inp_dim - 1, num_targets, dtype=torch.long)].T.contiguous()
    # regression_targets.shape: (num_targets, num_data)
    args = [(features, targets, n_neighbors) for targets in regression_targets]
    assert len(args) == num_targets
    
    P = min(cpu_count(), 6)
    if P >= 3:
        with Pool(P) as pool:
            results = list(pool.imap(__calc_MI_h_x, args, chunksize=1))
    else:
        results = []
        bar = tqdm(args) if verbose else args
        for arg in bar:
            mi_h_x = __calc_MI_h_x(arg)
            results.append(mi_h_x)
            if verbose:
                bar.set_description_str('[calc MI(h, x)]')
                bar.set_postfix(OrderedDict({'cur_mi': f'{mi_h_x:.3g}'}))
    
    assert len(results) == num_targets
    return results


def calc_MI_features_labels(features: torch.Tensor, labels: torch.Tensor, n_neighbors: int):
    return mutual_info_classif(features, labels, n_neighbors=n_neighbors)


def get_random_MI_features_labels_mean(features: torch.Tensor, labels: torch.Tensor, n_neighbors: int):
    return np.mean(mutual_info_classif(features, torch.randint(0, labels.max().item()+1, labels.shape).type_as(labels), n_neighbors=n_neighbors))


def speed_test():
    import time
    stt = time.time()
    print(calc_MI_features_inputs(True, torch.rand((3840, 2048)), torch.rand((3840, 4107)), 15))
    print(f'time cost: {time.time()-stt:.2f}s')


if __name__ == '__main__':
    speed_test()
