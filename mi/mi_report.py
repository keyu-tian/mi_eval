import datetime
import os
from collections import defaultdict

import numpy as np
import torch
import pandas as pd

try:
    import linklink as link
except:
    import spring.linklink as link


def report(cfg, hy_mean, hy_max, hy_top, hx_mean, hx_max, hx_top):
    ckpts = cfg.checkpoints
    
    rank, world_size = link.get_rank(), link.get_world_size()
    hy_means, hy_maxes, hy_topks = torch.zeros(world_size), torch.zeros(world_size), torch.zeros(world_size)
    hy_means[rank], hy_maxes[rank], hy_topks[rank] = hy_mean, hy_max, hy_top
    link.allreduce(hy_means), link.allreduce(hy_maxes), link.allreduce(hy_topks)
    
    hx_means, hx_maxes, hx_topks = torch.zeros(world_size), torch.zeros(world_size), torch.zeros(world_size)
    hx_means[rank], hx_maxes[rank], hx_topks[rank] = hx_mean, hx_max, hx_top
    link.allreduce(hx_means), link.allreduce(hx_maxes), link.allreduce(hx_topks)
    
    if rank == 0:
        all_hy_means, all_hy_maxes, all_hy_topks = defaultdict(list), defaultdict(list), defaultdict(list)
        all_hx_means, all_hx_maxes, all_hx_topks = defaultdict(list), defaultdict(list), defaultdict(list)
        for rk in range(world_size):
            idx = rk % len(ckpts)
            ckpt = ckpts[idx]
            all_hy_means[ckpt].append(hy_means[rk].item()), all_hy_maxes[ckpt].append(hy_maxes[rk].item()), all_hy_topks[ckpt].append(hy_topks[rk].item())
            all_hx_means[ckpt].append(hx_means[rk].item()), all_hx_maxes[ckpt].append(hx_maxes[rk].item()), all_hx_topks[ckpt].append(hx_topks[rk].item())
        
        pd.set_option('display.max_rows', None), pd.set_option('display.max_columns', None)
        pd.set_option('max_colwidth', 100), pd.set_option('display.width', 200)
        print(pd.DataFrame({
            'ckpt': [
                os.path.split(ckpt)[-1].replace('.tar', '').replace('.pth', '')
                for ckpt in ckpts
            ],
            'hy_mea': [f'{np.mean(all_hy_means[ckpt]).item():.3g}' for ckpt in ckpts],
            'hy_max': [f'{np.mean(all_hy_maxes[ckpt]).item():.3g}' for ckpt in ckpts],
            'hy_top': [f'{np.mean(all_hy_topks[ckpt]).item():.3g}' for ckpt in ckpts],
            'hx_mea': [f'{np.mean(all_hx_means[ckpt]).item():.3g}' for ckpt in ckpts],
            'hx_max': [f'{np.mean(all_hx_maxes[ckpt]).item():.3g}' for ckpt in ckpts],
            'hx_top': [f'{np.mean(all_hx_topks[ckpt]).item():.3g}' for ckpt in ckpts],
        }))

        df = pd.DataFrame({
            'ckpt': [
                os.path.split(ckpt)[-1].replace('.tar', '').replace('.pth', '')
                for ckpt in ckpts
            ],
            'hy_mea': [np.mean(all_hy_means[ckpt]).item() for ckpt in ckpts],
            'hy_max': [np.mean(all_hy_maxes[ckpt]).item() for ckpt in ckpts],
            'hy_top': [np.mean(all_hy_topks[ckpt]).item() for ckpt in ckpts],
            'hx_mea': [np.mean(all_hx_means[ckpt]).item() for ckpt in ckpts],
            'hx_max': [np.mean(all_hx_maxes[ckpt]).item() for ckpt in ckpts],
            'hx_top': [np.mean(all_hx_topks[ckpt]).item() for ckpt in ckpts],
        })
        dirname = os.path.split(os.getcwd())[-1]
        f_name = f'results_{dirname}_neib{cfg.n_neighbors}_{"trainset" if cfg.train_set else "testset"}_{datetime.datetime.now().strftime("%m-%d_%H-%M-%S")}.json'
        df.to_json(f_name)
        print(f'==> results saved at {os.path.abspath(f_name)}')
