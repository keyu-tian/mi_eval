import datetime
import os
import time
from collections import OrderedDict
from pprint import pformat

import numpy as np
import pytz
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from tap import Tap
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset.industry_dataset import Age, Gender, Liveness
from dataset.subimagenet import SubImageNetDataset
from mi.mi_calc import calc_MI_features_labels, calc_MI_features_inputs, get_random_MI_features_labels_mean, get_random_MI_features_inputs_mean
from mi.mi_report import report
from model import load_r50backbone

try:
    import linklink as link
except:
    import spring.linklink as link


def time_str():
    return datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime('[%m-%d %H:%M:%S]')


def link_init():
    dev_idx = int(os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(dev_idx)
    link.initialize()
    return link.get_rank(), link.get_world_size()


class Args(Tap):
    cfg_path: str
    dataset: str
    train: bool = False
    batch_size: int = 256
    n_neighbors: int = 12
    calc_hx: bool = False
    subimagenet_num_classes: int = 170


def main():
    args = Args().parse_args()
    
    # ========> 1. load ckpt paths <========
    with open(args.cfg_path.strip('\"\''), 'r') as fin:
        ckpts = yaml.safe_load(fin)['checkpoints']

    rank, world_size = link_init()
    if rank == 0:
        print(f'{time_str()}[rk{rank}]: args=\n{pformat(vars(args.as_dict()))}\n')

    # ========> 2. load checkpoints <========
    ckpt_idx = rank % len(ckpts)
    ckpt = ckpts[ckpt_idx]
    ckpt_name = os.path.split(ckpt)[-1].replace('.tar', '').replace('.pth', '')
    r50_bb, warning = load_r50backbone(ckpt)
    for rk in range(len(ckpts)):
        link.barrier()
        if rk == rank:
            print(f'{time_str()}[rk{rank}, {ckpt_name}]: {warning or "nothing"}')
    r50_bb = r50_bb.cuda()
    r50_bb.eval()

    # ========> 3. load data <========
    img_hw, features, inputs, labels = get_data(args, r50_bb, verbose=rank == 0)
    if rank == 0:
        print(f'\n{time_str()}[rk{rank}]: img={img_hw[0]}x{img_hw[1]}, features={features.dtype} {tuple(features.shape)},  labels.shape={labels.dtype} {tuple(labels.shape)},  inputs.shape={inputs.dtype} {tuple(inputs.shape)}')
    
    # ========> 4. calc MI(h, y) <========
    stt = time.time()
    hy_values = calc_MI_features_labels(features, labels, args.n_neighbors)
    hy_cost = time.time() - stt
    if rank == 0:
        print(f'{time_str()}[rk{rank}]: I(h, y) time cost = {hy_cost:.2f}s ({hy_cost / 60:.2f}min)')
    hy_random = get_random_MI_features_labels_mean(features, labels, args.n_neighbors)
    if rank == 0:
        print(
            f'{time_str()}[rk{rank}]: == RANDOM ==\n'
            f'I(h, y):    mean={hy_random:.3g}'
        )
    hy_values = [abs(v / hy_random) ** 0.5 for v in hy_values]      # normalize MI values
    hy_mean, hy_max = np.mean(hy_values).item(), max(hy_values)
    hy_top = np.mean(sorted(hy_values, reverse=True)[:max(2, round(len(hy_values) * 0.1))]).item()
    for i in range(len(ckpts)):
        link.barrier()
        if ckpt_idx == i:
            time.sleep(0.1 * rank)
            print(
                f'{time_str()}[rk{rank}]: ckpt={ckpt}\n'
                f'I(h, y):    mean={hy_mean:.3g},  max={hy_max:.3g},  top={hy_top:.3g}'
            )

    # ========> 5. calc MI(h, x) <========
    if args.calc_hx:
        stt = time.time()
        hx_values = calc_MI_features_inputs(rank == 0, features, inputs, args.n_neighbors)
        hx_cost = time.time() - stt
        if rank == 0:
            print(f'{time_str()}[rk{rank}]: I(h, x) time cost = {hx_cost:.2f}s ({hx_cost / 60:.2f}min)')
        hx_random = get_random_MI_features_inputs_mean(features, labels, args.n_neighbors)
        if rank == 0:
            print(
                f'{time_str()}[rk{rank}]: == RANDOM ==\n'
                f'I(h, x):    mean={hx_random:.3g}'
            )
        hx_values = [abs(v / hx_random) ** 0.5 for v in hx_values]  # normalize MI values
        hx_mean, hx_max = np.mean(hx_values).item(), max(hx_values)
        hx_top = np.mean(sorted(hx_values, reverse=True)[:max(2, round(len(hx_values) * 0.15))]).item()
        for i in range(len(ckpts)):
            link.barrier()
            if ckpt_idx == i:
                time.sleep(0.1 * rank)
                print(
                    f'{time_str()}[rk{rank}]: ckpt={ckpt}\n'
                    f'I(h, x):    mean={hx_mean:.3g},  max={hx_max:.3g},  top={hx_top:.3g}'
                )
    else:
        hx_mean = hx_max = hx_top = -1
    
    # ========> 6. report the results <========
    time.sleep(1)
    report(args, ckpts, hy_mean, hy_max, hy_top, hx_mean, hx_max, hx_top)
    
    link.barrier()
    link.finalize()


def get_data(args: Args, r50_bb, verbose=False):
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
    ])
    data_clz, data_kw = {
        'subimagenet': (SubImageNetDataset, dict(
            num_classes=args.subimagenet_num_classes,
            train=args.train, transform=test_transform
        )),
        'liveness': (Liveness, dict(train=args.train)),
        'gender': (Gender, dict(train=args.train)),
        'age': (Age, dict(train=args.train)),
    }[args.dataset]
    test_data = data_clz(**data_kw)
    test_ld = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
    tot_bs, inputs, features, labels = 0, [], [], []
    with torch.no_grad():
        bar = tqdm(test_ld) if verbose else test_ld
        for x, y in bar:
            img_hw = tuple(x.shape[-2:])
            bs = x.shape[0]
            tot_bs += bs
            if tot_bs > 8000:  # calc MI on a subset for saving time
                break
            h = r50_bb(x.cuda()).cpu()
            y = y.view(bs, 1).int()
            inputs.append(F.avg_pool2d(x, kernel_size=8).view(bs, -1))
            features.append(h)
            labels.append(y)
            if verbose:
                bar.set_description_str('[extracting features]')
                bar.set_postfix(OrderedDict({'tot_bs': tot_bs, 'cur_bs': bs}))
    if verbose:
        bar.clear()
        bar.close()

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0).reshape(-1)
    inputs = torch.cat(inputs, dim=0)
    assert features.shape[0] == labels.shape[0] == inputs.shape[0]
    return img_hw, features, inputs, labels


if __name__ == '__main__':
    main()
