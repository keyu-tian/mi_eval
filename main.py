from collections import OrderedDict, defaultdict

import argparse
import datetime
import time
from sklearn.feature_selection import mutual_info_classif
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import transforms
import os
from pandas import DataFrame

import knn_mi

from imagenet import SubImageNetDataset
from attribute import Age, Gender, Liveness
from resbackbone import load_r50backbone

import linklink as link
from tqdm import tqdm


def link_init():
    torch.cuda.set_device(int(os.environ['SLURM_LOCALID']))
    link.initialize()
    return link.get_rank(), link.get_world_size()


def calc_mi(features: torch.Tensor, labels: torch.Tensor, args):
    # return [knn_mi.mi(features.numpy(), labels.view(features.shape[0], -1).numpy(), k=args.n_neighbors)]
    return mutual_info_classif(features, labels, n_neighbors=args.n_neighbors)


def main():
    # stt = time.time()
    # from easydict import EasyDict
    # args = EasyDict({'n_neighbors': 50})
    # print(calc_mi(torch.rand((500, 128)), torch.rand((500, 2330)), args))
    # print(round(time.time()-stt, 2))
    # exit(0)
    
    parser = argparse.ArgumentParser(description='MI test')
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--num_classes', default=50, type=int)
    parser.add_argument('--n_neighbors', default=50, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()
    
    rank, world_size = link_init()
    
    if rank == 0:
        print(f'[rk{rank}]: args=\n{vars(args)}\n')
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
    ])

    # ~ num_classes * 1280 images
    if args.dataset == 'imagenet':
        test_data = SubImageNetDataset(num_classes=args.num_classes, root='/mnt/lustre/share/images', train=False, transform=test_transform, download=False)
    elif args.dataset == 'liveness':
        test_data = Liveness()
        args.batch_size = 64
    elif args.dataset == 'gender':
        test_data = Gender()
        args.batch_size = 64
    elif args.dataset == 'age':
        test_data = Age()
        args.batch_size = 64
    
    test_len = len(test_data)
    test_ld = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    prefix = os.path.join(os.path.expanduser('~'), 'htl_ckpt')
    ckpts = [
        os.path.join(prefix, 'DY_MTL_LV1_10_R50_convertBB.pth.tar'),
        os.path.join(prefix, 'DY_MTL_LV1_30_R50_convertBB.pth.tar'),
        os.path.join(prefix, 'xueshuClip.pth.tar'),
    ]
    ckpt_names = [
        os.path.split(ckpt)[-1].replace('.tar', '').replace('.pth', '')
        for ckpt in ckpts
    ]
    ckpt_idx = rank % len(ckpts)
    ckpt = ckpts[ckpt_idx]
    ckpt_name = ckpt_names[ckpt_idx]
    inputs, features, labels = [], [], []
    r50_bb, warning = load_r50backbone(ckpt)
    if rank < len(ckpts):
        print(f'[rk{rank}]: {warning or "nothing"}')
    r50_bb = r50_bb.cuda()
    r50_bb.eval()
    
    link.barrier()
    if rank == 0:
        bar = tqdm(test_ld)
    else:
        bar = test_ld
    with torch.no_grad():
        for x, y in bar:
            bs = x.shape[0]
            h = r50_bb(x.cuda()).cpu()
            y = y.view(bs, 1).int()
            inputs.append(F.avg_pool2d(x.mean(dim=1), kernel_size=4).view(bs, -1))
            features.append(h)
            labels.append(y)
            if rank == 0:
                bar.set_description_str('[extracting features]')
                bar.set_postfix(OrderedDict({'imgs': test_len, 'cur_bs': bs, 'ckpt': ckpt_name}))
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    labels = labels.reshape(-1)
    inputs = torch.cat(inputs, dim=0)
    assert features.shape[0] == labels.shape[0] == inputs.shape[0] == test_len
    
    if rank == 0:
        print(f'[rk{rank}]: features={features.dtype} {tuple(features.shape)},  labels.shape={labels.dtype} {tuple(labels.shape)},  inputs.shape={inputs.dtype} {tuple(inputs.shape)}')

    stt = time.time()
    hy_mi_values = calc_mi(features, labels, args)
    hy_cost = time.time()-stt

    stt = time.time()
    # todo: 如果放开I(h,x)计算，那记得去掉下面的注释
    # hx_mi_values = calc_mi(features, inputs, args)
    hx_mi_values = 0
    hx_cost = time.time()-stt
    
    if rank == 0:
        print(f'[rk{rank}]: I(h, y) time cost = {hy_cost:.2f}s ({hy_cost / 60:.2f}min)')
        print(f'[rk{rank}]: I(h, x) time cost = {hx_cost:.2f}s ({hx_cost / 60:.2f}min)')
    hy_mi_mean = sum(hy_mi_values) / len(hy_mi_values)
    hx_mi_mean = sum(hx_mi_values) / len(hx_mi_values)
    
    top_10_percent = max(1, round(len(hy_mi_values) * 0.1))
    hy_mi_top_10_percent = sorted(hy_mi_values, reverse=True)[:top_10_percent]
    hy_mi_top_10_percent = sum(hy_mi_top_10_percent) / len(hy_mi_top_10_percent)
    hx_mi_top_10_percent = sorted(hx_mi_values, reverse=True)[:top_10_percent]
    hx_mi_top_10_percent = sum(hx_mi_top_10_percent) / len(hx_mi_top_10_percent)
    
    hy_mi_max = max(hy_mi_values)
    hx_mi_max = max(hx_mi_values)
    
    for i in range(len(ckpts)):
        if ckpt_idx == i:
            time.sleep(0.1 * rank)
            print(
                f'[rk{rank}]: ckpt={ckpt}\n'
                f'I(h, y):    mean={hy_mi_mean:.4f},  max={hy_mi_max:.4f},  top={hy_mi_top_10_percent:.4f}\n'
                f'I(h, x):    mean={hx_mi_mean:.4f},  max={hx_mi_max:.4f},  top={hx_mi_top_10_percent:.4f}'
            )
        link.barrier()

    hy_mi_means, hy_mi_maxes, hy_mi_topks = torch.zeros(world_size), torch.zeros(world_size), torch.zeros(world_size)
    hy_mi_means[rank], hy_mi_maxes[rank], hy_mi_topks[rank] = hy_mi_mean, hy_mi_max, hy_mi_top_10_percent
    link.allreduce(hy_mi_means), link.allreduce(hy_mi_maxes), link.allreduce(hy_mi_topks)

    hx_mi_means, hx_mi_maxes, hx_mi_topks = torch.zeros(world_size), torch.zeros(world_size), torch.zeros(world_size)
    hx_mi_means[rank], hx_mi_maxes[rank], hx_mi_topks[rank] = hx_mi_mean, hx_mi_max, hx_mi_top_10_percent
    link.allreduce(hx_mi_means), link.allreduce(hx_mi_maxes), link.allreduce(hx_mi_topks)

    if rank == 0:
        all_hy_mi_means, all_hy_mi_maxes, all_hy_mi_topks = defaultdict(list), defaultdict(list), defaultdict(list)
        all_hx_mi_means, all_hx_mi_maxes, all_hx_mi_topks = defaultdict(list), defaultdict(list), defaultdict(list)
        for rk in range(world_size):
            idx = rk % len(ckpts)
            all_hy_mi_means[ckpts[idx]].append(hy_mi_means[rk].item())
            all_hy_mi_maxes[ckpts[idx]].append(hy_mi_maxes[rk].item())
            all_hy_mi_topks[ckpts[idx]].append(hy_mi_topks[rk].item())
            all_hx_mi_means[ckpts[idx]].append(hx_mi_means[rk].item())
            all_hx_mi_maxes[ckpts[idx]].append(hx_mi_maxes[rk].item())
            all_hx_mi_topks[ckpts[idx]].append(hx_mi_topks[rk].item())
        for d in [all_hy_mi_means, all_hy_mi_maxes, all_hy_mi_topks, all_hx_mi_means, all_hx_mi_maxes, all_hx_mi_topks]:
            for k, v in d.items(): v.insert(0, sum(v) / len(v))
        df = DataFrame({
            'ckpt': ckpt_names,
            'hy_mean': [all_hy_mi_means[ckpt][0] for ckpt in ckpts],
            'hy_max': [all_hy_mi_maxes[ckpt][0] for ckpt in ckpts],
            'hy_top': [all_hy_mi_topks[ckpt][0] for ckpt in ckpts],
            # todo: 记得去注释
            # 'hx_mean': [all_hx_mi_means[ckpt][0] for ckpt in ckpts],
            # 'hx_max': [all_hx_mi_maxes[ckpt][0] for ckpt in ckpts],
            # 'hx_top': [all_hx_mi_topks[ckpt][0] for ckpt in ckpts],
        })
        print(df)
        df.to_json(f'results_{args.dataset}_cls{args.num_classes}_neib{args.n_neighbors}_{datetime.datetime.now().strftime("%m-%d %H:%M:%S")}.json')
    
    link.barrier()
    link.finalize()


if __name__ == '__main__':
    main()
