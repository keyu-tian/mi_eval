from collections import OrderedDict, defaultdict

import argparse
import datetime
from sklearn.feature_selection import mutual_info_classif
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from pandas import DataFrame
# from calc_mi import mi

from imagenet import SubImageNetDataset
from resbackbone import load_r50backbone

import linklink as link
from tqdm import tqdm


def link_init():
    torch.cuda.set_device(int(os.environ['SLURM_LOCALID']))
    link.initialize()
    return link.get_rank(), link.get_world_size()


def main():
    parser = argparse.ArgumentParser(description='MI test')
    parser.add_argument('--num_classes', default=30, type=int)
    parser.add_argument('--n_neighbors', default=3, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()
    
    rank, world_size = link_init()
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
    ])

    # ~ num_classes * 1280 images
    test_data = SubImageNetDataset(num_classes=args.num_classes, root='/mnt/lustre/share/images', train=False, transform=test_transform, download=False)
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
            h = torch.from_numpy(r50_bb(x.cuda()).cpu().numpy())
            y = torch.from_numpy(y.view(bs, 1).numpy().astype(int))
            # inputs.append(x.numpy())
            features.append(h)
            labels.append(y)
            if rank == 0:
                bar.set_description_str('[extracting features]')
                bar.set_postfix(OrderedDict({'imgs': test_len, 'cur_bs': bs, 'ckpt': ckpt_name}))
    
    # inputs = torch.cat(inputs, dim=0)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    labels = labels.reshape(-1)
    assert features.shape[0] == labels.shape[0] == test_len
    
    if rank == 0:
        print(f'[rk{rank}]: features={features.dtype} {tuple(features.shape)},  labels.shape={labels.dtype} {tuple(labels.shape)}')

    mi_values = mutual_info_classif(features, labels, n_neighbors=args.n_neighbors)
    mi_mean = sum(mi_values) / len(mi_values)
    
    top_10_percent = max(1, round(len(mi_values) * 0.1))
    mi_top_10_percent = sorted(mi_values, reverse=True)[:top_10_percent]
    mi_top_10_percent = sum(mi_top_10_percent) / len(mi_top_10_percent)
    
    mi_max = max(mi_values)
    
    for i in range(len(ckpts)):
        if ckpt_idx == i:
            print(
                f'[rk{rank}]: ckpt={ckpt}, mi info:\n    mean={mi_mean:.4f},  max={mi_max:.4f},  top={mi_top_10_percent:.4f}'
            )
        link.barrier()
    
    mi_means = torch.zeros(world_size)
    mi_maxes = torch.zeros(world_size)
    mi_topks = torch.zeros(world_size)
    mi_means[rank] = mi_mean
    mi_maxes[rank] = mi_max
    mi_topks[rank] = mi_top_10_percent
    link.allreduce(mi_means), link.allreduce(mi_maxes), link.allreduce(mi_topks)

    if rank == 0:
        all_mi_means, all_mi_maxes, all_mi_topks = defaultdict(list), defaultdict(list), defaultdict(list)
        for rk in range(world_size):
            idx = rk % len(ckpts)
            all_mi_means[ckpts[idx]].append(mi_means[rk].item())
            all_mi_maxes[ckpts[idx]].append(mi_maxes[rk].item())
            all_mi_topks[ckpts[idx]].append(mi_topks[rk].item())
        for k, v in all_mi_means.items(): v.insert(0, sum(v) / len(v))
        for k, v in all_mi_maxes.items(): v.insert(0, sum(v) / len(v))
        for k, v in all_mi_topks.items(): v.insert(0, sum(v) / len(v))
        df = DataFrame({
            'ckpt': ckpt_names,
            'mean': [all_mi_means[ckpt][0] for ckpt in ckpts],
            'max': [all_mi_maxes[ckpt][0] for ckpt in ckpts],
            'top': [all_mi_topks[ckpt][0] for ckpt in ckpts],
        })
        print(df)
        df.to_json(f'results_imn{args.num_classes}_neib{args.n_neighbors}_{datetime.datetime.now().strftime("[%m-%d %H:%M:%S]")}.json')
    
    link.barrier()
    link.finalize()


if __name__ == '__main__':
    main()
