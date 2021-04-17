from pprint import pformat

import torch
from torch import nn as nn

from model.modified_resbackbone import modified_res50backbone
from model.resbackbone import ResBackbone, Bottleneck


def load_r50backbone(ckpt: str, norm_func=nn.BatchNorm2d, conv_func=nn.Conv2d):
    state = torch.load(ckpt, map_location='cpu')
    if 'state_dict' in state.keys():
        state = state['state_dict']
    if 'module.backbone.conv1.weight' in state.keys():
        state = {
            k.replace('module.backbone.', ''): v
            for k, v in state.items() if k.startswith('module.backbone.')
        }
    elif 'backbone.conv1.weight' in state.keys():
        state = {
            k.replace('backbone.', ''): v
            for k, v in state.items() if k.startswith('backbone.')
        }
    
    assert 'conv1.weight' in state.keys(), f'strange keys of the ckpt:\n{list(state.keys())}'
    
    conv1_shape = state['conv1.weight'].shape
    deep_stem = conv1_shape[0] == 32
    enable_attnpool = 'attnpool.k_proj.weight' in state
    
    if deep_stem:
        r50_bb, warning = modified_res50backbone(clip_pretrain_path=ckpt, enable_attnpool=enable_attnpool)
    else:
        r50_bb = ResBackbone(Bottleneck, [3, 4, 6, 3], norm_func=norm_func, conv_func=conv_func)
        msg = r50_bb.load_state_dict(state, strict=False)
        unexpected_missing = [k for k in msg.missing_keys if not k.startswith('fc.')]
        assert len(unexpected_missing) == 0, f'unexpected msg.missing_keys:\n{pformat(unexpected_missing)}'
        unexpected_extra = msg.unexpected_keys
        if len(msg.unexpected_keys):
            warning = f'[warning] msg.unexpected_keys:\n{pformat(unexpected_extra)}'
        else:
            warning = ''
    return r50_bb, warning
