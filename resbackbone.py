from pprint import pformat

import torch.nn as nn
import math
import torch

# will be overwritten during runtime
from modified_resbackbone import modified_res50backbone

BN = torch.nn.BatchNorm2d
Conv2d = torch.nn.Conv2d


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResBackbone(nn.Module):

    def __init__(self, block, layers, norm_func=nn.BatchNorm2d, conv_func=nn.Conv2d):
        super(ResBackbone, self).__init__()

        global BN
        global Conv2d

        BN = norm_func
        Conv2d = conv_func
        
        self.curr_inplanes = 64

        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # maskrcnn
        # for p in self.parameters():
        #     p.requires_grad = False
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.curr_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.curr_inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                BN(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.curr_inplanes, planes, stride, downsample))
        self.curr_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.curr_inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.avgpool(layer4)
        feature = torch.flatten(x, 1)    # pre_fc: (B, fea_dim)
        return feature


def load_r50backbone(ckpt: str, norm_func=nn.BatchNorm2d, conv_func=nn.Conv2d):
    d = torch.load(ckpt, map_location='cpu')
    if set(d.keys()) == {'state_dict'}:
        d = d['state_dict']
    conv1_shape = d['conv1.weight'].shape
    deep_stem = conv1_shape[0] == 32
    
    if deep_stem:
        r50_bb, warning = modified_res50backbone(clip_pretrain_path=ckpt)
    else:
        r50_bb = ResBackbone(Bottleneck, [3, 4, 6, 3], norm_func=norm_func, conv_func=conv_func)
        msg = r50_bb.load_state_dict(d, strict=False)
        unexpected_missing = [k for k in msg.missing_keys if not k.startswith('fc.')]
        assert len(unexpected_missing) == 0, f'unexpected msg.missing_keys:\n{pformat(unexpected_missing)}'
        unexpected_extra = msg.unexpected_keys
        if len(msg.unexpected_keys):
            warning = f'[warning] msg.unexpected_keys:\n{pformat(unexpected_extra)}'
        else:
            warning = ''
    return r50_bb, warning

