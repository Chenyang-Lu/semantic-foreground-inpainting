import torch
import math
import numbers
import numpy as np
from torch import nn
from torch.nn import functional as F
import ours_extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=False)
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=3, sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet50',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(ours_extractors, backend)(pretrained)
        self.convfore = getattr(ours_extractors, 'resnet18half')(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

    def forward(self, x, training, device, gt_fore, use_gt_fore):
        fore_sigmoid = self.convfore(x)
        if use_gt_fore:
            f = self.feats(x, gt_fore, training, device)
        else:
            f = self.feats(x, fore_sigmoid, training, device)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        return self.final(p), fore_sigmoid 

        # x_final = self.final(p)
        # x_patch = ours_extractors.MaxpoolingAsInpainting(x_final, fore_sigmoid)
        # x_final = x_patch * (F.interpolate(fore_sigmoid, size=x.size()[2:4]) > 0.5).float() +\
        #     x_final * (F.interpolate(fore_sigmoid, size=x.size()[2:4]) < 0.5).float()
        # return x_final, fore_sigmoid 


class PSPNetShareEarlyLayer(nn.Module):
    def __init__(self, n_classes=3, sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet50shareearlylayer',
                 pretrained=True):
        super().__init__()
        self.feats = getattr(ours_extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

    def forward(self, x, training, device, gt_fore, use_gt_fore):

        f, fore_sigmoid = self.feats(x, training, device)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        return self.final(p), fore_sigmoid 
