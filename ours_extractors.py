from collections import OrderedDict
import math
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision.models.densenet import densenet121, densenet161
from torchvision.models.squeezenet import squeezenet1_1
from util import random_rectangle_mask_gene


def MaxpoolingAsInpainting(x, x_fore_augment):
    x_fore_pool = F.interpolate(x_fore_augment, size=x.size()[2:4]).detach().clone()
    fore_msk = (x_fore_pool > 0.5).float().detach().clone()
    fore_msk = F.max_pool2d((fore_msk).detach().clone(), kernel_size=3, stride=1, padding=1)
    bkgd_msk_prev = (1.-fore_msk).detach().clone()
    bkgd_msk =  bkgd_msk_prev.detach().clone()
    x_new = x.detach().clone() * bkgd_msk.detach().clone()
    x_patch = x.detach().clone() * 0.

    while torch.mean(1.-bkgd_msk).item() > 0.0001 and torch.min(torch.mean(bkgd_msk, dim=(1, 2, 3))) > 0.:
        x_new = F.max_pool2d((x_new).detach().clone(), kernel_size=3, stride=1, padding=1)
        bkgd_msk = F.max_pool2d((bkgd_msk_prev).detach().clone(),kernel_size=3, stride=1, padding=1)
        x_patch = (x_patch + (bkgd_msk-bkgd_msk_prev)*x_new).detach().clone()
        bkgd_msk_prev = bkgd_msk.detach().clone()
    
    return x_patch


class Upsample(nn.Module):
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


def load_weights_sequential(target, source_state):
    model_to_load= {k: v for k, v in source_state.items() if k in target.state_dict().keys()}
    target.load_state_dict(model_to_load, strict=False)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNetHalf(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), input_channel=3):
        self.inplanes = 64
        super(ResNetHalf, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], dilation=2)
        self.layer2 = self._make_layer(block, 128, layers[1], dilation=2)
        self.convforeout = nn.Sequential(
            nn.Conv2d(128, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Upsample(512, 256),
            Upsample(256, 128),
            nn.Conv2d(128, 1, 1, bias=False)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.convforeout(x)
        x = torch.sigmoid(x)

        return x




class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), input_channel=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.maxpool_fore = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, img_in, x_fore_pred, training, device):

        x = img_in


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        temp_rect_mask = random_rectangle_mask_gene(img_size=x_fore_pred.size()[2:4], batch_size=x_fore_pred.size(0),
                                                    num_rect=3, device='cuda:0', max_size_factor=0.4)
        if training:
            x_fore_augment = 1 - (1 - (x_fore_pred > 0.5).float().detach().clone()) * temp_rect_mask
        else:
            x_fore_augment = x_fore_pred
        x_patch = MaxpoolingAsInpainting(x, x_fore_augment)
        x = x_patch * (F.interpolate(x_fore_augment, size=x.size()[2:4]) > 0.5).float() +\
            x * (F.interpolate(x_fore_augment, size=x.size()[2:4]) < 0.5).float()
        # x = x * ((F.interpolate(x_fore_augment, size=x.size()[2:4]) < 0.5).float())


        x_3 = self.layer3(x)
        x = self.layer4(x_3)

        return x


class ResNetShareEarlyLayer(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), input_channel=3):
        self.inplanes = 64
        super(ResNetShareEarlyLayer, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes_layer2 = self.inplanes
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.convforeout = nn.Sequential(
            nn.Conv2d(self.inplanes_layer2, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Upsample(512, 256),
            Upsample(256, 128),
            nn.Conv2d(128, 1, 1, bias=False)
        )

        self.maxpool_fore = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, img_in, training, device):

        x = img_in


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_fore_pred = self.convforeout(x)
        x_fore_pred = torch.sigmoid(x_fore_pred)
        x_fore_pred = F.interpolate(x_fore_pred, scale_factor=2)


        temp_rect_mask = random_rectangle_mask_gene(img_size=x_fore_pred.size()[2:4], batch_size=x_fore_pred.size(0),
                                                    num_rect=3, device='cuda:0', max_size_factor=0.4)
        if training:
            x_fore_augment = 1 - (1 - (x_fore_pred > 0.5).float().detach().clone()) * temp_rect_mask
        else:
            x_fore_augment = x_fore_pred
        x_patch = MaxpoolingAsInpainting(x, x_fore_augment)
        x = x_patch * (F.interpolate(x_fore_augment, size=x.size()[2:4]) > 0.5).float() +\
            x * (F.interpolate(x_fore_augment, size=x.size()[2:4]) < 0.5).float()
        # x = x * ((F.interpolate(x_fore_augment, size=x.size()[2:4]) < 0.5).float())


        x_3 = self.layer3(x)
        x = self.layer4(x_3)

        return x, x_fore_pred



def resnet18(pretrained=True):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18half(pretrained=True):
    model = ResNetHalf(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet18shareearlylayer(pretrained=True):
    model = ResNetShareEarlyLayer(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet50(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet50half(pretrained=True):
    model = ResNetHalf(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet50shareearlylayer(pretrained=True):
    model = ResNetShareEarlyLayer(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(model_urls['resnet50']))
    return model
