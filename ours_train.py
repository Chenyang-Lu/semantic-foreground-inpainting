import random
import time
import os
import copy
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as tdist
from torch.optim import lr_scheduler
from torch import autograd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from util import validation_metrics
from cs_data_loader import *
from ours_model import *

seed = 1
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


DATASET = 'Cityscapes'
num_epochs = 50
batch_size = 8
img_size = (256, 512)
n_classes = 3
weights_in_loss = [1., 5.81867525, 0.78376658]

restore = True
dataset_dir = 'dataset/' + DATASET
methodology = 'ours_hardthreshold_psp_pooling_resnet18_aug'
checkpoint_path = 'checkpoints/' + DATASET + '/' + methodology + '.pth.tar'
writer = SummaryWriter(comment='_'+ DATASET + '_' + methodology)


# Define dataloaders
if DATASET == 'Cityscapes':
    val_gt_list = sorted(glob.glob(os.path.join(dataset_dir, 'gt_manual', 'val') + '/*/*gt_manual.png'))[:100]
train_set = CSDataset(dataset_dir+'/train.csv', transform=transforms.Compose([Rescale(img_size), CSToTensor()]))
val_set = CSDataset(dataset_dir+'/val.csv', transform=transforms.Compose([Rescale(img_size), CSToTensor()]))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=6, drop_last=False)
dataloaders = {'train': train_loader, 'val': val_loader}


# G = PSPNetShareEarlyLayer(backend='resnet18shareearlylayer', psp_size=512, pretrained=False, n_classes=n_classes)
G = PSPNet(backend='resnet18', psp_size=512, pretrained=False, n_classes=n_classes)
G.to(device)

optimizerG = optim.Adam(G.parameters(), lr=0.0001, betas=(0.6, 0.9), weight_decay=0.0001)
schedulerG = lr_scheduler.StepLR(optimizerG, step_size=20, gamma=0.1)

if restore:
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path)
        epoch = state['epoch']
        G.load_state_dict(state['state_dict_G'])
        optimizerG.load_state_dict(state['optimizer_G'])
        schedulerG.load_state_dict(state['scheduler_G'])
    else:
        epoch = 0
else:
    epoch = 0

while epoch < num_epochs:
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))

    for phase in ['train', 'val']:
        if phase == 'train':
            schedulerG.step()
            G.train()
        else:
            G.eval()
            pred_segs = np.zeros((len(val_set), img_size[0], img_size[1])).astype(np.uint8)
            val_gt_segs = np.zeros((len(val_set), img_size[0], img_size[1])).astype(np.uint8)
            foregd_segs = np.zeros((len(val_set), img_size[0], img_size[1])).astype(np.float16)
        temp_val_loss = 0.0

        for i, temp_batch in enumerate(dataloaders[phase]):
            temp_rgb = temp_batch['rgb'].float().to(device)
            temp_foregd = temp_batch['foregd'].long().to(device)
            temp_partial_bkgd = temp_batch['partial_bkgd'].long().squeeze(1).to(device)

            with torch.set_grad_enabled(phase == 'train'):
                # pre-processing the input and target on the fly

                foregd_idx = (temp_foregd.float() > 0.5).float()

                # training
                pred_seg, fore_middle_msk = G(temp_rgb, phase=='train', device, foregd_idx, use_gt_fore=False)

                if phase == 'train':

                    loss_entropy_stage1 = F.cross_entropy(pred_seg, temp_partial_bkgd,
                                        weight=torch.Tensor(weights_in_loss).to(device), ignore_index=255)
                    loss_entropy_foregd = F.binary_cross_entropy(fore_middle_msk, foregd_idx)
                    loss_G_all = 1. *loss_entropy_stage1 + 1. *loss_entropy_foregd

                    optimizerG.zero_grad()
                    loss_G_all.backward()
                    optimizerG.step()

                    writer.add_scalar(phase + '_loss_main', loss_entropy_stage1, epoch * len(train_set)/batch_size + i)
                    writer.add_scalar(phase + '_loss_foregd', loss_entropy_foregd, epoch * len(train_set)/batch_size + i)

                else:
                    pred_segs[i, :, :] = np.argmax(pred_seg.to('cpu').numpy().squeeze(), axis=0).astype(np.uint8)
                    val_gt_segs[i, :, :] = transform.resize(io.imread(val_gt_list[i]), img_size, order=0, 
                                      mode='reflect', preserve_range=True, anti_aliasing=False).astype(np.uint8)
                    foregd_segs[i, :, :] = foregd_idx.to('cpu').numpy().squeeze().astype(np.float16)
                    temp_val_loss += loss_entropy_stage1.item()


        # statistics
        if phase == 'train':
            pass
        else:
            temp_val_loss = temp_val_loss / len(val_set)
            writer.add_scalar(phase + '_loss_main', temp_val_loss, epoch * len(train_set)/(batch_size+1))
            mean_accu, mean_iu = validation_metrics(pred_segs, val_gt_segs, False)
            mean_accu_foregd, mean_iu_foregd = validation_metrics(pred_segs, val_gt_segs, True, foregd_segs)

            writer.add_scalar(phase + '_mean_accu', mean_accu, (epoch + 1) * len(train_set)/batch_size)
            writer.add_scalar(phase + '_mean_iu', mean_iu, (epoch + 1) * len(train_set)/batch_size)
            writer.add_scalar(phase + '_mean_accu_foregd', mean_accu_foregd, (epoch + 1) * len(train_set)/batch_size)
            writer.add_scalar(phase + '_mean_iu_foregd', mean_iu_foregd, (epoch + 1) * len(train_set)/batch_size)

    torch.save({
        'epoch': epoch + 1,
        'state_dict_G': G.state_dict(),
        'optimizer_G': optimizerG.state_dict(),
        'scheduler_G': schedulerG.state_dict(),
        }, checkpoint_path)
    epoch += 1

writer.close()
