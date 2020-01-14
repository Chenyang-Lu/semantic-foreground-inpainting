import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import os
import glob
import warnings
import time
from cs_data_loader import *
from ours_model import *
from util import indexmap2colormap, count_parameters


device = 'cuda:0'
time_total = 0.
batch_size = 1
img_size = (256, 512)
methodology = 'ours_hardthreshold_psp_pooling_resnet18_aug_shareearlylayer'
checkpoint_path = 'checkpoints/Cityscapes/' + methodology +'.pth.tar'
device = torch.device(device if torch.cuda.is_available() else 'cpu')
dataset_dir = 'dataset/Cityscapes/'
img_list = sorted(glob.glob(os.path.join(dataset_dir, 'gtFine_partial', 'val', '*', '*gtFine_color.png')))[100:]


# Define dataloaders
test_set = CSDataset('dataset/Cityscapes/test.csv', transform=transforms.Compose([Rescale(img_size), CSToTensor()]))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)

# model and loss
# G = PSPNet(backend='resnet18', psp_size=512, pretrained=False).to(device)
G = PSPNetShareEarlyLayer(backend='resnet18shareearlylayer', psp_size=512, pretrained=False).to(device)
print(count_parameters(G))

if os.path.isfile(checkpoint_path):
    state = torch.load(checkpoint_path)
    G.load_state_dict(state['state_dict_G'])
else:
    print('No checkpoint found')
    exit()

G.eval()  # Set model to evaluate mode
# Iterate over data.
for i, temp_batch in enumerate(test_loader):

    temp_rgb = temp_batch['rgb'].float().to(device)
    temp_foregd = temp_batch['foregd'].long().to(device)
    temp_partial_bkgd = temp_batch['partial_bkgd'].long().squeeze().to(device)

    with torch.set_grad_enabled(False):
        # pre-processing the input and target on the fly
        foregd_idx = (temp_foregd.float() > 0.5).float()

        time_start = time.time()

        pred_seg, fore_middle_msk = G(temp_rgb, False, device, foregd_idx, use_gt_fore=False)
        pred_seg = np.argmax(pred_seg.to('cpu').numpy().squeeze(), axis=0)

        time_total += time.time() - time_start

        pred_color = indexmap2colormap(pred_seg)
        
        fore_middle_msk = F.interpolate((fore_middle_msk > 0.5).float(), scale_factor=1).int()
        fore_middle_msk = fore_middle_msk.to('cpu').numpy().squeeze()
        fore_middle_msk_color = fore_middle_msk * 255

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(img_list[i][:-16].replace('gtFine_partial', 'predictions') + methodology + '_labelTrainIds.png', pred_seg.astype(np.uint8))
        io.imsave(img_list[i][:-16].replace('gtFine_partial', 'predictions') + methodology + '_color.png', pred_color)
        io.imsave(img_list[i][:-16].replace('gtFine_partial', 'foregd_predictions') + methodology + '_labelTrainIds.png', fore_middle_msk.astype(np.uint8))
        io.imsave(img_list[i][:-16].replace('gtFine_partial', 'foregd_predictions') + methodology + '_color.png', fore_middle_msk_color)

print('total inference time for the val set', time_total)