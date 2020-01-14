import pandas as pd
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CSDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.examples = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        rgb = io.imread(self.examples.iloc[item, 0])
        foregd = io.imread(self.examples.iloc[item, 1])
        partial_bkgd = io.imread(self.examples.iloc[item, 2])

        example = {'rgb': rgb,
                   'foregd': foregd,
                   'partial_bkgd': partial_bkgd,
                   }

        if self.transform:
            example = self.transform(example)

        return example


class CSToTensor(object):

    def __call__(self, sample):
        rgb = sample['rgb']
        foregd = np.expand_dims(sample['foregd'], 0)
        partial_bkgd = np.expand_dims(sample['partial_bkgd'], 0)

        rgb = rgb.transpose((2, 0, 1))
        rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(torch.from_numpy(rgb))
        foregd = torch.from_numpy(foregd)
        partial_bkgd = torch.from_numpy(partial_bkgd)
        return {'rgb': rgb,
                'foregd': foregd,
                'partial_bkgd': partial_bkgd, 
                }


class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        rgb = sample['rgb']
        foregd = sample['foregd']
        partial_bkgd = sample['partial_bkgd']

        rgb = transform.resize(rgb, self.output_size, mode='constant', preserve_range=False, anti_aliasing=True)
        foregd = transform.resize(foregd, self.output_size, order=0, mode='constant', preserve_range=True,
                                  anti_aliasing=False)
        partial_bkgd = transform.resize(partial_bkgd, self.output_size, order=0, mode='constant', preserve_range=True,
                                        anti_aliasing=False)

        return {'rgb': rgb,
                'foregd': foregd,
                'partial_bkgd': partial_bkgd,
                }


if __name__ == '__main__':
    val_set = CSDataset('dataset/Cityscapes/val.csv',
                            transform=transforms.Compose([Rescale([256, 512]), CSToTensor()]))
    print('number of val examples:', len(val_set))
    print(val_set[0]['rgb'].shape)
    print(val_set[0]['foregd'].shape)
    print(val_set[0]['partial_bkgd'].shape)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=8)
    print('show 3 examples')
    for i, temp_batch in enumerate(val_loader):
        if i == 0:
            print(temp_batch['rgb'])
            print(temp_batch['foregd'])
            print(temp_batch['partial_bkgd'])
        break

