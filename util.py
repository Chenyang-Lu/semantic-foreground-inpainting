import torch
import torch.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import metrics.py_img_seg_eval.eval_segm as eval_segm



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def random_rectangle_mask_gene(img_size, batch_size, num_rect, device, max_size_factor):

    msk = torch.ones((batch_size, 1, img_size[0], img_size[1]), device=device).float()

    for batch_idx in range(batch_size):
        for i in range(num_rect):
            pos = np.random.rand(2)
            size = np.maximum(np.random.rand(2), 0.1/max_size_factor) * max_size_factor

            h_start = int(pos[0] * img_size[0])
            w_start = int(pos[1] * img_size[1])
            h = np.minimum(int(size[0] * img_size[0]), img_size[0]-h_start)
            w = np.minimum(int(size[1] * img_size[1]), img_size[1]-w_start)

            msk[batch_idx, :, h_start:h_start+h, w_start:w_start+w] = 0.

    return msk


def indexmap2colormap(seg_index):

    img_size = seg_index.shape
    color_img_road = np.repeat(np.repeat(np.array([[[128, 64, 128]]], dtype=np.uint8), img_size[0], axis=0),
                               img_size[1], axis=1)
    color_img_side = np.repeat(np.repeat(np.array([[[244, 35, 232]]], dtype=np.uint8), img_size[0], axis=0),
                               img_size[1], axis=1)
    color_img_ignore = np.repeat(np.repeat(np.array([[[128, 128, 128]]], dtype=np.uint8), img_size[0], axis=0),
                               img_size[1], axis=1)
    idx_road = np.repeat(np.expand_dims((seg_index == 0).astype(np.uint8), axis=2), 3, axis=2)
    idx_side = np.repeat(np.expand_dims((seg_index == 1).astype(np.uint8), axis=2), 3, axis=2)
    idx_ignore = np.repeat(np.expand_dims((seg_index == 255).astype(np.uint8), axis=2), 3, axis=2)

    black_color = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    color_img = black_color + color_img_road * idx_road + color_img_side * idx_side + color_img_ignore * idx_ignore

    return color_img


def validation_metrics(pred_segs, gt_segs, if_foregd, foregd_segs=None):
    assert pred_segs.shape == gt_segs.shape

    valid_idx = gt_segs != 255
    if if_foregd:
        foregd_segs = foregd_segs > 0.5
        valid_idx = valid_idx * foregd_segs
    pred_segs = pred_segs[valid_idx].reshape(1, -1)
    gt_segs = gt_segs[valid_idx].reshape(1, -1)


    mean_accu = eval_segm.mean_accuracy(pred_segs, gt_segs)
    mean_IU = eval_segm.mean_IU(pred_segs, gt_segs)

    return mean_accu[0], mean_IU[0]
