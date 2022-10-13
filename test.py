#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/7/29 16:03 
"""
import csv
import os

from torch import nn
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

from utils import load_pretrained_weight, resize_image
import numpy as np

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.6756, 0.5652, 0.5181], [0.2153, 0.2033, 0.2087])
])

softmax = nn.Softmax()
WIDTH = 256
HEIGHT = 256

CSV_FILE = 'result.csv'

TYPE = ['without_mask', 'with_mask', 'mask_weared_incorrect']


def create_csv(all_results):
    with open(CSV_FILE, 'wt', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        rows = ['path', 'label', 'prob']
        csv_writer.writerow(rows)
        csv_writer.writerows(all_results)


def test():
    files = os.listdir('data/test')
    model = resnet50(num_classes=3)
    model = model.cuda()
    outs = []
    model.eval()
    for file in tqdm(files, ncols=85):
        max_p = -1.0
        index = -1
        for k in range(3, 6):
            weight = 'best_k_train_{}_fold.pth'.format(k)
            load_pretrained_weight(model, weight)
            im = resize_image(os.path.join('data/test', file), WIDTH, HEIGHT)
            im = transforms(im)
            im = im.cuda()
            im = im.unsqueeze(0)
            out = model(im)
            out = softmax(out).cpu().detach().numpy()[0]
            idx = np.argmax(out)
            p = out[idx]
            if p > max_p:
                max_p = p
                index = idx
        outs.append([file, TYPE[index], max_p])

    create_csv(outs)


if __name__ == '__main__':
    test()