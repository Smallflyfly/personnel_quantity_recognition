#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/8/4 9:49 
"""
import argparse
import csv
import os

import torch
from torch.backends import cudnn
from tqdm import tqdm

from config import get_config
from logger import create_logger
from models import build_model
from torchvision import transforms

from utils import load_pretrained, resize_image
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    # parser.add_argument('--fused_window_process', action='store_true', help='Fused window shift & window partition, similar for reversed part.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.6756, 0.5652, 0.5181], [0.2153, 0.2033, 0.2087])
])


CSV_FILE = 'result.csv'
TYPE = ['without_mask', 'with_mask', 'mask_weared_incorrect']


def create_csv(all_results):
    with open(CSV_FILE, 'wt', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        rows = ['path', 'label', 'prob']
        csv_writer.writerow(rows)
        csv_writer.writerows(all_results)


def test(config):
    files = os.listdir('data/test')
    model = build_model(config)
    load_pretrained(config, model, logger)
    model = model.cuda()
    model.eval()
    softmax = torch.nn.Softmax(dim=-1)
    outs = []
    for file in tqdm(files, ncols=120):
        im = resize_image(os.path.join('data/test', file), config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
        im = transforms(im)
        im = im.cuda()
        im = im.unsqueeze(0)
        out = model(im)
        out = softmax(out).cpu().detach().numpy()[0]
        idx = np.argmax(out)
        p = out[idx]
        outs.append([file, TYPE[idx], p])
    create_csv(outs)


if __name__ == '__main__':
    args, config = parse_option()
    torch.cuda.set_device(config.LOCAL_RANK)
    cudnn.benchmark = True
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")
    test(config)
