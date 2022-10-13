#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/7/28 11:28 
"""
import logging
import os
import time

from sklearn.model_selection import KFold
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from dataset import MaskDataset
from utils import seed_it, load_pretrained_weight, build_optimizer, build_scheduler
import tensorboardX as tb
import numpy as np
import torch

K_FOLD = 10
WIDTH = 256
HEIGHT = 256
ROOT_PATH = 'data/train'
BATCH_SIZE = 8
EPOCH = 10

writer = tb.SummaryWriter()

t = time.strftime("%Y%m%d%H%M%S", time.localtime())
train_log = 'logs/' + t + '.log'
logging.basicConfig(filename=train_log, format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)


def val_model(model, val_dataloader):
    model.eval()
    softmax = nn.Softmax()
    sum = 0
    for index, data in enumerate(val_dataloader):
        image, label = data
        image = image.cuda()
        out = model(image)
        out = softmax(out).cpu().detach().numpy()
        id = np.argmax(out)
        sum += 1 if id == label.numpy()[0] else 0
    return sum / len(val_dataloader)


def run_train(model, train_dataloader, val_dataloader, loss_func, optimizer, scheduler, fold):
    best_acc = 0
    for epoch in range(1, EPOCH+1):
        model.train()
        for index, data in enumerate(train_dataloader):
            image, label = data
            image = image.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            out = model(image)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()

            if index % 50 == 0:
                logging.info('Fold:{} Epoch:[{}/{} {}/{}] lr:{:6f} loss:{:6f}'.format(
                    fold + 1, epoch, EPOCH, index, len(train_dataloader), optimizer.param_groups[-1]['lr'],
                    loss.item()))

            idx = fold * EPOCH * len(train_dataloader) + epoch * len(train_dataloader) + index + 1
            if idx % 20 == 0:
                writer.add_scalar('loss', loss, idx)
                writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], idx)

        scheduler.step()
        val_acc = val_model(model, val_dataloader)
        logging.info('Fold:{}/Epoch:{} val acc: {:6f}'.format(fold + 1, epoch, val_acc))
        writer.add_scalar('val_acc', val_acc, fold * EPOCH + epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_k_train_{}_fold.pth'.format(fold + 1))


def train():
    seed = 2022
    seed_it(seed)
    images = os.listdir(os.path.join('data/train', 'images'))
    images = np.array(images)
    folds = KFold(n_splits=K_FOLD, shuffle=True, random_state=seed).split(range(len(images)))
    # model
    model = resnet50(num_classes=3)
    model_path = 'weights/resnet50-19c8e357.pth'
    model = model.cuda()
    load_pretrained_weight(model, model_path)
    # loss
    loss_func = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optim='adam', lr=0.0005)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine')

    cudnn.benchmark = True

    # log
    logger = logging.getLogger()
    KZT = logging.StreamHandler()
    KZT.setLevel(logging.INFO)
    logger.addHandler(KZT)

    for fold, (train_idx, val_idx) in enumerate(folds):
        train_dataset = MaskDataset('data/train', 'data/train/labels.txt', images[train_idx], width=WIDTH, height=HEIGHT)
        val_dataset = MaskDataset('data/train', 'data/train/labels.txt', images[val_idx], width=WIDTH, height=HEIGHT)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
        run_train(model, train_dataloader, val_dataloader, loss_func, optimizer, scheduler, fold)


if __name__ == '__main__':
    train()
    writer.close()