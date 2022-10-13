#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/7/28 10:43 
"""
import os

from torch.utils.data import Dataset
from torchvision import transforms as transform

from utils import resize_image


class PersonDataset(Dataset):
    def __init__(self, root_path, label_file, image_list=None, mode='train', width=640, height=640, transforms=None):
        self.root_path = root_path
        self.label_file = label_file
        self.image_list = image_list
        self.mode = mode
        self.width = width
        self.height = height
        self.train_images = []
        self.train_labels = []
        self.label_map = {}

        self.transforms = transforms if transforms is not None else transform.Compose([
            transform.RandomRotation(30),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize([0.406, 0.378, 0.359], [0.242, 0.226, 0.216])
        ]) if self.mode == 'train' else transform.Compose([
            transform.ToTensor(),
            transform.Normalize([0.406, 0.378, 0.359], [0.242, 0.226, 0.216])
        ])

        self._process_label()
        self._process_data()

    def _process_label(self):
        with open(self.label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                file = line[0]
                label = int(line[1])
                self.label_map[file] = label

    def _process_data(self):
        if self.image_list is not None:
            images = self.image_list
        else:
            images = os.listdir(os.path.join(self.root_path, 'images'))
        for image in images:
            label = self.label_map[image]
            image = os.path.join(self.root_path, 'images', image)
            self.train_images.append(image)
            self.train_labels.append(label)

    def __getitem__(self, index):
        image = self.train_images[index]
        label = self.train_labels[index]
        im = resize_image(image, self.width, self.height)
        im = self.transforms(im)
        return im, label

    def __len__(self):
        return len(self.train_images)
