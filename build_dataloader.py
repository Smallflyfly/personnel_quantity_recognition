#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/8/3 9:46 
"""
from timm.data import create_transform, Mixup
from torch.utils.data import DataLoader

from dataset import PersonDataset


def build_dataset(is_train=True, config=None, image_list=None):
    if is_train:
        transforms = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
            mean=config.DATA.MEAN,
            std=config.DATA.STD
        )
        dataset = PersonDataset(config.DATA.ROOT_PATH, config.DATA.LABEL_FILE, width=config.DATA.IMG_SIZE,
                              height=config.DATA.IMG_SIZE, transforms=transforms, image_list=image_list)
    else:
        dataset = PersonDataset(config.DATA.ROOT_PATH, config.DATA.LABEL_FILE, width=config.DATA.IMG_SIZE,
                              height=config.DATA.IMG_SIZE, transforms=None, image_list=image_list)
    return dataset


def build_dataloader(config):
    config.defrost()
    train_dataset = build_dataset(is_train=True, config=config)
    config.freeze()
    dataset_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )
    mixup_fn = None
    # mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #         prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #         label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES
    #     )
    return dataset_loader, mixup_fn


def build_dataloader_k(config, is_train=True, image_list=None, batch_size=1):
    dataset = build_dataset(is_train=is_train, config=config, image_list=image_list)
    dataset_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )
    mixup_fn = None
    # if is_train:
    #     mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    #     if mixup_active:
    #         mixup_fn = Mixup(
    #             mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #             prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #             label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES
    #         )
    return dataset_loader, mixup_fn