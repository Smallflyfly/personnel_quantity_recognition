#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/10/17 15:43 
"""
import pandas


def check():
    file = 'data/train/train_label.csv'
    dataframe = pandas.read_csv(file, 'r', error_bad_lines=False)
    print(dataframe)


if __name__ == '__main__':
    check()