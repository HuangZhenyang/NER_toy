#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : test.py
@Time    : 2021/3/9 19:10
@Author  : y4ung
@Desc    : 
"""

from utils import load_sentences


train_file = "./data/NER_toy_data/train.txt"


if __name__ == '__main__':
    train_sentence_list, train_label_list = load_sentences(train_file)
    print(len(train_sentence_list))