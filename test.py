#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : test.py
@Time    : 2021/3/9 19:10
@Author  : y4ung
@Desc    : 
"""
import pickle
from utils import load_sentences
import torch


train_file = "./data/NER_toy_data/train.txt"


if __name__ == '__main__':
    # train_sentence_list, train_label_list = load_sentences(train_file)
    # print(len(train_sentence_list))

    with open("./data/prepared_data.pkl", "rb") as f:
        prepared_data = pickle.load(f)
        for sentence_fea in prepared_data:
            sentence_fea = torch.tensor(sentence_fea)
            print(sentence_fea.shape)
            for fea in sentence_fea:
                print(fea.shape)
            break




