#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : test.py
@Time    : 2021/3/9 19:10
@Author  : y4ung
@Desc    : 
"""
import pickle

from data_loader import BatchLoader
from utils import load_sentences
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

train_file = "./data/NER_toy_data/train.txt"


if __name__ == '__main__':
    # train_sentence_list, train_label_list = load_sentences(train_file)
    # print(len(train_sentence_list))

    # with open("./data/prepared_data.pkl", "rb") as f:
    #     prepared_data = pickle.load(f)
    #     for sentence_fea in prepared_data:
    #         sentence_fea = torch.tensor(sentence_fea)
    #         print(sentence_fea.shape)
    #         for fea in sentence_fea:
    #             print(fea.shape)
    #         break

    #  embedding的测试
    # word_to_ix = {"hello": 0, "world": 1}
    # # 第一个参数vocab_size是word的数量，也就是word_to_ix中key的数量；
    # # 第二个参数embedding_dim是需要嵌入到多少维的空间中，也就是指定了embedding vector的维度
    # embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
    # # lookup_tensor = torch.tensor([word_to_ix["hello"], word_to_ix["world"]], dtype=torch.long)
    # lookup_tensor = torch.tensor([0, 1], dtype=torch.long)
    # hello_embed = embeds(lookup_tensor)
    # print(hello_embed)
    # print(hello_embed.shape)

    with open("./data/map_dict.pkl", "rb") as f:
        map_dict = pickle.load(f)
        for each in map_dict:
            print(each)
            print(map_dict[each][1])
            print(len(map_dict[each][1]))

    batch_loader = BatchLoader(10, "prepared_train_data")
    # for a, b, c in batch_loader.iter_batch():
    #     print(len(a[0]))

