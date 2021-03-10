#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : model.py
@Time    : 2021/3/9 20:18
@Author  : y4ung
@Desc    : 模型
"""

# import
import torch
import torch.autograd as autograd
import torch.nn as nn
import pickle
from configparser import ConfigParser
from data_loader import *

# 全局变量

# 参数配置
config_obj = {
    # embedding dimension
    "word_embed_dim": 100,
    "flag_embed_dim": 50,
    "bound_embed_dim": 50,
    "radical_embed_dim": 50,
    "pinyin_embed_dim": 80,
}


# TODO: 将每个字的word, flag, bound, radical, pinyin 都转换成嵌入向量，再拼接
# TODO: 不同的特征有不同的embedding，
# TODO: 不同的embedding的vocab size就是len(map_dict[fea][1])


def get_fea_vocab_size(map_dict: dict) -> dict:
    """
    获取不同特征的vocabulary size

    Args:
        map_dict: 不同特征的id2item和item2id

    Returns:
        vocab_size_list: 不同特征及其对应的vocabulary size
    """
    vocab_size_dict = dict()  # 不同特征及其对应的vocabulary size

    for fea in map_dict.keys():
        fea_vocab_size = len(map_dict[fea][1])
        vocab_size_dict[fea] = fea_vocab_size

    return vocab_size_dict


class BiLSTMCRF(nn.Module):
    """
    BiLSTM_CRF 模型
    """

    def __init__(self, map_dict, config):
        """
        初始化函数

        Args:
            map_dict: 不同特征的id2item和item2id
            config: 保存配置的对象
        """
        super(BiLSTMCRF, self).__init__()
        self.map_dict = map_dict
        self.vocab_size_dict = get_fea_vocab_size(self.map_dict)  # 不同特征及其对应的vocabulary size
        self.embed_dict = {
            fea: nn.Embedding(self.vocab_size_dict[fea], config[fea + "_embed_dim"]) if fea != "label" else None for fea
            in map_dict.keys()
        }  # 为不同的特征创建Embedding层

    def embed_concat(self, batch_data):
        """
        对输入做Embedding
        将单个字不同特征经过Embedding之后的向量进行拼接

        Args:
            batch_data: 一个batch的数据

        Returns:

        """
        embedding = []  # batch_data中各个特征嵌入后的矩阵
        # 对于batch_data中的每一个特征矩阵：[batch_size, 单个句子在该特征上的向量长度]
        for i, fea in enumerate(map_dict.keys()):
            if fea == "label":
                continue
            embeds = self.embed_dict[fea](batch_data[i])  # [batch_size, 单个句子在该特征上的向量长度, embed_dim]
            embedding.append(embeds)


if __name__ == '__main__':
    with open("./data/map_dict.pkl", "rb") as f:
        map_dict = pickle.load(f)

    model = BiLSTMCRF(map_dict, config_obj)
    batch_loader = BatchLoader(10, "prepared_data")
    batch_data = torch.tensor(next(batch_loader.iter_batch()))
    model.embed_concat(batch_data)
