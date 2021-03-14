#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : data_loader.py
@Time    : 2021/3/9 16:55
@Author  : y4ung
@Desc    : 加载数据
"""

# import
from utils import data_root_dir
import torch
import pickle
import os
import pandas as pd
from tqdm import tqdm
import math
import random


# 全局变量


def prepare_sequence(seq: list, item2id: dict) -> list:
    """
    将特征值序列，转换成对应的字典映射值构成的Tensor

    Args:
        seq: 特征值序列
        item2id: 将特征值映射到下标的字典

    Returns:

    """
    idx_list = [item2id[fea_val] if fea_val in item2id else item2id["UNK"] for fea_val in seq]

    # return torch.tensor(idx_list, dtype=torch.long)
    return idx_list


def prepare_data(file_name="train") -> None:
    """
    训练集的训练数据较少，只有7809个句子，需要做数据增强
    1. 将词换成对应的下标
    2. 数据增强，将多个句子对应特征的数值列表进行合并

    Args:
        file_name: 训练集/测试集文件的名字

    Returns:
        None
    """
    print("[i] 准备数据，将提取的特征转换为数值下标，并进行数据增强")

    # 读取特征映射字典
    with open("./data/map_dict.pkl", "rb") as f:
        map_dict = pickle.load(f)

    # 读取处理后的特征csv文件
    file_path = os.path.join(data_root_dir, f"{file_name}.csv")
    df = pd.read_csv(file_path, sep=",")

    num_of_row = len(df)
    # 找到"sep"所在行的索引，添上-1和num_of_row分别是因为第一句话的起始是0，最后一句话末尾没有sep
    sep_idx_list = [-1] + df[df["word"] == "sep"].index.tolist() + [num_of_row]

    # 下面是不同级别的句子特征集合
    # [
    #   [[word特征值],[label特征值],...,[pinyin特征值]]_{句子1},
    #   [[word特征值],[label特征值],...,[pinyin特征值]]_{句子2}, ...
    # ]
    one_sen_fea_list = []  # 单个句子的特征集合
    two_sen_fea_list = []  # 增强后的两个句子的特征集合
    three_sen_fea_list = []  # 增强后的三个句子的特征集合

    print("[i] 处理单个句子的特征")
    for i in tqdm(range(len(sep_idx_list) - 1), ascii=True):
        # 获取每句话的起始和结束下标
        st_idx = sep_idx_list[i] + 1
        ed_idx = sep_idx_list[i + 1]
        sentence_fea_list = []  # [word_idx]
        for feature in df.columns:  # 获取当前句子每一个特征的one-hot
            fea_dict = map_dict[feature][1]  # map_dict[feature]是不同方向映射的元组
            fea_num_seq = prepare_sequence(list(df[feature])[st_idx:ed_idx], fea_dict)
            sentence_fea_list.append(fea_num_seq)
        one_sen_fea_list.append(sentence_fea_list)

    # 数据增强，将两个句子、三个句子的特征拼接
    print("[i] 数据增强，多个句子特征进行拼接")
    for i in tqdm(range(len(one_sen_fea_list) - 1), ascii=True):
        first_matrix = one_sen_fea_list[i]
        second_matrix = one_sen_fea_list[i + 1]
        enhanced_matrix = [first_matrix[j] + second_matrix[j] for j in range(len(first_matrix))]  # 将对应维度的特征数据进行拼接
        two_sen_fea_list.append(enhanced_matrix)

    for i in tqdm(range(len(one_sen_fea_list) - 2), ascii=True):
        first_matrix = one_sen_fea_list[i]
        second_matrix = one_sen_fea_list[i + 1]
        third_matrix = one_sen_fea_list[i + 2]
        enhanced_matrix = [first_matrix[j] + second_matrix[j] + third_matrix[j] for j in
                           range(len(first_matrix))]  # 将对应维度的特征数据进行拼接
        three_sen_fea_list.append(enhanced_matrix)

    # 通过extend将所有的数据都合在一起
    # [
    #   [[word特征值],[label特征值],...,[pinyin特征值]]_{句子1},
    #   [[word特征值],[label特征值],...,[pinyin特征值]]_{句子2}, ...
    #   [[word特征值],[label特征值],...,[pinyin特征值]]_{增强的句子1}, ...
    # ]
    prepared_data_list = []  # 保存所有处理好的数据
    prepared_data_list.extend(one_sen_fea_list + two_sen_fea_list + three_sen_fea_list)

    # 保存到文件中
    save_file_path = os.path.join(data_root_dir, "prepared_data.pkl")
    print(f"[i] 保存数据到文件{save_file_path}中...")
    with open(save_file_path, "wb") as f:
        pickle.dump(prepared_data_list, f)
    print(f"[i] 保存完毕")


class BatchLoader(object):
    """
    加载batch数据的工具类
    """

    def __init__(self, batch_size, file_name="prepared_data"):
        """
        类初始化函数

        Args:
            batch_size: 一个batch的样本数
            dir_name: 训练集/测试集的文件夹名称
        """
        self.batch_size = batch_size
        self.file_name = file_name
        file_path = os.path.join(data_root_dir, file_name + ".pkl")
        print(f"[i] 读取文件: {file_path}")
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)
        self.num_of_batch, self.batch_data_list = self.sort_and_pad()
        self.batch_data_list_len = len(self.batch_data_list)

    def sort_and_pad(self) -> tuple:
        """
        按照句子的长度进行排序，使得长度相近的句子排在一起，避免太长和太短的句子放到一个batch，导致短句padding得太多

        Returns:
            num_of_batch: batch的数量
            batch_data_list: 每个batch数据的集合列表
        """
        num_of_batch = int(math.ceil(len(self.data) / self.batch_size))  # batch的数量，一共有多少个batch
        sorted_data = sorted(self.data, key=lambda x: len(x[0]))  # 按照句子长度进行排序
        batch_data_list = []

        for i in range(num_of_batch):
            batch_data = sorted_data[
                         i * int(self.batch_size): (i + 1) * int(self.batch_size)]  # 按照batch_size从排序后的句子数据集中获取数据
            padded_batch_data = self.pad_data(batch_data)  # 按照该batch中最长句子的长度进行padding操作
            batch_data_list.append(padded_batch_data)

        return num_of_batch, batch_data_list

    def get_num_of_batch(self):
        """
        获取数据集处理后的batch的数量

        Returns:
            self.num_of_batch: batch的数量
            need_except: 是否需要排除最后一个batch(样本数不满足batch_size)
        """
        if len(self.batch_data_list[-1][0]) != self.batch_size:
            need_except = True
        else:
            need_except = False
        return self.num_of_batch, need_except

    @staticmethod
    def pad_data(batch_data: list) -> list:
        """
        对每个bacth的数据进行填充，然后返回一个batch的数据list

        Args:
            batch_data: 一个batch的数据，需要根据最长的句子的长度进行填充

        Returns:
            padded_batch_data: 返回一个对该batch中各个特征都进行填充以后的batch数据
                               [
                                 [ [填充后的句子1的word向量], [填充后的句子2的word特征], ...],
                                 [ [填充后的句子1的label向量], [填充后的句子2的label向量], ...],
                                 [ [填充后的句子1的flag向量], [填充后的句子2的flag向量], ...],
                                 ...,
                                 [ [填充后的句子1的pinyin向量], [填充后的句子2的pinyin向量], ...]
                               ]
        """
        word_list = []
        label_list = []
        flag_list = []
        bound_list = []
        radical_list = []
        pinyin_list = []
        max_length = max([len(sentence[0]) for sentence in batch_data])  # 当前batch中最大的句子长度

        for sentence_fea_matrix in batch_data:  # 对于原始batch_data中每个句子的特征矩阵
            word, label, flag, bound, radical, pinyin = sentence_fea_matrix  # 分别获取每个句子的特征矩阵
            padding_part = [0] * (max_length - len(word))  # 根据长度决定需要填充内容的长度
            # 对当前句子的不同特征向量进行padding，并添加到相应的特征矩阵中
            word_list.append(word + padding_part)
            label_list.append(label + padding_part)
            flag_list.append(flag + padding_part)
            bound_list.append(bound + padding_part)
            radical_list.append(radical + padding_part)
            pinyin_list.append(pinyin + padding_part)

        padded_batch_data = [word_list, label_list, flag_list, bound_list, radical_list, pinyin_list]

        return padded_batch_data

    def split_fea_label(self, one_batch_data, label_idx=1):
        """
        将一个batch中的数据和label分开

        Args:
            one_batch_data: 一个batch的数据
            label_idx: label在该batch中的索引

        Returns:

        """
        fea_data = one_batch_data[0:label_idx] + one_batch_data[label_idx+1:]
        label_data = one_batch_data[label_idx:label_idx+1]

        return fea_data, label_data

    def iter_batch(self, shuffle=False) -> list:
        """
        返回一个batch的数据

        Args:
            shuffle: 是否需要打乱数据

        Returns:
            fea_data: 一batch的数据
                [
                    [ [填充后的句子1的word向量], [填充后的句子2的word特征], ...],

                    [ [填充后的句子1的flag向量], [填充后的句子2的flag向量], ...],
                    ...,
                    [ [填充后的句子1的pinyin向量], [填充后的句子2的pinyin向量], ...]
                ]

            label_data:
                [
                    [ [填充后的句子1的label向量], [填充后的句子2的label向量], ...]
                ]
        """
        if shuffle:
            random.shuffle(self.batch_data_list)

        for i in range(self.batch_data_list_len):
            one_batch_data = self.batch_data_list[i]
            fea_data, label_data = self.split_fea_label(one_batch_data, 1)
            label_data = label_data[0]
            yield fea_data, label_data


if __name__ == '__main__':
    # prepare_data("train")
    batch_loader = BatchLoader(10, "prepared_data")
    fea_data, label_data = next(batch_loader.iter_batch())
    print(len(fea_data), "\n\n", label_data)
    print(batch_loader.get_num_of_batch())