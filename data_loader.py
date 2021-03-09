#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : data_loader.py.py
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


# 全局变量


def prepare_sequence(seq: list, item2id: dict) -> list:
    """
    将特征值序列，转换成对应的字典映射值构成的Tensor

    Args:
        seq: 特征值序列
        item2id: 将特征值映射到下标的字典

    Returns:

    """
    idx_list = [item2id[word] if word in item2id else item2id["UNK"] for word in seq]

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

    result_data_list = []  # 保存所有处理好的数据

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

    result_data_list.extend(one_sen_fea_list + two_sen_fea_list + three_sen_fea_list)

    # 保存到文件中
    save_file_path = os.path.join(data_root_dir, "enhanced_data.pkl")
    print(f"[i] 保存数据到文件{save_file_path}中...")
    with open(save_file_path, "wb") as f:
        pickle.dump(result_data_list, f)
    print(f"[i] 保存完毕")


if __name__ == '__main__':
    prepare_data("train")
