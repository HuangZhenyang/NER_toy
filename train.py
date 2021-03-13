#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : train.py
@Time    : 2021/3/13 15:02
@Author  : y4ung
@Desc    : 模型训练
"""

# import
import datetime
import torch
import torch.optim as optim

import argparse  # 命令行参数
import time
from model import *

# 全局变量
# 解析参数
parser = argparse.ArgumentParser(description="Pytorch NER Toy")
parser.add_argument('--epoch', type=int, help='Number of epoch', default=100)
parser.add_argument('--batch_size', type=int, help='Batch size', default=10)
parser.add_argument('--hidden_dim', type=int, help='Hidden dimension of BiLSTM', default=128)
parser.add_argument('--word_embed_dim', type=int, help='Word embedding dimension', default=100)
parser.add_argument('--flag_embed_dim', type=int, help='Flag embedding dimension', default=50)
parser.add_argument('--bound_embed_dim', type=int, help='Bound embedding dimension', default=50)
parser.add_argument('--radical_embed_dim', type=int, help='Radical embedding dimension', default=50)
parser.add_argument('--pinyin_embed_dim', type=int, help='Pinyin embedding dimension', default=80)
args = parser.parse_args()


def check_pred(config, model):
    """
    在训练之前测试一下模型的预测功能

    Args:
        config: 配置类的实例对象
        model: 模型的实例对象

    Returns:
        None
    """
    print("[i] check_pred: 训练之前测试模型的预测功能")
    batch_loader = BatchLoader(config.batch_size, "prepared_data")
    fea_data, label_data = next(batch_loader.iter_batch())
    fea_data, label_data = torch.tensor(fea_data), torch.tensor(label_data)
    with torch.no_grad():
        print(model(fea_data))


def train(config, model, optimizer):
    """
    模型训练

    Args:
        config: 配置类的实例对象
        model: 模型的实例对象
        optimizer: 优化器

    Returns:
        None
    """
    print(f"[i] 开始训练...")
    st_time = datetime.datetime.now()
    epoch_num = config.epoch
    batch_loader = BatchLoader(config.batch_size, "prepared_data")

    for epoch in range(epoch_num):
        for fea_data, label_data in batch_loader.iter_batch():
            if len(fea_data[0]) != config.batch_size:
                continue
            fea_data, label_data = torch.tensor(fea_data), torch.tensor(label_data)

            model.zero_grad()  # PyTorch默认会累积梯度; 而我们需要每条样本单独算梯度，因此需要重置

            # 计算loss
            loss = model.calc_loss(fea_data, label_data)

            # 反向传播
            loss.sum().backward()

            # 更新参数
            optimizer.step()

    ed_time = datetime.datetime.now()
    print(f"[i] 训练时间: {(ed_time - st_time).seconds}s")

    # 将训练好的模型保存到文件中
    model_save_path = "./data/model.pkl"
    print(f"[i] 保存模型到文件{model_save_path}中...")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[i] 保存完毕")


if __name__ == '__main__':
    # 创建模型的实例对象
    with open("./data/map_dict.pkl", "rb") as f:
        map_dict = pickle.load(f)

    config = Config(args.epoch, args.batch_size, args.hidden_dim, args.word_embed_dim, args.flag_embed_dim,
                    args.bound_embed_dim, args.radical_embed_dim, args.pinyin_embed_dim)

    model = BiLSTMCRF(map_dict, config)

    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 测试模型的预测功能
    # check_pred(config, model)

    # 训练
    train(config, model, optimizer)
