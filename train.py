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
from tqdm import tqdm
import argparse  # 命令行参数
import time
from model import *

# 全局变量
# 解析参数
parser = argparse.ArgumentParser(description="Pytorch NER Toy")
parser.add_argument('--epoch', type=int, help='Number of epoch', default=10)
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
    epoch_num = config.epoch
    batch_loader = BatchLoader(config.batch_size, "prepared_data")
    num_of_batch, need_except = batch_loader.get_num_of_batch()
    if need_except:
        num_of_batch -= 1
    train_loss_list = []  # 保存每个epoch的loss
    train_acc_list = []  # 保存每个epoch的准确率

    for epoch in range(epoch_num):
        print("=== Epoch {}/{} ===".format(epoch + 1, epoch_num))
        st_time = datetime.datetime.now()
        train_loss = 0.0
        train_correct = 0.0  # 预测正确的句子数
        total_sample_num = 0  # 已经处理过的句子数

        # 将所有的batch喂给模型进行训练
        for fea_data, label_data in tqdm(batch_loader.iter_batch(), total=num_of_batch):
            if len(fea_data[0]) != config.batch_size:  # 对于最后一个不满足batch_size的batch，直接跳过
                continue

            fea_data, label_data = torch.tensor(fea_data), torch.tensor(label_data)
            curr_batch_size = label_data.size(0)
            total_sample_num += curr_batch_size

            # PyTorch默认会累积梯度; 而我们需要每条样本单独算梯度，因此需要重置
            model.zero_grad()

            _, batch_best_path = model.forward(fea_data)

            # 正确预测出的label的个数
            train_correct += (label_data == batch_best_path).sum().item()

            # 计算loss
            loss = model.calc_loss(fea_data, label_data)
            train_loss += loss.sum().item()

            # 反向传播
            loss.sum().backward()

            # 更新参数
            optimizer.step()

        # loss
        train_loss = train_loss / total_sample_num
        train_loss_list.append(train_loss)
        # 准确率
        train_acc = train_correct / total_sample_num * 100
        train_acc_list.append(train_acc)
        # 花费时间
        ed_time = datetime.datetime.now()
        sp_time = ed_time - st_time

        print("[i] loss: {:.4f}, training accuracy: {:.4f}%, spending time: {:.2f}".format(train_loss, train_acc, sp_time))

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
