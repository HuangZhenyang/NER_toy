#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : train.py
@Time    : 2021/3/13 15:02
@Author  : y4ung
@Desc    : 模型训练
"""

# import
import time
import torch
import torch.optim as optim
from tqdm import tqdm
import argparse  # 命令行参数
import time
from model import *
import matplotlib.pyplot as plt

# 全局变量
model_save_path = "./data/model.pkl"
process_data_save_path = "./data/train_process_data.pkl"
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


def info_plot(epoch_num, train_loss_list, train_acc_list, valid_x, valid_loss_list, valid_acc_list):
    """
    可视化训练过程中指标的变化情况

    Args:
        epoch_num: epoch的数量
        train_loss_list: 训练集上loss的数据
        train_acc_list: 训练集上准确率的数据
        valid_x: 验证集上的下标
        valid_loss_list: 验证集上loss的数据
        valid_acc_list: 验证集上准确率的数据

    Returns:
        None
    """

    train_x = range(0, epoch_num)
    plt.subplot(2, 1, 1)

    # loss
    plt.plot(train_x, train_loss_list, "-", color="orange", label="training loss")
    plt.plot(valid_x, valid_loss_list, ".-", color="blue", label="validating loss")
    plt.legend()
    plt.title("loss vs. epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    # accuracy
    plt.plot(train_x, train_acc_list, "-", color="orange", label="training accuracy")
    plt.plot(valid_x, valid_acc_list, ".-", color="blue", label="validating accuracy")
    plt.legend()
    plt.title("accuracy vs. epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.show()


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
    fea_data, label_data, init_sentence_len = next(batch_loader.iter_batch())
    fea_data, label_data, init_sentence_len = torch.tensor(fea_data), \
                                              torch.tensor(label_data), \
                                              torch.tensor(init_sentence_len)
    with torch.no_grad():
        print(model(fea_data))


def test(model, test_type="valid"):
    """
    在验证集 or 测试集上测试效果

    Args:
        model: 模型
        test_type: 测试的类型，valid or test

    Returns:
        loss: 在测试的数据集上的loss
        acc: 在测试的数据集上的准确率

    """
    batch_loader = BatchLoader(config.batch_size, f"prepared_{test_type}_data")
    num_of_batch, need_except = batch_loader.get_num_of_batch()

    with torch.no_grad():
        loss = 0.0
        correct = 0
        total_word_num = 0

        for fea_data, label_data, init_sentence_len in tqdm(batch_loader.iter_batch(), total=num_of_batch):
            if len(fea_data[0]) != config.batch_size:  # 对于最后一个不满足batch_size的batch，直接跳过
                continue

            fea_data, label_data, init_sentence_len = torch.tensor(fea_data), \
                                                      torch.tensor(label_data), \
                                                      torch.tensor(init_sentence_len)
            total_word_num += init_sentence_len.sum().item()

            _, batch_best_path = model.forward(fea_data)

            # 正确预测出的label的个数，不包括padding的部分
            for i in range(len(label_data)):  # 对于每一句话的label
                sentence_len = init_sentence_len[i]
                tmp_label_data = label_data[i][:sentence_len]
                tmp_batch_best_path = batch_best_path[i][:sentence_len]
                correct += (tmp_label_data == tmp_batch_best_path).sum().item()

            # 计算loss
            loss = model.calc_loss(fea_data, label_data)
            loss += loss.mean().item()

        # loss
        loss = loss / num_of_batch
        # 准确率
        acc = correct / total_word_num * 100

        return loss, acc


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
    valid_x = []  # 验证集的epoch下标
    valid_loss_list = []  # 验证集上的loss集合
    valid_acc_list = []  # 验证集上的准确率集合

    for epoch in range(epoch_num):
        print("=== Epoch {}/{} ===".format(epoch + 1, epoch_num))
        st_time = time.time()
        train_loss = 0.0
        train_correct = 0.0  # 预测正确的句子数
        total_word_num = 0  # 已经处理过的word数

        # 将所有的batch喂给模型进行训练
        for fea_data, label_data, init_sentence_len in tqdm(batch_loader.iter_batch(), ascii=True, total=num_of_batch):
            if len(fea_data[0]) != config.batch_size:  # 对于最后一个不满足batch_size的batch，直接跳过
                continue

            fea_data, label_data, init_sentence_len = torch.tensor(fea_data), \
                                                      torch.tensor(label_data), \
                                                      torch.tensor(init_sentence_len)
            total_word_num += init_sentence_len.sum().item()

            # PyTorch默认会累积梯度; 而我们需要每条样本单独算梯度，因此需要重置
            model.zero_grad()

            _, batch_best_path = model.forward(fea_data)

            # 正确预测出的label的个数，不包括padding的部分
            for i in range(len(label_data)):  # 对于每一句话的label
                sentence_len = init_sentence_len[i]
                tmp_label_data = label_data[i][:sentence_len]
                tmp_batch_best_path = batch_best_path[i][:sentence_len]
                train_correct += (tmp_label_data == tmp_batch_best_path).sum().item()

            # 计算loss
            loss = model.calc_loss(fea_data, label_data)
            train_loss += loss.mean().item()

            # 反向传播
            loss.mean().backward()

            # 更新参数
            optimizer.step()

        # loss
        train_loss = train_loss / num_of_batch
        train_loss_list.append(train_loss)
        # 准确率
        train_acc = train_correct / total_word_num * 100
        train_acc_list.append(train_acc)
        # 花费时间
        ed_time = time.time()
        sp_time = ed_time - st_time

        print("[i] loss: {:.4f}, training accuracy: {:.4f}%, spending time: {:.2f}".format(train_loss, train_acc,
                                                                                           sp_time))

        # 在验证集上测试
        if (epoch + 1) % 1 == 0:  # TODO:测试通过以后，记得改成5
            valid_x.append(epoch)
            valid_loss, valid_acc = test(model)
            print("[i] 验证集. loss: {:.4f}, accuracy: {:.4f}%".format(valid_loss, valid_acc))
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)

    # 将训练好的模型保存到文件中
    print(f"[i] 保存模型到文件{model_save_path}中...")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[i] 保存完毕")

    # 将训练过程中的数据保存到文件中，方便可视化
    train_process_data = {
        "epoch_num": epoch_num,
        "train_loss_list": train_loss_list,
        "train_acc_list": train_acc_list,
        "valid_x": valid_x,
        "valid_loss_list": valid_loss_list,
        "valid_acc_list": valid_acc_list
    }
    print(f"[i] 保存训练过程数据到文件{process_data_save_path}中...")
    with open(process_data_save_path, "wb") as f:
        pickle.dump(train_process_data, f)
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
    check_pred(config, model)

    # 训练
    train(config, model, optimizer)

    # 训练过程可视化
    with open(process_data_save_path, "rb") as f:
        train_process_data = pickle.load(f)
    epoch_num = train_process_data["epoch_num"]
    train_loss_list = train_process_data["train_loss_list"]
    train_acc_list = train_process_data["train_acc_list"]
    valid_x = train_process_data["valid_x"]
    valid_loss_list = train_process_data["valid_loss_list"]
    valid_acc_list = train_process_data["valid_acc_list"]

    info_plot(epoch_num, train_loss_list, train_acc_list, valid_x, valid_loss_list, valid_acc_list)
