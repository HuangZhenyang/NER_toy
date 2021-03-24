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
import numpy as np

# 全局变量
model_save_path = "./data/model.pkl"
process_data_save_path = "./data/train_process_data.pkl"
# 解析参数
parser = argparse.ArgumentParser(description="Pytorch NER Toy")
parser.add_argument('--epoch', type=int, help='Number of epoch', default=20)
parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
parser.add_argument('--hidden_dim', type=int, help='Hidden dimension of BiLSTM', default=128)
parser.add_argument('--word_embed_dim', type=int, help='Word embedding dimension', default=100)
parser.add_argument('--flag_embed_dim', type=int, help='Flag embedding dimension', default=50)
parser.add_argument('--bound_embed_dim', type=int, help='Bound embedding dimension', default=50)
parser.add_argument('--radical_embed_dim', type=int, help='Radical embedding dimension', default=50)
parser.add_argument('--pinyin_embed_dim', type=int, help='Pinyin embedding dimension', default=80)
args = parser.parse_args()


def info_plot(epoch_num, train_loss_list, train_acc_list, valid_x, valid_loss_list, valid_f1_list):
    """
    可视化训练过程中指标的变化情况

    Args:
        epoch_num: epoch的数量
        train_loss_list: 训练集上loss的数据
        train_acc_list: 训练集上准确率的数据
        valid_x: 验证集上的下标
        valid_loss_list: 验证集上loss的数据
        valid_f1_list: 验证集上准确率的数据

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
    plt.subplot(2, 1, 2)
    plt.plot(train_x, train_acc_list, "-", color="red", label="training f1")
    plt.plot(valid_x, valid_f1_list, ".-", color="green", label="validating f1")
    plt.legend()
    plt.title("f1 vs. epoch")
    plt.xlabel("epoch")
    plt.ylabel("f1")

    plt.show()


def check_pred(config, model, file_name):
    """
    在训练之前测试一下模型的预测功能

    Args:
        config: 配置类的实例对象
        model: 模型的实例对象
        file_name: 准备好的数据集文件名

    Returns:
        None
    """
    print("[i] check_pred: 训练之前测试模型的预测功能")
    model.eval()
    batch_loader = BatchLoader(config.batch_size, f"prepared_{file_name}_data")
    fea_data, label_data, init_sentence_len = next(batch_loader.iter_batch())
    fea_data, label_data, init_sentence_len = torch.tensor(fea_data), \
                                              torch.tensor(label_data), \
                                              torch.tensor(init_sentence_len)
    with torch.no_grad():
        print(model(fea_data))


def get_tag_bound_list(y_list, label2id):
    """
    获取实体的边界数组

    Args:
        y_list: label list
        label2id: label to index

    Returns:
        tag_bound_list: [ [3, 5], [8, 11], ...]
    """
    tag_bound_list = []
    begin_idx_list = [label2id[i] for i in label2id.keys() if "B-" in i]
    inner_idx_list = [label2id[i] for i in label2id.keys() if "I-" in i]
    other_idx = label2id.get("O")
    BEGIN_INI = -1  # 当前tag的起始的默认值
    begin = BEGIN_INI  # 当前tag的起始
    end = 0  # 当前tag的结束索引 + 1 (加1后的索引是当前tag的下一个O的)
    y_list_len = len(y_list)

    for i, label_idx in enumerate(y_list):  # 对于标签集合中的每一个标签
        if begin == BEGIN_INI and label_idx in begin_idx_list:  # 以B-开头
            begin = i
        elif begin != BEGIN_INI and label_idx == other_idx:  # 当前tag的起始是有值的，并且当前的label index是O的index
            end = i
            tag_bound_list.append([begin, end])
            begin = BEGIN_INI
        elif begin != BEGIN_INI and label_idx in begin_idx_list:  # 当前tag的起始是有值的，并且遇到了下一个实体的起始（连着两个实体）
            end = i
            tag_bound_list.append([begin, end])
            begin = i
        elif begin != BEGIN_INI and i == y_list_len - 1:  # 当前tag的起始是有值的，并且I-的index是y_list的最后一个值
            end = i
            tag_bound_list.append([begin, end])
            begin = BEGIN_INI

    return tag_bound_list


def metric(y_true, y_pred):
    """
    micro, 计算整体的精确率，召回率和f1 score

    Args:
        y_true: 真实标签，[ ... ]
        y_pred: 预测标签，[ ... ]

    Returns:
        precision: 精确率
        recall: 召回率
        f1:  F1 Score
    """
    # 找出y_true和y_pred中的实体边界
    assert len(y_true) == len(y_pred)

    with open("./data/map_dict.pkl", "rb") as f:
        map_dict = pickle.load(f)
        _, label2id = map_dict["label"]

    true_tag_bound_list = get_tag_bound_list(y_true, label2id)
    pred_tag_bound_list = get_tag_bound_list(y_pred, label2id)

    # 计算TP, FP, FN
    TP = 0.
    FP = 0.
    FN = 0.
    for pred_tag_bound in pred_tag_bound_list:
        if pred_tag_bound in true_tag_bound_list:  # 预测的有 P，真实的有 P => TP
            TP += 1
        else:  # 预测的有 P，真实的没有 N => FP
            FP += 1

    for true_tag_bound in true_tag_bound_list:
        if true_tag_bound not in pred_tag_bound_list:  # 预测的没有 N，真实的有 P => FN
            FN += 1

    # 计算精确率、召回率、F1-Score
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


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
    model.eval()
    batch_loader = BatchLoader(config.batch_size, f"prepared_{test_type}_data")
    num_of_batch, need_except = batch_loader.get_num_of_batch()

    with torch.no_grad():
        test_loss = 0.0
        total_true_label = torch.tensor([])
        total_pred_label = torch.tensor([])

        for fea_data, label_data, init_sentence_len in tqdm(batch_loader.iter_batch(shuffle=True), total=num_of_batch):
            if len(fea_data[0]) != config.batch_size:  # 对于最后一个不满足batch_size的batch，直接跳过
                continue

            fea_data, label_data, init_sentence_len = torch.tensor(fea_data), \
                                                      torch.tensor(label_data), \
                                                      torch.tensor(init_sentence_len)

            _, batch_best_path = model(fea_data)

            if (not total_true_label.numel()) and (not total_pred_label.numel()):  # 如果还未初始化
                total_true_label = torch.cat(
                    [sentence[:sentence_end] for sentence, sentence_end in zip(label_data, init_sentence_len)])
                total_pred_label = torch.cat(
                    [sentence[:sentence_end] for sentence, sentence_end in zip(batch_best_path, init_sentence_len)])
            else:
                tmp_true_label = torch.cat(
                    [sentence[:sentence_end] for sentence, sentence_end in zip(label_data, init_sentence_len)])
                tmp_pred_label = torch.cat(
                    [sentence[:sentence_end] for sentence, sentence_end in zip(batch_best_path, init_sentence_len)])

                total_true_label = torch.cat([total_true_label, tmp_true_label], -1)
                total_pred_label = torch.cat([total_pred_label, tmp_pred_label], -1)

            # 计算loss
            loss = model.calc_loss(fea_data, label_data)
            test_loss += loss.mean().item()

    # loss
    test_loss = test_loss / num_of_batch
    # 精确率，召回率，f1_score
    precision, recall, f1 = metric(total_true_label, total_pred_label)

    return test_loss, precision, recall, f1


def train(config, model, optimizer, file_name):
    """
    模型训练

    Args:
        config: 配置类的实例对象
        model: 模型的实例对象
        optimizer: 优化器
        file_name: 准备好的数据集文件名

    Returns:
        None
    """
    print(f"[i] 开始训练...")
    epoch_num = config.epoch
    batch_loader = BatchLoader(config.batch_size, f"prepared_{file_name}_data")
    num_of_batch, need_except = batch_loader.get_num_of_batch()
    if need_except:
        num_of_batch -= 1

    train_loss_list = []  # 保存每个epoch的loss
    train_f1_list = []  # 保存每个epoch的准确率
    valid_x = []  # 验证集的epoch下标
    valid_loss_list = []  # 验证集上的loss集合
    valid_f1_list = []  # 验证集上的f1_score集合
    valid_best_f1 = -999  # 验证集上最优的f1_score

    for epoch in range(epoch_num):
        print("=== Epoch {}/{} ===".format(epoch + 1, epoch_num))
        model.train()
        st_time = time.time()
        train_loss = 0.0
        total_true_label = torch.tensor([])
        total_pred_label = torch.tensor([])

        # 将所有的batch喂给模型进行训练
        for fea_data, label_data, init_sentence_len in tqdm(batch_loader.iter_batch(shuffle=True), ascii=True,
                                                            total=num_of_batch):
            if len(fea_data[0]) != config.batch_size:  # 对于最后一个不满足batch_size的batch，直接跳过
                continue

            fea_data, label_data, init_sentence_len = torch.tensor(fea_data), \
                                                      torch.tensor(label_data), \
                                                      torch.tensor(init_sentence_len)

            # PyTorch默认会累积梯度; 而我们需要每条样本单独算梯度，因此需要重置
            model.zero_grad()

            _, batch_best_path = model.forward(fea_data)

            if (not total_true_label.numel()) and (not total_pred_label.numel()):  # 如果还未初始化
                total_true_label = torch.cat(
                    [sentence[:sentence_end] for sentence, sentence_end in zip(label_data, init_sentence_len)])
                total_pred_label = torch.cat(
                    [sentence[:sentence_end] for sentence, sentence_end in zip(batch_best_path, init_sentence_len)])
            else:
                tmp_true_label = torch.cat(
                    [sentence[:sentence_end] for sentence, sentence_end in zip(label_data, init_sentence_len)])
                tmp_pred_label = torch.cat(
                    [sentence[:sentence_end] for sentence, sentence_end in zip(batch_best_path, init_sentence_len)])

                total_true_label = torch.cat([total_true_label, tmp_true_label], -1)
                total_pred_label = torch.cat([total_pred_label, tmp_pred_label], -1)

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
        # 精确率，召回率，f1_score
        precision, recall, f1 = metric(total_true_label, total_pred_label)
        train_f1_list.append(f1)
        # 花费时间
        ed_time = time.time()
        sp_time = ed_time - st_time

        print(
            "[i] loss: {:.4f}, precision_score: {:.4f}, recall_score: {:.4f}, f1_score: {:.4f}, spending time: {:.2f}s".format(
                train_loss, precision, recall, f1, sp_time))

        # 在验证集上测试
        if (epoch + 1) % 2 == 0:
            valid_x.append(epoch)
            valid_loss, valid_precision_score, valid_recall_score, valid_f1_score = test(model)
            print("[i] 验证集. loss: {:.4f}, precision_score: {:.4f}, recall_score: {:.4f}, f1_score: {:.4f}".format(
                valid_loss, valid_precision_score, valid_recall_score, valid_f1_score))

            # 保存在验证集上效果最好的模型
            if valid_f1_score > valid_best_f1:
                print("[i] 验证集上的效果: {:.4f} 优于已有的最优效果: {:.4f}".format(valid_f1_score, valid_best_f1))
                valid_best_f1 = valid_f1_score
                # 将训练好的模型保存到文件中
                print(f"[i] 保存模型到文件{model_save_path}中...")
                with open(model_save_path, "wb") as f:
                    pickle.dump(model, f)
                print(f"[i] 保存完毕")
            valid_loss_list.append(valid_loss)
            valid_f1_list.append(valid_f1_score)

    # 将训练过程中的数据保存到文件中，方便可视化
    train_process_data = {
        "epoch_num": epoch_num,
        "train_loss_list": train_loss_list,
        "train_f1_list": train_f1_list,
        "valid_x": valid_x,
        "valid_loss_list": valid_loss_list,
        "valid_f1_list": valid_f1_list
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
    check_pred(config, model, "train")

    # 训练
    train(config, model, optimizer, "train")

    # 训练过程可视化
    with open(process_data_save_path, "rb") as f:
        train_process_data = pickle.load(f)
    epoch_num = train_process_data["epoch_num"]
    train_loss_list = train_process_data["train_loss_list"]
    train_f1_list = train_process_data["train_f1_list"]
    valid_x = train_process_data["valid_x"]
    valid_loss_list = train_process_data["valid_loss_list"]
    valid_f1_list = train_process_data["valid_f1_list"]

    info_plot(epoch_num, train_loss_list, train_f1_list, valid_x, valid_loss_list, valid_f1_list)
