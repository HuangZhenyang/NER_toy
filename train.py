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
import pickle
from model import Config, BatchLoader, BiLSTMCRF
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


# 全局变量
torch.cuda.manual_seed(1)  # 为GPU设置随机种子
torch.backends.cudnn.deterministic = True
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0") if use_gpu else torch.device("cpu")
model_save_path = "./data/model.pkl"
process_data_save_path = "./data/train_process_data.pkl"
# 解析参数
parser = argparse.ArgumentParser(description="Pytorch NER Toy")
parser.add_argument('--epoch', type=int, help='Number of epoch', default=10)
parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
parser.add_argument('--hidden_dim', type=int, help='Hidden dimension of BiLSTM', default=128)
parser.add_argument('--word_embed_dim', type=int, help='Word embedding dimension', default=100)
parser.add_argument('--flag_embed_dim', type=int, help='Flag embedding dimension', default=50)
parser.add_argument('--bound_embed_dim', type=int, help='Bound embedding dimension', default=50)
parser.add_argument('--radical_embed_dim', type=int, help='Radical embedding dimension', default=50)
parser.add_argument('--pinyin_embed_dim', type=int, help='Pinyin embedding dimension', default=80)
args = parser.parse_args()

config = Config(args.epoch, args.batch_size, args.hidden_dim, args.word_embed_dim, args.flag_embed_dim,
                args.bound_embed_dim, args.radical_embed_dim, args.pinyin_embed_dim)


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
    plt.plot(train_x, train_acc_list, "-", color="red", label="training accuracy")
    plt.plot(valid_x, valid_acc_list, ".-", color="green", label="validating accuracy")
    plt.legend()
    plt.title("accuracy vs. epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

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
    batch_loader = BatchLoader(config.batch_size, f"prepared_{file_name}_data")
    fea_data, label_data, init_sentence_len = next(batch_loader.iter_batch())
    fea_data, label_data, init_sentence_len = torch.tensor(fea_data).to(device), \
                                              torch.tensor(label_data).float().to(device), \
                                              torch.tensor(init_sentence_len).to(device)

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
        test_loss = 0.0
        total_true_label = torch.tensor([])
        total_pred_label = torch.tensor([])

        for fea_data, label_data, init_sentence_len in tqdm(batch_loader.iter_batch(shuffle=True), total=num_of_batch):
            if len(fea_data[0]) != config.batch_size:  # 对于最后一个不满足batch_size的batch，直接跳过
                continue

            fea_data, label_data, init_sentence_len = torch.tensor(fea_data).to(device), \
                                                      torch.tensor(label_data).float().to(device), \
                                                      torch.tensor(init_sentence_len).to(device)

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
        ds_precision_score, ds_recall_score, ds_f1_score = precision_score(total_true_label, total_pred_label,
                                                                           average="micro"), \
                                                           recall_score(total_true_label, total_pred_label,
                                                                        average="micro"), \
                                                           f1_score(total_true_label, total_pred_label, average="micro")

        return test_loss, ds_precision_score, ds_recall_score, ds_f1_score


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
    train_acc_list = []  # 保存每个epoch的准确率
    valid_x = []  # 验证集的epoch下标
    valid_loss_list = []  # 验证集上的loss集合
    valid_f1_list = []  # 验证集上的f1_score集合
    valid_best_f1 = -999  # 验证集上最优的f1_score

    for epoch in range(epoch_num):
        print("=== Epoch {}/{} ===".format(epoch + 1, epoch_num))
        st_time = time.time()
        train_loss = 0.0
        total_true_label = torch.tensor([])
        total_pred_label = torch.tensor([])

        # 将所有的batch喂给模型进行训练
        for fea_data, label_data, init_sentence_len in tqdm(batch_loader.iter_batch(shuffle=True), ascii=True,
                                                            total=num_of_batch):
            if len(fea_data[0]) != config.batch_size:  # 对于最后一个不满足batch_size的batch，直接跳过
                continue

            fea_data, label_data, init_sentence_len = torch.tensor(fea_data).to(device), \
                                                      torch.tensor(label_data).float().to(device), \
                                                      torch.tensor(init_sentence_len).to(device)

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
        train_precision_score, train_recall_score, train_f1_score = precision_score(total_true_label, total_pred_label,
                                                                                    average="micro"), \
                                                                    recall_score(total_true_label, total_pred_label,
                                                                                 average="micro"), \
                                                                    f1_score(total_true_label, total_pred_label,
                                                                             average="micro")
        # 花费时间
        ed_time = time.time()
        sp_time = ed_time - st_time

        print(
            "[i] loss: {:.4f}, precision_score: {:.4f}, recall_score: {:.4f}, f1_score: {:.4f}, spending time: {:.2f}s".format(
                train_loss, train_precision_score, train_recall_score, train_f1_score, sp_time))

        # 在验证集上测试
        if (epoch + 1) % 2 == 0:
            valid_x.append(epoch)
            valid_loss, valid_precision_score, valid_recall_score, valid_f1_score = test(model)
            print("[i] 验证集. loss: {:.4f}, precision_score: {:.4f}, recall_score: {:.4f}, f1_score: {:.4f}".format(
                valid_loss, valid_precision_score, valid_recall_score, valid_f1_score))

            # 保存在验证集上效果最好的模型
            if valid_f1_score > valid_best_f1:
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
        "train_acc_list": train_acc_list,
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

    model = BiLSTMCRF(map_dict, config)
    model = model.to(device)

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
    train_acc_list = train_process_data["train_acc_list"]
    valid_x = train_process_data["valid_x"]
    valid_loss_list = train_process_data["valid_loss_list"]
    valid_f1_list = train_process_data["valid_f1_list"]

    info_plot(epoch_num, train_loss_list, train_acc_list, valid_x, valid_loss_list, valid_f1_list)