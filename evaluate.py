#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : evaluate.py
@Time    : 2021/3/14 16:25
@Author  : y4ung
@Desc    : 评估模型效果
"""
# import
import torch
import pickle
from data_loader import BatchLoader
from train import test, model_save_path, config

# 全局变量
with open(model_save_path, "rb") as f:
    model = pickle.load(f)


def test_on_test_ds(model):
    """
    在测试集上进行测试评估

    Args:
        model:模型

    Returns:
        None
    """
    test_loss, test_acc = test(model, "test")
    print("[i] 测试集. loss: {:.4f}, accuracy: {:.4f}%".format(test_loss, test_acc))


def infer(model):
    """
    用训练好的模型对单个batch的句子进行预测

    Args:
        model: 训练好的模型

    Returns:
        None
    """
    # 准备测试数据
    batch_loader = BatchLoader(config.batch_size, "prepared_test_data")
    fea_data, label_data, init_sentence_len = next(batch_loader.iter_batch())
    fea_data, label_data, init_sentence_len = torch.tensor(fea_data), \
                                              torch.tensor(label_data).float(), \
                                              torch.tensor(init_sentence_len)

    with open("./data/map_dict.pkl", "rb") as f:
        map_dict = pickle.load(f)
        id2word, word2id = map_dict["word"]
        id2label, label2id = map_dict["label"]

    word_idx_data = fea_data[0]

    with torch.no_grad():
        _, batch_pred_path = model(fea_data)

    # 打印预测和真实的差距
    print("word\tpred_label\treal_label")
    for i in range(config.batch_size):  # 对于每个句子
        print(f"[i] === 句子[{i + 1}] ===")
        sentence_word = word_idx_data[i]
        pred_path = batch_pred_path[i]
        real_label = label_data[i]

        init_len = init_sentence_len[i]  # 句子的真实长度

        for j in range(init_len):  # 对于句子中的每个字
            line = ""  # 每一行要打印的内容
            line += id2word[sentence_word[j]] + "\t"
            line += id2label[pred_path[j]] + "\t"
            line += id2label[real_label[j].long()]
            print(line)

        print("--------------------------------")  # 不同句子之间的分割


if __name__ == '__main__':
    test_on_test_ds(model)
    infer(model)
