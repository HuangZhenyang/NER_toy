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
<<<<<<< HEAD
from train import test, model_save_path, device
=======
from train import test, model_save_path
>>>>>>> 3af78f0f05d44b51f8f7106062b4899c5f881ec1


# 全局变量
with open(model_save_path, "rb") as f:
    model = pickle.load(f)
<<<<<<< HEAD
    model = model.to(device)
=======
>>>>>>> 3af78f0f05d44b51f8f7106062b4899c5f881ec1


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
    # 准备测试数据
    batch_loader = BatchLoader(10, "prepared_test_data")
    fea_data, label_data, init_sentence_len = next(batch_loader.iter_batch())
<<<<<<< HEAD
    fea_data, label_data, init_sentence_len = torch.tensor(fea_data).to(device), \
                                              torch.tensor(label_data).to(device), \
                                              torch.tensor(init_sentence_len).to(device)
=======
    fea_data, label_data, init_sentence_len = torch.tensor(fea_data), \
                                              torch.tensor(label_data), \
                                              torch.tensor(init_sentence_len)
>>>>>>> 3af78f0f05d44b51f8f7106062b4899c5f881ec1
    with torch.no_grad():
        print(label_data)
        print(model(fea_data))


if __name__ == '__main__':
    test_on_test_ds(model)
    infer(model)




