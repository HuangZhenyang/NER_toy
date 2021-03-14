#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : utils.py
@Time    : 2021/3/8 17:13
@Author  : y4ung
@Desc    : 数据预处理；准备数据集处理后的特征csv文件，以及映射字典map_dict
"""

# import
from collections import Counter
import logging
import jieba.posseg as psg  # 结巴分词的词性标注
from cnradical import Radical, RunOption
import re
from tqdm import tqdm
import pandas as pd
import os
import pickle
import jieba

# 全局设置
jieba.setLogLevel(logging.INFO)

# 全局变量
data_root_dir = "./data/"
train_file = "./data/NER_toy_data/train.txt"
valid_file = "./data/NER_toy_data/valid.txt"
test_file = "./data/NER_toy_data/test.txt"
START_TAG = '<START>'
STOP_TAG = '<STOP>'


def load_sentences(file_path: str) -> tuple:
    """
    从文件中加载句子

    Args:
        file_path: 数据集文件

    Returns:
        all_sentence_list: 所有字级别的句子集合
        all_label_list： 所有字级别的标签集合
    """
    print(f"[i] load_sentences：打开文件{file_path}，加载句子")

    all_sentence_list = []  # 保存所有句子中的词
    sentence_list = []  # 保存当前句子中的词
    all_label_list = []  # 保存所有句子中的标签
    label_list = []  # 保存当前句子中的标签

    with open(train_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f.readlines(), ascii=True)):
            if line == "\n":  # 当前句子结束
                if len(sentence_list) == 0:  # 跳过句子为\n的，即连着出现两个\n的情况
                    continue
                all_sentence_list.append(sentence_list)
                sentence_list = []
                all_label_list.append(label_list)
                label_list = []
            else:  # 句子中
                line = line[:-1]
                line_split = line.split(" ")
                label = line_split[-1]
                label_st_idx = line.index(label)
                word = line[:label_st_idx - 1].strip()
                if word == "":
                    word = " "
                sentence_list.append(word)
                label_list.append(label)

    assert [len(sen) for sen in all_sentence_list] == [len(la) for la in all_label_list]

    return all_sentence_list, all_label_list


def label_encoder(label_list: list) -> tuple:
    """
    生成label的one-hot

    Args:
        label_list: 所有的label集合

    Returns:
        id2label: list，id to label
        label2id: dict, label to id
    """
    print("[i] label_encoder：生成label的one-hot")
    id2label = list(set([label for sentence_labels in label_list for label in sentence_labels]))
    label2id = {label: id for id, label in enumerate(id2label)}

    return id2label, label2id


def is_chinese(char: str) -> bool:
    """
    判断字符是不是中文字符

    Args:
        char: 字符
    Returns:
        True: 是中文
        False: 不是中文
    """
    if "\u4e00" <= char <= "\u9fff":
        return True
    else:
        return False


def replace_word(word: str) -> str:
    """
    替换掉一些不可见的词或字符

    Args:
        word: 需要判断和可能需要替换的词或字符

    Returns:
        如果需要替换，则返回替换后的词或字符；否则，原样返回
    """
    if word == "\n":
        return "LB"
    if word in [" ", "\t", "\u2003"]:
        return "SPACE"
    if word.isdigit():
        return "num"

    return word


def extract_features(sentence_list, label_list, save_file_name="train") -> None:
    """
    处理单个数据集文件的文本：读取文本，打上标记，并提取词级别的特征，保存到csv文件中

    Args:
        sentence_list: 所有的句子，[[句子1的字集合], [句子2的字集合]]
        label_list: 所有的标签，[[句子1的label集合], [句子2的label集合]]
        dataset_type: 数据集的类型
        save_file_name: 处理后的数据集保存的文件名

    Returns:
        None
    """
    print("[i] extract_features：提取句子字级别的特征，保存到csv文件")

    # 1. 提取词性和词位特征
    flag_list = []  # 字的词性特征列表
    bound_list = []  # 字的词位特征列表

    for i, sentence in enumerate(tqdm(sentence_list, ascii=True)):
        # 当前句子的词性和词位特征
        word_flags = []
        word_bounds = ["M"] * len(sentence)
        join_sentence = "".join(sentence)
        for word, flag in psg.cut(join_sentence):
            if len(word) == 1:  # 单独成词
                st_idx = len(word_flags)  # word_flags的长度，也就是我们已经处理了sentence中多少个字
                word_bounds[st_idx] = "S"
                word_flags.append(flag)
            else:
                pat = r"^([0-9A-Za-z]+)$"
                if re.match(pat, word):  # 如果是英文和数字的词，可能在原句中占了多个字的位置
                    # if word.islower() or word.isupper():  # 如果是英文和数字的词，可能在原句中占了多个字的位置
                    st_idx = len(word_flags)
                    word_bounds[st_idx] = "B"  # 词首
                    # 找出当前英文数字的词在字级别的原句list中，占了多少个位置
                    add_idx = 1
                    while True:
                        if "".join(sentence[st_idx: st_idx + add_idx]) == word:
                            break
                        else:
                            add_idx += 1
                    word_flags += [flag] * add_idx
                    ed_idx = len(word_flags) - 1
                    word_bounds[ed_idx] = "E"  # 词尾
                else:
                    st_idx = len(word_flags)
                    word_bounds[st_idx] = "B"  # 词首
                    word_flags += [flag] * len(word)
                    ed_idx = len(word_flags) - 1
                    word_bounds[ed_idx] = "E"  # 词尾
        flag_list.append(word_flags)
        bound_list.append(word_bounds)

    # 2. 提取拼音和偏旁部首特征
    radical = Radical(RunOption.Radical)  # 用来提取偏旁部首
    pinyin = Radical(RunOption.Pinyin)  # 用来提取拼音
    radical_list = [
        [radical.trans_ch(word) if is_chinese(word) and radical.trans_ch(word) else "UNK" for word in sentence] for
        sentence in
        sentence_list]
    pinyin_list = [[pinyin.trans_ch(word) if is_chinese(word) and pinyin.trans_ch(word) else "UNK" for word in sentence]
                   for sentence in
                   sentence_list]

    assert len(flag_list) == len(bound_list) == len(radical_list) == len(pinyin_list)

    # 3. 保存到文件中
    data = dict()  # 保存处理后的所有特征
    data["word"] = sentence_list
    data["label"] = label_list
    data["flag"] = flag_list
    data["bound"] = bound_list
    data["radical"] = radical_list
    data["pinyin"] = pinyin_list

    num_of_rows = len(sentence_list)
    num_of_columns = len(data.keys())
    dataset = []
    for i in range(num_of_rows):
        records = list(zip(*[list(v[i]) for v in data.values()]))  # 一句话中，每个词的特征，在这一步，也把每句话拆成了单个的词
        dataset += records
        dataset += [tuple(["sep"] * num_of_columns)]  # 一句话添加完以后，再添加一个间隔
    dataset = dataset[:-1]  # 去掉最后一个间隔
    dataset = pd.DataFrame(dataset,
                           columns=["word" if i == "sentence" else i for i in list(data.keys())])  # 转换成DataFrame
    dataset["word"] = dataset["word"].apply(replace_word)
    # 保存
    save_file_name = save_file_name
    save_path = os.path.join(data_root_dir, save_file_name + ".csv")
    print(f"[i] 正在保存到文件{save_path}中...")
    dataset.to_csv(save_path, index=False, encoding="utf-8")
    print(f"[i] 已保存到文件{save_path} :D")


def feature_encoder(feature_val_list: list, threshold=10, is_word=False, sep="sep", is_label=False) -> tuple:
    """
    将特征映射到下标，构建one-hot

    Args:
        feature_val_list: DataFrame中一个特征的值列表
        threshold: 词频的阈值，用于过滤出现次数不足的词
        is_word: 是否是词特征
        sep: 句子结束填充的分隔符
        is_label: 判断是否是标签

    Returns:
        id2item: id映射到item，list
        item2id: item映射到id，dict
    """
    counter = Counter(feature_val_list)
    if sep:
        counter.pop(sep)

    if is_word:
        counter["PAD"] = 100000001  # 句子长度不一致时，用PAD填充，保证一个batch中样本的长度一致，将其设的很大，从而保证排序以后下标为0
        counter["UNK"] = 100000000
        feature_val_list = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        feature_val_list = [x[0] for x in feature_val_list if x[1] >= threshold]
        id2item = feature_val_list
        item2id = {id2item[i]: i for i in range(len(id2item))}
    elif is_label:
        counter[START_TAG] = -100000000
        counter[STOP_TAG] = -100000001
        feature_val_list = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        feature_val_list = [x[0] for x in feature_val_list]
        id2item = feature_val_list
        item2id = {id2item[i]: i for i in range(len(id2item))}
    else:
        counter["PAD"] = 100000001
        feature_val_list = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        feature_val_list = [x[0] for x in feature_val_list]
        id2item = feature_val_list
        item2id = {id2item[i]: i for i in range(len(id2item))}

    return id2item, item2id


def generate_dict(file_path: str, save_file_name: str) -> None:
    """
    生成每个特征数值的映射字典

    Args:
        file_path: 处理后包含原始特征的数据集的文件路径
        save_file_name: 保存的文件名

    Returns:
        None
    """
    print("[i] generate_dict: 生成每个特征数值的映射字典，用于构建特征的one-hot")

    map_dict = dict()  # 每个值是一个元组，(id2item, item2id)

    # 读取文件
    df = pd.read_csv(file_path, sep=",")
    word_list = df["word"].tolist()
    label_list = df["label"].tolist()
    flag_list = df["flag"].tolist()
    bound_list = df["bound"].tolist()
    radical_list = df["radical"].tolist()
    pinyin_list = df["pinyin"].tolist()

    map_dict["word"] = feature_encoder(word_list, threshold=20, is_word=True)
    map_dict["label"] = feature_encoder(label_list, is_label=True)
    map_dict["flag"] = feature_encoder(flag_list)
    map_dict["bound"] = feature_encoder(bound_list)
    map_dict["radical"] = feature_encoder(radical_list)
    map_dict["pinyin"] = feature_encoder(pinyin_list)

    # 保存到文件中
    save_file_name = save_file_name
    save_path = os.path.join(data_root_dir, save_file_name + ".pkl")
    print(f"[i] 保存 map_dict 到文件{save_path}中...")
    with open(save_path, "wb") as f:
        pickle.dump(map_dict, f)
    print(f"[i] 已保存到文件{save_path}")


if __name__ == '__main__':
    # train
    train_sentence_list, train_label_list = load_sentences(train_file)
    train_id2label, train_label2id = label_encoder(train_label_list)  # 这个好像没什么用
    extract_features(train_sentence_list, train_label_list, "train")

    # generate_dict("./data/train.csv", "map_dict")

    # valid
    valid_sentence_list, valid_label_list = load_sentences(valid_file)
    valid_id2label, valid_label2id = label_encoder(valid_label_list)  # 这个好像没什么用
    extract_features(valid_sentence_list, valid_label_list, "valid")

    # test
    test_sentence_list, test_label_list = load_sentences(test_file)
    test_id2label, test_label2id = label_encoder(test_label_list)  # 这个好像没什么用
    extract_features(test_sentence_list, test_label_list, "test")
