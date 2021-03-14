#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : model.py
@Time    : 2021/3/9 20:18
@Author  : y4ung
@Desc    : 模型定义部分
"""

# NOTE: <START>和<STOP>只是为了增加CRF转移矩阵的鲁棒性，在源代码中并没有添加到句子中
# TODO: 计算真实路径得分，计算所有路径得分，维特比算法

# import
import torch
import torch.nn as nn
import pickle
from data_loader import *
from train import device

torch.manual_seed(1)

# 全局变量
START_TAG = '<START>'
STOP_TAG = '<STOP>'


class Config(object):
    """
    保存配置的类
    """

    def __init__(self, epoch=100, batch_size=10, hidden_dim=128, word_embed_dim=100, flag_embed_dim=50,
                 bound_embed_dim=50, radical_embed_dim=50, pinyin_embed_dim=80):
        self.epoch = epoch
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embed = {
            "word_embed_dim": word_embed_dim,
            "flag_embed_dim": flag_embed_dim,
            "bound_embed_dim": bound_embed_dim,
            "radical_embed_dim": radical_embed_dim,
            "pinyin_embed_dim": pinyin_embed_dim,
        }


config = Config()  # 配置类的实例化对象


def get_fea_vocab_size(map_dict: dict) -> dict:
    """
    获取不同特征的vocabulary size

    Args:
        map_dict: 不同特征的id2item和item2id

    Returns:
        vocab_size_list: 不同特征及其对应的vocabulary size
    """
    vocab_size_dict = dict()  # 不同特征及其对应的vocabulary size

    for fea in map_dict.keys():
        fea_vocab_size = len(map_dict[fea][1])
        vocab_size_dict[fea] = fea_vocab_size

    return vocab_size_dict


def argmax(vec):
    """
    计算vec中的argmax，即数值最大的下标

    Args:
        vec:

    Returns:

    """
    _, idx = torch.max(vec, 1)

    return idx.item()


def log_sum_exp(vec):
    """

    Args:
        vec:

    Returns:

    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTMCRF(nn.Module):
    """
    BiLSTM_CRF 模型
    """

    def __init__(self, map_dict, config):
        """
        初始化函数

        Args:
            map_dict: 不同特征的id2item和item2id
            config: 保存配置的对象
        """
        super(BiLSTMCRF, self).__init__()
        self.map_dict = map_dict
        self.config = config

        self.batch_size = self.config.batch_size
        self.fea_list = list(self.map_dict.keys())  # 特征名称的集合
        self.fea_list.remove("label")
        self.vocab_size_dict = get_fea_vocab_size(self.map_dict)  # 不同特征及其对应的vocabulary size
        embed_config = self.config.embed
        self.embedding_dim = sum(embed_config.values())  # 嵌入后的batch矩阵最内层的维度，也是喂进BiLSTM的词向量的维度
        self.hidden_dim = self.config.hidden_dim  # 隐藏层的维度
        self.tag_to_ix = map_dict["label"][1]  # 标签的映射字典
        self.tagset_size = self.vocab_size_dict["label"]

        # === BiLSTM ===
        # 为不同的特征创建Embedding层
        self.embed_dict = {
            fea: nn.Embedding(self.vocab_size_dict[fea], embed_config[fea + "_embed_dim"]).to(device) if fea != "label" else None
            for fea in map_dict.keys()
        }
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, batch_first=True,
                            bidirectional=True).to(device)
        self.hidden = self._init_hidden()  # 初始化隐藏层

        # 全连接层，将LSTM的输出映射到标签的向量空间
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size).to(device)

        # === CRF ===
        # 转移矩阵的参数。[i][j] 是从i转移到j的得分
        # 学习标签之间的约束条件，i->j就是从i标签转移到j标签的得分
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)).to(device)
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000  # START_TAG这一行全为-10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000  # STOP_TAG这一列全为-10000

    def _embed_concat(self, batch_data) -> torch.Tensor:
        """
        对输入做Embedding
        将单个字不同特征经过Embedding之后的向量进行拼接

        Args:
            batch_data: 一个batch的数据

        Returns:
            embeds_list: batch_data中各个特征嵌入后的矩阵，[batch_size, seq_len, word的embed_dim+flag的embed_dim+...+pinyin的embed_dim]
        """
        # batch_data中各个特征嵌入后的矩阵，[batch_size, seq_len, word的embed_dim+flag的embed_dim+...+pinyin的embed_dim]
        embeds_list = []
        # 对于batch_data中的每一个特征矩阵：[batch_size, seq_len]
        for i, fea in enumerate(self.fea_list):
            embeds = self.embed_dict[fea](batch_data[i])  # [batch_size, seq_len, embed_dim]
            embeds_list.append(embeds)

        batch_embed = torch.cat(embeds_list, -1)

        return batch_embed

    def _init_hidden(self) -> tuple:
        """
        随机初始化BiLSTM的隐藏层/隐状态的参数

        Returns:
            tuple: 隐藏层参数，tuple中每个元素都是 [2,1,self.hidden_dim // 2]
        """
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2).to(device),
                torch.randn(2, self.batch_size, self.hidden_dim // 2).to(device))

    def _get_lstm_features(self, fea_data):
        """
        对输入的特征数据做嵌入，然后输入到BiLSTM中，得到发射得分矩阵

        Args:
            fea_data: 一batch的数据：[num_of_fea, batch_size, seq_len]，即[特征的数量，batch_size，该batch中最长句子的长度]
                [
                    [ [填充后的句子1的word向量], [填充后的句子2的word特征], ...],

                    [ [填充后的句子1的flag向量], [填充后的句子2的flag向量], ...],
                    ...,
                    [ [填充后的句子1的pinyin向量], [填充后的句子2的pinyin向量], ...]
                ]

        Returns:
            lstm_feats: 发射得分矩阵，[batch_size, seq_len, tagset_size]。最里面每一个向量表示该字的label预测情况
        """
        fea_data = fea_data.to(device)
        self.hidden = self._init_hidden()
        embeds = self._embed_concat(fea_data)
        # print(f"[i] 嵌入后的特征维度：{embeds.shape}")
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # print(f"[i] lstm_out的维度：{lstm_out.shape}")
        lstm_feats = self.hidden2tag(lstm_out)
        # print(f"[i] BiLSTM输出的lstm_feats维度：{lstm_feats.shape}")

        return lstm_feats

    def _calc_real_path_score(self, emission_matrix, tags):
        """
        求正确路径的分值
        P_{realpath} = EmissionScore + TransitionScore = \sum x_{iy_{j}} + \sum t_{y_{i}y_{j}}

        Args:
            emission_matrix: BiLSTM的输出lstm_out，一个batch中所有句子的Emission Score Matrix, [batch_size, seq_len, tagset_size]
            tags: 当前batch的label。[batch_size, seq_len]， e.g. [10, 10]
                  最内层的每个值代表的是当前的字真实的label index

        Returns:
            real_path_score: batch_size个句子的真实路径得分. [batch_size, 1]
        """
        real_path_score = torch.zeros((self.batch_size, 1)).to(device)  # batch_size个句子的真实路径得分
        # 创建[self.batch_size, 1]的START_TAG下标的tensor
        st_tag_tensor = torch.full([self.batch_size, 1], self.tag_to_ix[START_TAG]).to(device)
        tags = torch.cat([st_tag_tensor, tags], -1).to(device)  # 将st_tag_tensor添加到tags中，所以每个句子的tag seq会多一个值

        for i, sentence_emi_mat in enumerate(emission_matrix):  # 对于每个句子的Emission Score: [seq_len, tagset_size]
            sentence_tags = tags[i].long()  # 当前句子的label index序列
            for j, word_emi_mat in enumerate(sentence_emi_mat):  # 对于单个字的Emission Score: [tagset_size]
                real_path_score[i][0] += word_emi_mat[sentence_tags[j + 1]] + \
                                         self.transitions[
                                             sentence_tags[j + 1], sentence_tags[j]]  # 从j+1开始是因为j==0时是START

            real_path_score[i][0] += self.transitions[self.tag_to_ix[STOP_TAG], sentence_tags[-1]]

        # print(f"[i] P_{{realpath}} = \n\t{real_path_score}")

        return real_path_score

    def _calc_all_path_score(self, emission_matrix):
        """
        计算所有路径的得分和

        Args:
            emission_matrix: BiLSTM的输出lstm_out，一个batch中所有句子的Emission Score Matrix, [batch_size, seq_len, tagset_size]

        Returns:
            all_path_score: 每个句子所有路径的得分. [batch_size, 1]
        """
        init_alphas = torch.full((self.batch_size, 1, self.tagset_size),
                                 -10000.).to(device)  # 初始化一个[1, self.target_size]的二维矩阵，每一维的内容都是-10000
        for i in range(self.batch_size):
            init_alphas[i][0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        all_path_score = torch.zeros((self.batch_size, 1))  # batch_size个句子的所有路径得分

        for i, sentence_emi_mat in enumerate(emission_matrix):  # 对于每个句子的Emission Score: [seq_len, tagset_size]
            for j, word_emi_mat in enumerate(sentence_emi_mat):  # 对于单个字的Emission Score: [tagset_size]
                alphas_t = []
                for next_tag in range(self.tagset_size):  # 遍历每个tag
                    # 发射得分
                    # 先将发射得分变成[1,1]，再拓展成[1, tagset_size]
                    emit_score = word_emi_mat[next_tag].view(1, -1).expand(1, self.tagset_size).to(device)
                    # 转移得分, [tagset_size] -> [1, tagset_size]
                    trans_score = self.transitions[next_tag].view(1, -1).to(device)
                    next_tag_var = forward_var[i] + trans_score + emit_score  # [1, tagset_size]
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))  # 添加[1]
                forward_var[i] = torch.cat(alphas_t).view(1, -1)
            terminal_var = forward_var[i] + self.transitions[self.tag_to_ix[STOP_TAG]]
            sentence_all_path_score = log_sum_exp(terminal_var)
            all_path_score[i] = sentence_all_path_score

        # print(f"[i] P_{{1}}+...+P_{{N}} = \n\t{all_path_score}")

        return all_path_score

    def calc_loss(self, fea_data, tags):
        """
        计算loss
        公式为 -\log(\frac{P_{realpath}}{(P_1 + ... + P_N)}) =
              -\log(\frac{e^{s_{realpath}}}{(e^{s_1} + e^{s_2} + ... + e^{s_n})})

        Args:
            fea_data: 一batch的数据：[num_of_fea, batch_size, seq_len]，即[特征的数量，batch_size，该batch中最长句子的长度]
                [
                    [ [填充后的句子1的word向量], [填充后的句子2的word特征], ...],

                    [ [填充后的句子1的flag向量], [填充后的句子2的flag向量], ...],
                    ...,
                    [ [填充后的句子1的pinyin向量], [填充后的句子2的pinyin向量], ...]
                ]
            tags: 一batch的标签数据：[batch_size, seq_len]
                [
                    [ ... ],  # 表示一个句子的label sequence
                    [ ... ],
                      ...
                    [     ]
                ]

        Returns:
            loss: [batch_size, 1]
        """
        feats = self._get_lstm_features(fea_data).to(device)  # 得到BiLSTM输出的发射得分矩阵
        real_path_score = self._calc_real_path_score(feats, tags).to(device)  # 正确路径的分数
        all_path_score = self._calc_all_path_score(feats).to(device)  # 所有路径的分数和

        loss = all_path_score - real_path_score  # -log(正确/所有) = -(log(正确)-log(所有)) = log(所有) - log(正确)

        # print(f"[i] loss is \n\t{loss}")

        return loss

    def _viterbi_decode(self, emission_matrix):
        """
        维特比算法，逆向解码路径

        Args:
            emission_matrix:  BiLSTM的输出lstm_out，一个batch中所有句子的Emission Score Matrix, [batch_size, seq_len, tagset_size]

        Returns:

        """
        batch_path_score = []
        batch_best_path = []

        init_vvars = torch.full((self.batch_size, 1, self.tagset_size), -10000.).to(device)
        for i in range(self.batch_size):
            init_vvars[i][0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars

        for i, sentence_emi_mat in enumerate(emission_matrix):  # 对于每个句子的Emission Score: [seq_len, tagset_size]
            backpointers = []

            for j, word_emi_mat in enumerate(sentence_emi_mat):  # 对于单个字的Emission Score: [tagset_size]
                bptrs_t = []
                viterbivars_t = []

                for next_tag in range(self.tagset_size):
                    next_tag_var = forward_var[i] + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

                forward_var[i] = (torch.cat(viterbivars_t) + word_emi_mat).view(1, -1)
                backpointers.append(bptrs_t)

            # 转移到STOP_TAG
            terminal_var = forward_var[i] + self.transitions[self.tag_to_ix[STOP_TAG]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            # 根据back pointers解码得到最优路径
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == self.tag_to_ix[START_TAG]  # Sanity check
            best_path.reverse()

            batch_path_score.append(path_score)
            batch_best_path.append(best_path)

        batch_path_score, batch_best_path = torch.tensor(batch_path_score), torch.tensor(batch_best_path)

        return batch_path_score, batch_best_path

    def forward(self, fea_data):
        """
        模型inference逻辑

        Args:
            fea_data:

        Returns:

        """
        fea_data = fea_data.to(device)

        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(fea_data)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)

        return score, tag_seq


if __name__ == '__main__':
    with open("./data/map_dict.pkl", "rb") as f:
        map_dict = pickle.load(f)

    model = BiLSTMCRF(map_dict, config)
    with torch.no_grad():
        batch_loader = BatchLoader(config.batch_size, "prepared_data")
        fea_data, label_data = next(batch_loader.iter_batch())
        fea_data, label_data = torch.tensor(fea_data), torch.tensor(label_data)
        print(f"[i] Real Label Data: \n {label_data}")
        # model._embed_concat(fea_data)
        model.calc_loss(fea_data, label_data)
        batch_path_score, batch_best_path = model.forward(fea_data)
        print(
            f"batch_path_score:\n{torch.tensor(batch_path_score)}\n\nbatch_best_path:\n{torch.tensor(batch_best_path)}")
