#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : model.py
@Time    : 2021/3/9 20:18
@Author  : y4ung
@Desc    : 模型
"""

# import
import torch
import torch.autograd as autograd
import torch.nn as nn
import pickle
from configparser import ConfigParser
from data_loader import *

# 全局变量

# 参数配置
config_obj = {
    "batch_size": 10,
    "hidden_dim": 4,
    # embeds_list dimension
    "embed": {
        "word_embed_dim": 100,
        "flag_embed_dim": 50,
        "bound_embed_dim": 50,
        "radical_embed_dim": 50,
        "pinyin_embed_dim": 80,
    }
}


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
    在forward algorithm中计算 log sum exp

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

        self.batch_size = self.config["batch_size"]
        self.fea_list = list(self.map_dict.keys())  # 特征名称的集合
        self.fea_list.remove("label")
        self.vocab_size_dict = get_fea_vocab_size(self.map_dict)  # 不同特征及其对应的vocabulary size
        embed_config = self.config["embed"]
        self.embedding_dim = sum(embed_config.values())  # 嵌入后的batch矩阵最内层的维度
        self.hidden_dim = self.config["hidden_dim"]  # 隐藏层的维度
        self.tag_to_ix = map_dict["label"][1]  # 标签的映射字典
        self.tagset_size = self.vocab_size_dict["label"]

        # === BiLSTM ===
        # 为不同的特征创建Embedding层
        self.embed_dict = {
            fea: nn.Embedding(self.vocab_size_dict[fea], embed_config[fea + "_embed_dim"]) if fea != "label" else None
            for fea in map_dict.keys()
        }
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)

        # 全连接层，将LSTM的输出映射到标签的向量空间
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # === CRF ===
        # 转移矩阵的参数。[i][j] 是从i转移到j的得分
        # 学习标签之间的约束条件，i->j就是从i标签转移到j标签的得分
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.hidden = self._init_hidden()  # 初始化隐藏层

    def _embed_concat(self, batch_data):
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
        print(batch_embed.shape)

        return batch_embed

    def _init_hidden(self) -> tuple:
        """
        随机初始化隐藏层的参数

        Returns:
            tuple: 隐藏层参数，tuple中每个元素都是 [2,1,self.hidden_dim // 2]
        """
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        """
        forward 部分

        Args:
            feats: lstm_out，也就是BiLSTM的输出，[batch_size, seq_len, tagset_size]

        Returns:

        """
        # self.batch_size,
        init_alphas = torch.full((1, self.tagset_size), -10000.)  # 初始化一个[1, self.target_size]的二维矩阵，每一维的内容都是-1000

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # alpha_list = []  # 保存单个句子的forward score
        #
        # for i, sentence_feat in enumerate(feats):
        #     for feat in sentence_feat:
        #         alphas_t = []  # The forward tensors at this timestep
        #         for next_tag in range(self.tagset_size):
        #             emit_score = feat[next_tag].view(
        #                 1, -1).expand(1, self.tagset_size)
        #             trans_score = self.transitions[next_tag].view(1, -1)
        #             next_tag_var = forward_var + trans_score + emit_score
        #             alphas_t.append(log_sum_exp(next_tag_var).view(1))
        #         forward_var = torch.cat(alphas_t).view(1, -1)
        #     terminal_var = forward_var
        #     alpha = log_sum_exp(terminal_var)
        #
        #     alpha_list.append(alpha)
        #
        # return alpha_list

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var
        alpha = log_sum_exp(terminal_var)

        return alpha

    def _get_lstm_features(self, fea_data):
        sentence_len = len(fea_data[0][0])
        self.hidden = self._init_hidden()
        # embeds = self.word_embeds(fea_data).view(len(fea_data), 1, -1)
        embeds = self._embed_concat(fea_data)
        # embeds = embeds.view(len(fea_data), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out = lstm_out.view(sentence_len, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags):

        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i], tags[i]] + feat[tags[i]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, fea_data, tags):
        """


        Args:
            fea_data:
            tags:

        Returns:

        """
        feats = self._get_lstm_features(fea_data)  # 得到BiLSTM的feature
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)

        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


if __name__ == '__main__':
    with open("./data/map_dict.pkl", "rb") as f:
        map_dict = pickle.load(f)

    model = BiLSTMCRF(map_dict, config_obj)
    batch_loader = BatchLoader(10, "prepared_data")
    fea_data, label_data = next(batch_loader.iter_batch())
    fea_data, label_data = torch.tensor(fea_data), torch.tensor(label_data)
    model._embed_concat(fea_data)
    # model.neg_log_likelihood(fea_data, label_data)
