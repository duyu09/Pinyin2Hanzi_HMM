# _*_ coding:utf-8 _*_
"""
@Version  : 1.2.0
@Time     : 2024年12月29日
@Author   : DuYu (@duyu09, 202103180009@stu.qlu.edu.cn)
@File     : py2hz.py
@Describe : 基于隐马尔可夫模型(HMM)的拼音转汉字程序。
@Copyright: Copyright (c) 2024 DuYu (No.202103180009), Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences).
@Note     : 训练集csv文件，要求第一列为由汉语拼音构成的句子，第二列为由汉字构成的句子。
"""

import re
import bz2
import pickle
import numpy as np
import pandas as pd
from hmmlearn import hmm


# 1. 数据预处理：加载CSV数据集
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    sentences = data.iloc[:, 0].tolist()  # 第一列：汉字句子
    pinyins = data.iloc[:, 1].tolist()  # 第二列：拼音句子
    return sentences, pinyins

# 分词函数，确保英文单词保持完整
def segment_sentence(sentence):
    tokens = re.findall(r'[a-zA-Z]+|[一-鿿]', sentence)  # 使用正则表达式分割句子，确保英文单词保持完整
    return tokens

# 2. 构建字典和状态观测集合
def build_vocab(sentences, pinyins):
    hanzi_set = set()
    pinyin_set = set()

    for sentence, pinyin in zip(sentences, pinyins):
        hanzi_set.update(segment_sentence(sentence))
        pinyin_set.update(pinyin.split())

    hanzi_list = list(hanzi_set)
    pinyin_list = list(pinyin_set)

    hanzi2id = {h: i for i, h in enumerate(hanzi_list)}
    id2hanzi = {i: h for i, h in enumerate(hanzi_list)}
    pinyin2id = {p: i for i, p in enumerate(pinyin_list)}
    id2pinyin = {i: p for i, p in enumerate(pinyin_list)}

    return hanzi2id, id2hanzi, pinyin2id, id2pinyin

# 3. 模型训练
def train_hmm(sentences, pinyins, hanzi2id, pinyin2id):
    n_states = len(hanzi2id)
    n_observations = len(pinyin2id)

    model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=1e-4)

    # 统计初始状态概率、转移概率和发射概率
    start_prob = np.zeros(n_states)
    trans_prob = np.zeros((n_states, n_states))
    emit_prob = np.zeros((n_states, n_observations))

    for sentence, pinyin in zip(sentences, pinyins):
        # print(sentence, pinyin)
        hanzi_seq = [hanzi2id[h] for h in segment_sentence(sentence)]
        pinyin_seq = [pinyin2id[p] for p in pinyin.split()]

        # 初始状态概率
        if len(hanzi_seq) == 0:
            continue
        start_prob[hanzi_seq[0]] += 1

        # 转移概率
        for i in range(len(hanzi_seq) - 1):
            trans_prob[hanzi_seq[i], hanzi_seq[i + 1]] += 1

        # 发射概率
        for h, p in zip(hanzi_seq, pinyin_seq):
            emit_prob[h, p] += 1

    # 确保矩阵行和为1，并处理全零行
    if start_prob.sum() == 0:
        start_prob += 1
    start_prob /= start_prob.sum()

    row_sums = trans_prob.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()  # 修复索引错误
    trans_prob[zero_rows, :] = 1.0 / n_states  # 用均匀分布填充全零行
    trans_prob /= trans_prob.sum(axis=1, keepdims=True)

    emit_sums = emit_prob.sum(axis=1, keepdims=True)
    zero_emit_rows = (emit_sums == 0).flatten()
    emit_prob[zero_emit_rows, :] = 1.0 / n_observations  # 均匀填充
    emit_prob /= emit_prob.sum(axis=1, keepdims=True)
    
    model.startprob_ = start_prob
    model.transmat_ = trans_prob
    model.emissionprob_ = emit_prob
    return model

# 4. 保存和加载模型
def save_model(model, filepath, mode='compress'):  # mode='normal'意味着不使用压缩
    if mode == 'normal':
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    else:
        with bz2.BZ2File(filepath, 'wb') as f:
            pickle.dump(model, f)

def load_model(filepath, mode='compress'):   # mode='normal'意味着不使用压缩
    if mode == 'normal':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        with bz2.BZ2File(filepath, 'rb') as f:
            return pickle.load(f)

    
def train(dataset_path='train.csv', model_path='hmm_model.pkl.bz2'):
    sentences, pinyins = load_dataset(dataset_path)  # 加载数据集
    hanzi2id, id2hanzi, pinyin2id, id2pinyin = build_vocab(sentences, pinyins)  # 构建字典
    model = train_hmm(sentences, pinyins, hanzi2id, pinyin2id)  # 训练模型
    model.pinyin2id = pinyin2id
    model.id2hanzi = id2hanzi
    model.hanzi2id = hanzi2id
    model.id2pinyin = id2pinyin
    save_model(model, model_path)  # 保存模型
    
    
def pred(model_path='hmm_model.pkl.bz2', pinyin_str='ce4 shi4', n_trials=3):
    model = load_model(model_path)
    pinyin_list = pinyin_str.split()
    pinyin2id, id2hanzi = model.pinyin2id, model.id2hanzi
    obs_seq = np.zeros((len(pinyin_list), len(pinyin2id)))  # 转换观测序列为 one-hot 格式
    for t, p in enumerate(pinyin_list):
        if p in pinyin2id:
            obs_seq[t, pinyin2id[p]] = 1
        else:
            obs_seq[t, 0] = 1  # 未知拼音默认处理

    # 解码预测
    model.n_trials = n_trials
    log_prob, state_seq = model.decode(obs_seq, algorithm=model.algorithm)
    result = ''.join([id2hanzi[s] for s in state_seq])
    print('预测结果：', result)

if __name__ == '__main__':
    # train(dataset_path='train.csv', model_path='hmm_model_large.pkl.bz2')
    pred(model_path='hmm_model_large.pkl.bz2', pinyin_str='hong2 yan2 bo2 ming4')  # 预测结果：红颜薄命
