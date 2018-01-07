#! /data/sever/python/bin/python
# -*- coding:utf-8 -*-
"""
决策树ID3算法
"""
from __future__ import division
import math
import operator
from collections import Counter

__author__ = 'xyz'


def calc_shannon_ent(data_set):
    """
    计算香农熵
    :param data_set:
    :return:
    """
    data_length = len(data_set)
    label_counts = Counter([val[-1] for val in data_set])
    pilog2pi = [val / data_length * math.log(val / data_length, 2) for val in label_counts.itervalues()]
    return - reduce(
        operator.add,
        pilog2pi
    ) if pilog2pi else 0


def split_data_set(data_set, axis, value):
    """
    分割数据集，筛选指定特征下的数据值的集合
    :param data_set: 数据集合
    :param axis: 第几列
    :param value: 筛选的值
    :return: 除去axis列的，并且axis列的值为value的的数据集合
    """
    return [[v for i, v in enumerate(val) if i != axis] for val in data_set if val[axis] == value]


def choose_best_feature_to_split(data_set):
    """
    选择最好的数据集划分方式
    :param data_set: 数据集
    :return: 划分方式最好是第几项
    """
    base_ent = calc_shannon_ent(data_set)
    # 定义最好的信息增益，信息增益最好的那项
    best_info_gain, best_feature = 0.0, -1
    for i in range(len(data_set[0]) - 1):
        unique_value = set(data_set[i])
        child_ent = 0.0
        for val in unique_value:
            child_data_set = split_data_set(data_set, i, val)
            child_ent += (len(data_set) - 1) / len(data_set) * calc_shannon_ent(child_data_set)
        # 信息增益
        info_gain = base_ent - child_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_ent(class_list):
    """
    取出出现次数最多的标签
    :param class_list:
    :return:
    """
    class_count = Counter(class_list)
    sorted_class_count = sorted(class_count.items(), key=lambda x, y: cmp(x[1], y[1]), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    创建树
    :param data_set: 数据集
    :param labels: 标签集合
    :return: 决策树
    """
    class_list = [val[-1] for val in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_ent(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del labels[best_feat]
    feat_values = [val[best_feat] for val in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree

if __name__ == "__main__":
    data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    # 计算熵
    print calc_shannon_ent(data_set)
    # 分割数据集
    print split_data_set(data_set, 0, 1)
    # 获取最大信息增益项
    print choose_best_feature_to_split(data_set)
    # 生成决策树
    print create_tree(data_set, ['no surfacing', 'flippers'])
