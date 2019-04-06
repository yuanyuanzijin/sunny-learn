# -*- coding: utf-8 -*-
# Author: yuanyuanzijin
# Email：yuanyuanzijin@gmail.com
# 地址：https://github.com/yuanyuanzijin/sunny-learn
# 更新时间：2019-4-5

"""决策树模型的Python实现"""

import numpy as np
import time

class TreeNode:
    def __init__(self):
        self.childlist = None
        self.feature = None
        self.data = None
        self.isleaf = None
        self.cls= None

class DicisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, X, y, val_X=None, val_y=None, divide_method='gini', cut_method='None'):
        assert divide_method in ['gini'], "cut_method参数应为gini中的一个！Method should be gini."
        self.X = np.array(X)
        self.y = np.array(y)
        self.val_X = np.array(val_X)
        self.val_y = np.array(val_y)
        assert self.X.shape[0] == self.y.shape[0], "X和y的第一个维度应该相同（相同的样本数量）！X should have the same size as y on dimension 0."
        if self.val_X and self.val_y:
            assert self.val_X.shape[0] == self.val_y.shape[0] == self.X.shape[0], "X和y的第一个维度应该相同（相同的样本数量）！X should have the same size as y on dimension 0."
        data_num = self.X.shape[0]

        data_indices = list(range(data_num))
        feature_indices = list(range(self.X.shape[1]))
        # 计算D的基尼指数
        # gini_D = self._cal_gini(self.y)
        self.tree = self._build_tree(data_indices, feature_indices)
        return

    def _build_tree(self, data_indices, feature_indices):
        node = TreeNode()
        node.data_indices = data_indices
        data_X = self.X[data_indices]
        data_y = self.y[data_indices]
        # 如果样本都是同一类别
        if len(set(data_y)) == 1:
            node.isleaf = True
            node.cls = data_y[0]
            return node
        # 如果属性集合用完了
        if not feature_indices:
            node.isleaf = True
            node.cls = self._class_has_max_num(data_y)
            return node
        
        # 在所有特征中循环，计算每个属性的基尼指数
        best_feature_index = self._choose_feature_index(data_indices, feature_indices)
        print('Best feature index is index %d.' % best_feature_index)
        best_feature_value = data_X[:, best_feature_index]
        print('未完待续')


    def _class_has_max_num(self, data_y):
        tmp = {}
        for label in set(data_y):
            tmp[label] = np.sum(data_y == label)
        choose_class = max(tmp.keys(), key=lambda x: tmp[x])
        return choose_class

    def _choose_feature_index(self, data_indices, feature_indices):
        data_X = self.X[data_indices]
        data_y = self.y[data_indices]
        gini_D_a = []
        for f_index in feature_indices:  
            feature_value_list = data_X[:, f_index]
            feature_values = set(feature_value_list)
            gini_D_a_i = 0
            for f_value in feature_values:
                new_data_index = np.where(feature_value_list == f_value)
                new_data_y = data_y[new_data_index]
                gini_D_a_i += new_data_y.shape[0] / data_y.shape[0] * self._cal_gini(new_data_y)
            gini_D_a.append(gini_D_a_i)
        feature_index = gini_D_a.index(min(gini_D_a))
        return feature_index



    def _cal_gini(self, np_y):
        y_values = set(np_y)
        data_num = np_y.shape[0]
        p_list = []
        gini_D = 1
        for y_value in y_values:
            p = np.sum(np_y == y_value) / data_num
            p_list.append(p)
            gini_D -= p ** 2
        return gini_D


if __name__ == "__main__":
    train_X = [[0,0,1,0,0,0],
               [1,0,0,0,0,0],
               [1,0,1,0,0,0],
               [0,1,1,0,1,1],
               [1,1,1,1,1,1],
               [0,2,2,0,2,1],
               [2,1,0,1,0,0],
               [1,1,1,0,1,1],
               [2,0,1,2,2,0],
               [0,0,0,1,1,0]]
    train_y = [1,1,1,1,1,-1,-1,-1,-1,-1]
    val_X = [[0,0,0,0,0,0],
             [2,0,1,0,0,0],
             [1,1,1,0,1,0],
             [1,1,0,1,1,0],
             [2,2,2,2,2,0],
             [2,0,1,2,2,1],
             [0,1,1,1,0,0]]
    val_y = [1,1,1,-1,-1,-1,-1]
    clf = DicisionTree()
    clf.fit(train_X + val_X, train_y + val_y, divide_method='gini', cut_method='None')
