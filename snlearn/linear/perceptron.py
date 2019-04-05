# -*- coding: utf-8 -*-
# Author: yuanyuanzijin
# Email：yuanyuanzijin@gmail.com
# 地址：https://github.com/yuanyuanzijin/sunny-learn
# 更新时间：2019-4-5

"""感知机模型的Python实现"""

import numpy as np
import time
import random


class Perceptron:
    """感知机模型"""

    def __init__(self, learning_rate=1.0, max_epoch=100000):
        """构造方法

        :param learning_rate: float，默认1.0
        :param max_epoch: int，默认1000
        """
        assert 0 < learning_rate <= 1, "学习率需要介于0，1之间！Learning rate should be in the range of 0 to 1."
        self._learning_rate = learning_rate
        self.n_iter = 0                     # 记录更新次数
        self._max_epoch = max_epoch
        self._stop_iter = self._max_epoch
        self._w = None
        self._b = None
        self._alpha = None
        self._gram_metrix = None

    def fit(self, X, y, method='standard'):
        """训练方法

        :param X: array, shape(sample_num, sample_features)
        :param y: array, shape(sample_num, )
        :param method: {'standard', 'dual'}, 训练方法，前者为原始求解形式，后者为对偶形式
        :return: tuple，返回训练后的w和b
        """
        assert method in ['standard', 'dual'], "method参数应为standard或dual中的一个！Method should be standard or dual."
        X = np.array(X)
        y = np.array(y)
        assert X.shape[0] == y.shape[0], "X和y的第一个维度应该相同（相同的样本数量）！X should have the same size as y on dimension 0."
        self._stop_iter = self._max_epoch       # 停止迭代次数，防止由于数据集线性不可分导致死循环
        self._w = np.zeros(X.shape[1])
        self._b = 0
        self.n_iter = 0

        # 原始形式
        if method == "standard":
            while True:
                for xi, yi in zip(X, y):    # 打包返回xi和yi
                    if yi * (np.dot(self._w, xi) + self._b) <= 0:
                        self._update_step(xi, yi, method)
                        break
                # 如果全部样本都分类正确，则退出
                else:
                    break
        # 对偶形式
        elif method == "dual":
            self._alpha = np.zeros(X.shape[0])
            self._gram_metrix = np.dot(X, X.T)
            assert self._gram_metrix.shape == (X.shape[0], X.shape[0])
            while True:
                for index, (xi, yi) in enumerate(zip(X, y)):
                    if yi * (np.dot(self._alpha * y, self._gram_metrix[index]) + self._b) <= 0:
                        self._update_step(xi, yi, method, alpha_i=index)
                        break
                else:
                    break
            self._w = np.dot(self._alpha * y, X)
        return self._w, self._b

    def predict(self, data):
        """测试方法

        :param data: array, shape(sample_num, sample_features)
        :return: array, shape(sample_num, sample_features)，返回预测结果
        """
        data = np.array(data)
        self._w = np.array(self._w)
        assert data.shape[1] == self._w.shape[0], "预测集和训练集的特征维度不同！\nThe dimension 1 of test and training set are different!"
        y = np.dot(data, self._w.T) + self._b
        y[y >= 0] = 1       # 将结果为正数的输出+1
        y[y < 0] = -1       # 将结果为负数的输出-1
        return y

    def _update_step(self, xi, yi, method, alpha_i=0):
        """更新参数"""
        assert self._stop_iter > 0, "已达到最大迭代次数！The maximum number of iterations has been reached. \n" \
                                    "请调整学习率或最大迭代次数，并检查数据集是否线性可分！Please adjust the learning rate or the maximum number " \
                                    "of iterations and check if the data set is linearly separable."
        # 根据method参数判断学习方式，二者w学习方式不同，b相同
        if method == 'standard':
            self._w += self._learning_rate * yi * xi
        elif method == 'dual':
            self._alpha[alpha_i] += self._learning_rate
        self._b += self._learning_rate * yi
        self.n_iter += 1
        self._stop_iter -= 1


if __name__ == "__main__":
    train_X = [[3, 3], [4, 3], [1, 1]]
    train_y = [1, 1, -1]
    clf = Perceptron(learning_rate=1)
    # 训练
    t1 = time.time()
    w, b = clf.fit(train_X, train_y, method='standard')
    t2 = time.time()
    print("w: {}, b: {}, steps: {}, time:{}".format(w, b, clf.n_iter, t2-t1))
    t1 = time.time()
    w, b = clf.fit(train_X, train_y, method='dual')
    t2 = time.time()
    print("w: {}, b: {}, steps: {}, time:{}".format(w, b, clf.n_iter, t2-t1))
    # 预测新数据
    test_X = [[5, 6], [1, 1]]
    predict = clf.predict(test_X)
    print("预测结果：", predict)
