# -*- coding: utf-8 -*-
"""
 author: lvsongke@oneniceapp.com
 data:2019/09/11
"""
import numpy as np
import random


def stocGrandAscent(dataMatrix, classLabels, k, max_iter, alpha):
    """
    利用随机梯度下降法训练fm模型
    :param dataMatrix: 数据矩阵
    :param classLabels: 标签
    :param k: v的维数
    :param max_iter:最大迭代次数
    :param alpha: 学习率
    :return: w0, w,v
    """
    m, n = np.shape(dataMatrix)
    # 1.初始化参数
    # w = np.zeros((n, 1))  # n是维度
    w0 = 0  # 偏执
    w, v = initialize_v(n, k)  # 初始化w, V

    # 训练
    for it in range(max_iter):
        for x in range(m):  # 随机优化,对每一个样本
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)

            # 完成交叉项
            interaction = np.sum(np.multiply(inter_1, inter_2) - inter_2) / 2.
            p = w0 + dataMatrix[x] * w + interaction

            loss = sigmoid(classLabels[x] * p[0, 0]) - 1
            w0 = w0 - alpha * loss * classLabels[x]

            # if loss <= 0.001:
                # print('loss:'.format(loss))
                # break

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]

                    for f in range(k):
                        v[i, f] = v[i, f] - alpha * loss * classLabels[x] * (
                                dataMatrix[x, i] * inter_1[0, f] - v[i, f] * dataMatrix[x, i] * dataMatrix[x, i])   ##梯度下降
        if it % 1000 == 0:
            print('\t---------iter:{it}, cost:{cost}'.format(it=it,
                                                             cost=getCost(getPrediction(np.mat(dataMatrix), w0, w, v),
                                                                          classLabels)))

    return w0, w, v


def initialize_v(n, k):
    """
    初始化交叉项
    :param n: 维度
    :param k: 因子分解后的维度
    :return: 交叉项的系数权重V
    """
    w = np.zeros((n, 1))
    v = np.mat(np.zeros((n, k)))
    for i in range(n):
        for j in range(k):
            v[i, j] = random.gauss(0, 0.2)
    return w, v


def sigmoid(wx):
    return 1 / 1 + np.exp(-wx)


def getPrediction(dataMatrix, w0, w, v):
    m = np.shape(dataMatrix)[0]
    result = []
    for x in range(m):
        inter_1 = dataMatrix[x] * v
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
        interaction = 0.5 * np.sum(np.multiply(inter_1, inter_1) - inter_2)
        p = w0 + dataMatrix[x] * w + interaction
        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result
    pass


def getCost(pred, classLabels):
    m = np.shape(pred)[0]
    cost = []
    error = 0.
    for i in range(m):
        error -= np.log(sigmoid(pred[i] * classLabels[i]))
        cost.append(error)
    return error


def getAccuracy(pred, classLabels):
    m = np.shape(pred)[0]
    allItem = 0
    error = 0
    for i in range(m):
        allItem += 1
        if float(pred[i]) < 0.5 and classLabels[i] == 1.0:
            error += 1
        elif float(pred[i]) >= 0.5 and classLabels[i] == -1.0:
            error += 1
        else:
            continue
    return float(error) / allItem


def load_data(filename):
    data = open(filename)
    feature = []
    label = []
    for line in data.readlines():
        feature_tmp = []
        lines = line.strip().split('\t')
        for x in range(len(lines) - 1):
            feature_tmp.append(float(lines[x]))
        label.append(int(lines[-1]) * 2 - 1)
        feature.append(feature_tmp)
    data.close()
    return feature, label


if __name__ == '__main__':
    feature, label = load_data('train_fm.txt')
    w0, w, v = stocGrandAscent(np.mat(feature), label, 4, 20000, 0.01)
    predict_result = getPrediction(np.mat(feature), w0, w, v)
    print('训练精度为：'.format(1 - getAccuracy(predict_result, label)))
