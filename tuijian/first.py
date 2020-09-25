# -*- coding: utf-8 -*-
"""
 author: lvsongke@oneniceapp.com
 data:2019/09/11
"""
import math
import pandas as pd
from operator import itemgetter
from sklearn.model_selection import train_test_split

from tqdm import tqdm


def load_set(filename, group_col, value, size=0.9):
    data = pd.read_csv(filename)
    res_data, _ = train_test_split(data, train_size=size, random_state=14)
    result_group = res_data.groupby([group_col])[value].unique()
    result_group = result_group[:100]
    return result_group


def recall(train, test, N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, train, N, W)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)


def Precision(train, test, N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, train, N, W)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += N
    return hit / (all * 1.0)


def UserSimilarity(train):
    """
    计算用户与用户之间的相似度
    :param train:
    :return:
    """
    item_users = dict()
    ###train={'user': [items]}
    ##建立物品到用户的到排表----建立物品和用户的映射关系
    for u, item in tqdm(train.items(), desc='items to user'):
        for i in item:
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)

    ##计算用户与用户之间的共同拥有的商品

    # item--user 数量映射表
    C = dict()
    # 用户总数
    N = dict()
    for i, users in tqdm(item_users.items(), desc='C[u][v]'):
        for u in users:
            if u not in N:
                N[u] = 0
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                if u not in C:
                    C[u] = {}
                if v not in C[u]:
                    C[u][v] = 0
                ## 增加用户u和用户v共同兴趣列表中热门物品对他们相似度的影响
                C[u][v] += 1 / math.log(1 + len(users))
    # 建立相似度矩阵
    W = dict()
    for u, related_users in tqdm(C.items(), desc='建立相似矩阵'):
        for v, cuv in related_users.items():
            if u not in W:
                W[u] = {}
            W[u][v] = cuv / math.sqrt(N[u] * N[v])

    return W


def Recommend(user, train, W):
    rank = dict()
    ##获取数据集中这个用户的喜好列表
    interacted_items = train[user]
    print('获取rank')
    for v, wuv in sorted(W[user].items(), key=itemgetter(1), reverse=True)[0: K]:
        for v_item in train[v]:
            ##如果用户v和用户u共有的商品则不再推荐
            if v_item in interacted_items:
                continue
            if v_item not in rank:
                rank[v_item] = 0
            ###推荐的商品和相似度映射表
            rank[v_item] += wuv * 1.0
    return rank


def GetRecommendation(user, train, N, W):
    """
    一个用户得到的建议
    :param user:
    :param N:
    :return:
    """
    rank = Recommend(user, train, W)
    print('获取过滤结果')
    rank_N = dict(sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N])
    return rank_N


if __name__ == '__main__':
    filename = '../../data_sets/ml-25m/ratings.csv'
    K = 10
    N = 10
    data = load_set(filename, 'userId', 'movieId')
    train, test = train_test_split(data, test_size=0.3, random_state=14)
    train = train.to_dict()
    test = test.to_dict()
    W = UserSimilarity(train)
    print(W)
    rank = GetRecommendation(2, train, N, W)
    print(rank)

    # print('召回率:\n')
    # print(recall(train, test, N))