# -*- coding: utf-8
"""
@author：67543
@date：  2022/4/8
@contact：675435108@qq.com
"""
import math
import numpy as np
import multiprocessing
import torch
import pickle

def recall_k(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    rec = hits/(1.0 * len(ground_list))
    return rec

def ndcg_k(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i+1
        dcg += 1/ math.log(rank+1, 2)
    return dcg / idcg


def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i+2, 2)
    return idcg

def pract_one_perf(X):
    uid,max_topk = X[0],max(topks)
    score = scores[uid] # poi推荐中访问过的，可能会再访问
    uid_train_pos_itmes = list(train_user_item[uid])
    uid_test_pos_items = list(test_user_item[uid])
    score[uid_train_pos_itmes] = -np.inf
    indices = np.argpartition(score, -max_topk)[-max_topk:]
    score_indices = indices[np.argsort(-score[indices])] # 找到前topk个概率最大的值
    topks_eval = np.zeros(2 * len(topks), dtype=np.float32)
    for i, topk in enumerate(topks):
        topks_eval[i * 2 + 0] = recall_k(score_indices[:topk], uid_test_pos_items)
        topks_eval[i * 2 + 1] = ndcg_k(score_indices[:topk], uid_test_pos_items)
    return topks_eval


def init(_scores,_train_U2I,_test_U2I,_topks):
    global scores, train_user_item, test_user_item, topks
    scores = _scores
    train_user_item = _train_U2I
    test_user_item = _test_U2I
    topks =_topks


if __name__ == '__main__':
    with open("processed/Yelp_train.pkl", 'r') as f:
        train_set = pickle.load(f)
        train_U2I = pickle.load(f)
    with open("processed/Yelp_test.pkl", 'r') as f:
        test_U2I = pickle.load(f)
    user_embed, item_embed = torch.load("Dataset/run/Yelp-2022-04-08_14:12graph_model.pth")
    scores = np.matmul(user_embed, item_embed.T)
    topks = [10, 20]
    test_user_set = list(test_U2I.keys())
    perf_info = np.zeros((len(test_user_set), 2 * len(topks)), dtype=np.float32)
    test_paras = zip(test_user_set, )  # list=[(0,),(1,),...,(n_users,)]
    pool = multiprocessing.Pool(processes=4, initializer=init, initargs=(scores, train_U2I, test_U2I, topks))
    all_perf = pool.map(pract_one_perf, test_paras)
    pool.close()
    pool.join()
    for i, one_perf in enumerate(all_perf):
        perf_info[i] = one_perf
    perf_info = np.mean(perf_info, axis=0)


