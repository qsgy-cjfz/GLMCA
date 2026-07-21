# -*- coding: utf-8
"""
@author：67543
@date：  2022/4/18
@contact：675435108@qq.com
"""
import pickle
import json
import os
from collections import defaultdict

from util import *
from tools import *

import multiprocessing as mp

"""fix-09: 添加多进程支持，加速预测阶段,把原来 predict() 中的单用户计算逻辑提取到 cal_unite_metrics() 函数"""
import multiprocessing as mp

"""分数归一化 + 调高 gnn_w
对每个用户，将 FUC 和 GCN 的原始分数转换为排名百分位后再融合，消除尺度差异
"""
def cal_unite_metrics(params):
    """多进程辅助函数：计算单个用户的联合预测指标"""
    cnt, uid = params
    metrics = "pre@5={},rec@5={},ndcg@5={},pre@10={},rec@10={},ndcg@10={}," \
              "pre@15={},rec@15={},ndcg@15={},pre@20={},rec@20={},ndcg@20={}"
    FUC_score = [PFM.predict(uid, lid) *
                 ((MGMWK['day'].predict(uid, lid) + MGMWK['night'].predict(uid, lid)) +
                  (MGMED['day'].predict(uid, lid) + MGMED['night'].predict(uid, lid)))
                 * TAMF.predict(uid, lid) * LFBCA.predict(uid, lid)
                 if (uid, lid) not in training_tuples else -1
                 for lid in all_lids]
    FUC_score = np.array(FUC_score)
    GCN_score = scores[uid].copy()

    fuc_rank = np.zeros(len(all_lids))
    gcn_rank = np.zeros(len(all_lids))
    valid_fuc = FUC_score > -1
    valid_gcn = GCN_score > -np.inf

    if valid_fuc.sum() > 0:
        order = FUC_score.argsort().argsort()
        fuc_rank = order.astype(float) / len(all_lids)
    if valid_gcn.sum() > 0:
        order = GCN_score.argsort().argsort()
        gcn_rank = order.astype(float) / len(all_lids)

    unite_score = (1 - args.gnn_w) * fuc_rank + args.gnn_w * gcn_rank
    for idx in range(len(all_lids)):
        if not valid_fuc[idx] or not valid_gcn[idx]:
            unite_score[idx] = -1

    predict_topk = list(reversed(unite_score.argsort()))[:top_k]
    actual = ground_truth[uid]
    result = get_metrics(actual, predict_topk)
    metrics_str = metrics.format(*result)
    print(cnt, uid, metrics_str)
    return result.tolist()

# def cal_unite_metrics(params):
#     """多进程辅助函数：计算单个用户的联合预测指标"""
#     cnt, uid = params
#     metrics = "pre@5={},rec@5={},ndcg@5={},pre@10={},rec@10={},ndcg@10={}," \
#               "pre@15={},rec@15={},ndcg@15={},pre@20={},rec@20={},ndcg@20={}"
#     FUC_score = [PFM.predict(uid, lid) *
#                  ((MGMWK['day'].predict(uid, lid) + MGMWK['night'].predict(uid, lid)) +
#                   (MGMED['day'].predict(uid, lid) + MGMED['night'].predict(uid, lid)))
#                  * TAMF.predict(uid, lid) * LFBCA.predict(uid, lid)
#                  if (uid, lid) not in training_tuples else -1
#                  for lid in all_lids]
#     FUC_score, GCN_score = np.array(FUC_score), scores[uid]
#     unite_score = (1 - args.gnn_w) * FUC_score + args.gnn_w * GCN_score
#     predict_topk = list(reversed(unite_score.argsort()))[:top_k]
#     actual = ground_truth[uid]
#     result = get_metrics(actual, predict_topk)
#     metrics_str = metrics.format(*result)
#     print(cnt, uid, metrics_str)
#     return result.tolist()
"""fix: 添加向量化计算，加速预测阶段"""
# def cal_unite_metrics(params):
#     """多进程辅助函数：计算单个用户的联合预测指标（向量化版本）"""
#     cnt, uid = params
#     metrics = "pre@5={},rec@5={},ndcg@5={},pre@10={},rec@10={},ndcg@10={}," \
#               "pre@15={},rec@15={},ndcg@15={},pre@20={},rec@20={},ndcg@20={}"
#
#     pfm_all = PFM.U[uid].dot(PFM.L.T)
#     tamf_all = np.zeros(len(all_lids))
#     for t in range(TAMF.T):
#         tamf_all += TAMF.U[t][uid].dot(TAMF.L.T)
#     lfbc_all = LFBCA.rec_score[uid]
#
#     FUC_score = np.empty(len(all_lids))
#     for idx, lid in enumerate(all_lids):
#         if (uid, lid) in training_tuples:
#             FUC_score[idx] = -1
#         else:
#             mgm_val = ((MGMWK['day'].predict(uid, lid) + MGMWK['night'].predict(uid, lid)) +
#                        (MGMED['day'].predict(uid, lid) + MGMED['night'].predict(uid, lid)))
#             FUC_score[idx] = pfm_all[lid] * mgm_val * tamf_all[lid] * lfbc_all[lid]
#
#     GCN_score = scores[uid]
#     unite_score = (1 - args.gnn_w) * FUC_score + args.gnn_w * GCN_score
#     predict_topk = list(reversed(unite_score.argsort()))[:top_k]
#     actual = ground_truth[uid]
#     result = get_metrics(actual, predict_topk)
#     metrics_str = metrics.format(*result)
#     print(cnt, uid, metrics_str)
#     return result.tolist()

def predict():
    # results = []
    """fix-10: 用 mp.Pool(processes=args.cores) 多进程并行，args.cores 默认 4，可通过 --cores 8 调整.与 FUC.py 的多进程方式完全一致"""
    test_paras = [(i, uid) for i, uid in enumerate(test_uids)]
    pool = mp.Pool(processes=args.cores)
    results = pool.map(cal_unite_metrics, test_paras)
    pool.close()
    pool.join()

    results = np.array(results)
    final_gnn_resuluts = np.mean(results, axis=0)
    metrics = "pre@5={},rec@5={},ndcg@5={},pre@10={},rec@10={},ndcg@10={}," \
              "pre@15={},rec@15={},ndcg@15={},pre@20={},rec@20={},ndcg@20={}".format(*final_gnn_resuluts)

    with open(result_path + "/uniteModels_result.txt", 'w') as f:
        for result in results:
            str_reslut = '\t'.join(str(i) for i in result)
            f.write(str_reslut + '\n')
        f.write(metrics + '\n')
    print(metrics)

if __name__ == '__main__':
    """
      1.基本配置
   """
    args = parse_args()
    temp_path, result_path = get_config(args, "uniteModel")
    data_name = args.dataset
    FUC_load_path = "./uniteModels/" + data_name + "-FUC/"
    GCN_load_path = "./uniteModels/" + data_name + "-GCN/"
    """
   2.加载测试数据
   """
    """fix-02: 'r' -> 'rb'，Python 3 中 pickle 必须以二进制模式读取"""
    with open("processed/" + data_name + "/graph.pkl", 'rb') as f:
        train_set = pickle.load(f)
        train_U2I = pickle.load(f)
        test_U2I = pickle.load(f)
        training_matrix2 = pickle.load(f)
        social_matrix = pickle.load(f)
        user_num, poi_num = pickle.load(f)

    """fix-03: 'r' -> 'rb'，同上"""
    with open("processed/" + data_name + "/fuc.pkl", 'rb') as f:
        """fix-04: data.py 中 sparse_training_matrix_WT 和 sparse_training_matrix_LT 已注释掉不再保存，
        原代码加载 WT/LT 导致后续所有对象错位。现按 data.py 保存顺序逐个加载，补上 sparse_training_matrices，
        并改为加载 training_data_workday/training_data_weekend 字典"""
        sparse_training_matrix = pickle.load(f)
        training_tuples = pickle.load(f)
        training_tuples = set(training_tuples)
        training_matrix = pickle.load(f)
        training_matrix2 = pickle.load(f)
        sparse_training_matrices = pickle.load(f)
        sparse_training_matrix_WT = pickle.load(f)
        sparse_training_matrix_LT = pickle.load(f)
        training_data_workday = pickle.load(f)
        training_data_weekend = pickle.load(f)
        ground_truth = pickle.load(f)
        poi_coos = pickle.load(f)

    """fix-05: test_U2I.keys() -> list(test_U2I.keys())，Python 3 中 dict_keys 视图对象不可被 shuffle"""
    test_uids = list(test_U2I.keys())
    all_uids, all_lids = list(range(user_num)), list(range(poi_num))
    np.random.shuffle(test_uids)
    """
   3.初始化模块
   """
    PFM = PoissonFactorModel(K=30, alpha=20.0, beta=0.2)
    """fix-06: MGMWT/MGMLT 单矩阵已废弃，改为 MGMWK/MGMED 字典结构（day/night），与 FUC.py 一致"""
    MGMWK = {'day': None, 'night': None}
    MGMED = {'day': None, 'night': None}
    for key in MGMWK.keys():
        MGMWK[key] = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=args.wk_dmax)
        MGMED[key] = MultiGaussianModel(alpha=0.2, theta=0.02, dmax=args.ed_dmax)
    TAMF = TimeAwareMF(K=100, Lambda=1.0, beta=2.0, alpha=2.0, T=24)
    LFBCA = LocationFriendshipBookmarkColoringAlgorithm(alpha=0.85, beta=float(args.beta), epsilon=0.001)
    """
   4.模块参数装配
   """
    """fix-07: PyTorch 2.6+ 默认 weights_only=True，加载含 numpy 的旧模型需显式设为 False"""
    user_embed, item_embed = torch.load(GCN_load_path + data_name + "-graph_model.pth", weights_only=False)
    scores = np.matmul(user_embed, item_embed.T)
    PFM.load_model(FUC_load_path)
    """fix-08: MGMWT/MGMLT 改为 MGMWK/MGMED，使用 training_data_workday/weekend 字典"""
    """for key in MGMWK.keys():
        MGMWK[key].multi_center_discovering(training_data_workday[key], poi_coos)
        MGMED[key].multi_center_discovering(training_data_weekend[key], poi_coos)"""

    """add:LLM"""
    llm_coord_file = f"processed/activity_centers/{data_name}_llm_coordinate_centers.json"
    if os.path.exists(llm_coord_file):
        with open(llm_coord_file, 'r', encoding='utf-8') as f:
            llm_coord_data = json.load(f)
        format_print(f"使用LLM活动中心坐标 ({llm_coord_data['total_users']}个用户)")
        for key in MGMWK.keys():
            MGMWK[key].load_centers_from_llm(training_data_workday[key], poi_coos, llm_coord_data["user_centers"])
            MGMED[key].load_centers_from_llm(training_data_weekend[key], poi_coos, llm_coord_data["user_centers"])
    else:
        format_print("未找到LLM坐标文件，使用原始距离聚类")
        for key in MGMWK.keys():
            MGMWK[key].multi_center_discovering(training_data_workday[key], poi_coos)
            MGMED[key].multi_center_discovering(training_data_weekend[key], poi_coos)



    TAMF.load_model(FUC_load_path)
    LFBCA.load_model(FUC_load_path)
    """
      5.模块预测
      """
    predict()
