# -*- coding: utf-8
import time
import numpy as np
import scipy.sparse as sparse


class TimeAwareMF(object):
    def __init__(self, K, Lambda, alpha, beta, T):
        self.K = K
        self.T = T
        self.Lambda = Lambda
        self.alpha = alpha
        self.beta = beta
        self.U = None
        self.L = None

    def save_model(self, path):
        ctime = time.time()
        for i in range(self.T):
            np.save(path + "U" + str(i), self.U[i])
        np.save(path + "L", self.L)
        print("保存时间感知模块-隐矩阵U和L完毕，用时{}s:".format(time.time() - ctime))

    def load_model(self, path):
        ctime = time.time()
        self.U = [np.load(path + "U%d.npy" % i) for i in range(self.T)]
        self.L = np.load(path + "L.npy")
        print("重新加载时间感知模块-隐矩阵：U和L完毕，用时{}s:".format(time.time() - ctime))

    def get_phi(self, C, i, t):
        t_1 = (t - 1) if not t == 0 else (self.T - 1)
        norm_t = np.linalg.norm(C[t][i, :].toarray(), 'fro')
        norm_t_1 = np.linalg.norm(C[t_1][i, :].toarray(), 'fro')
        if norm_t == 0 or norm_t_1 == 0:
            return 0.0
        return C[t][i, :].dot(C[t_1][i, :].T)[0, 0] / norm_t / norm_t_1

    def train(self, sparse_check_in_matrices, temp_dir, max_iters=100, load_sigma=False,find_max_iters=False):
        print("====== DEBUG: TAMF train called, max_iters={} ======".format(max_iters))
        Lambda = self.Lambda
        alpha = self.alpha
        beta = self.beta
        T = self.T
        K = self.K

        C = sparse_check_in_matrices
        M, N = sparse_check_in_matrices[0].shape

        if load_sigma:
            ctime = time.time()
            print("...",)
            sigma = np.load(temp_dir + "sigma.npy")
            print("加载时间感知模块 sigma完毕，用时{}".format(time.time() - ctime))
        else:
            ctime = time.time()
            print("初始化时间感知模型sigma...")
            sigma = [np.zeros(M) for _ in range(T)]
            for t in range(T):
                C[t] = C[t].tocsr()
                for i in range(M):
                    sigma[t][i] = self.get_phi(C, i, t)
            sigma = [sparse.dia_matrix(sigma_t) for sigma_t in sigma]
            print("时间感知模型sigma初始化完毕，用时{}s".format(time.time() - ctime))
            np.save(temp_dir + "/sigma", sigma)

        U = [np.random.rand(M, K) for _ in range(T)]
        L = np.random.rand(N, K)

        C = [Ct.tocoo() for Ct in C]
        """fix:将 zip 转为 list：Python 3 中 zip() 返回的是迭代器（iterator），不是列表。
        第一次遍历后迭代器就被消耗完了，后续所有迭代中 for i, j in entry_index[t] 都是空循环！
        所以 C_est 从未被更新，error 永远是初始的 0.0。"""
        entry_index = [list(zip(C[t].row, C[t].col)) for t in range(T)]

        C_est = [Ct for Ct in C]
        C = [Ct.tocsr() for Ct in C]
        iters, last_error = 1,float('inf')
        while(iters>0):
            for t in range(T):
                C_est[t] = C_est[t].todok()
                for i, j in entry_index[t]:
                    C_est[t][i, j] = U[t][i].dot(L[j])
                C_est[t] = C_est[t].tocsr()

            for t in range(T):
                t_1 = (t - 1) if not t == 0 else (self.T - 1)
                numerator = C[t] * L + Lambda * sigma[t] * U[t_1]
                denominator = np.maximum(1e-6, C_est[t] * L + Lambda * sigma[t] * U[t_1] + alpha * U[t_1])
                U[t] *= np.sqrt(1.0 * numerator / denominator)

            numerator = np.sum([C[t].T * U[t] for t in range(T)], axis=0)
            denominator = np.maximum(1e-6, np.sum([C_est[t].T * U[t]], axis=0) + beta * L)
            L *= np.sqrt(1.0 * numerator / denominator)

            error = 0.0
            for t in range(T):
                C_est_dok = C_est[t].todok()
                C_dok = C[t].todok()
                for i, j in entry_index[t]:
                    error += (C_est_dok[i, j] - C_dok[i, j]) * (C_est_dok[i, j] - C_dok[i, j])
            if find_max_iters and error >= last_error and max_iters < iters  :
                break
            if find_max_iters is False and iters > max_iters:
                break
            print('Iteration:', iters, error)
            if error < last_error:
                self.U, self.L = U, L
                last_error = error
            iters += 1
        """fix-07:注释掉多余循环"""
        for iters in range(max_iters):
            for t in range(T):
                C_est[t] = C_est[t].todok()
                for i, j in entry_index[t]:
                    C_est[t][i, j] = U[t][i].dot(L[j])
                C_est[t] = C_est[t].tocsr()

            for t in range(T):
                t_1 = (t - 1) if not t == 0 else (self.T - 1)
                numerator = C[t] * L + Lambda * sigma[t] * U[t_1]
                denominator = np.maximum(1e-6, C_est[t] * L + Lambda * sigma[t] * U[t_1] + alpha * U[t_1])
                U[t] *= np.sqrt(1.0 * numerator / denominator)

            numerator = np.sum([C[t].T * U[t] for t in range(T)], axis=0)
            denominator = np.maximum(1e-6, np.sum([C_est[t].T * U[t]], axis=0) + beta * L)
            L *= np.sqrt(1.0 * numerator / denominator)

            error = 0.0
            for t in range(T):
                C_est_dok = C_est[t].todok()
                C_dok = C[t].todok()
                for i, j in entry_index[t]:
                    error += (C_est_dok[i, j] - C_dok[i, j]) * (C_est_dok[i, j] - C_dok[i, j])
            print('Iteration:', iters, error)
        self.U, self.L = U, L

    def predict(self, i, j):
        return np.sum([self.U[t][i].dot(self.L[j]) for t in range(self.T)])
