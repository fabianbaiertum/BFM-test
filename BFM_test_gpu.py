import cupy as cp
from scipy.stats import norm
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
path_y="C:\\Users\\Fabia\\Downloads\\y1.csv"
path_x="C:\\Users\\Fabia\\Downloads\\x.csv"
path_y2="C:\\Users\\Fabia\\Downloads\\y2.csv"

def kern_k(x, y=None, bw=1.0, amp=1.0):
    x = cp.asarray(x, dtype=cp.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = x if y is None else cp.asarray(y, dtype=cp.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    x_norm = cp.sum(x ** 2, axis=1).reshape(-1, 1)
    y_norm = cp.sum(y ** 2, axis=1).reshape(1, -1)
    dists_squared = x_norm + y_norm - 2 * cp.dot(x, y.T)
    k = amp * cp.exp(-(1 / (2 * bw * bw)) * dists_squared)
    return k


def MMD_stand(X, Y, bw, unbiased=True):
    Kxx = kern_k(X, X, bw=bw)
    Kyy = kern_k(Y, Y, bw=bw)
    Kxy = kern_k(X, Y, bw=bw)
    n = X.shape[0]
    if unbiased:
        cp.fill_diagonal(Kxx, 0.0)
        cp.fill_diagonal(Kyy, 0.0)
        cp.fill_diagonal(Kxy, 0.0)
        term3 = 2 * cp.sum(Kxy) / (n * (n - 1))
        term1 = cp.sum(Kxx) / (n * (n - 1))
        term2 = cp.sum(Kyy) / (n * (n - 1))
    else:
        term3 = 2 * cp.sum(Kxy) / (n * n)
        term1 = cp.sum(Kxx) / (n * n)
        term2 = cp.sum(Kyy) / (n * n)
    return term1 + term2 - term3


def u_stat(X, Y, bw):
    n = X.shape[0]
    m = n // 2
    X1, X2 = X[::2], X[1::2]
    Y1, Y2 = Y[::2], Y[1::2]
    def rbf(A, B):
        A_norm = cp.sum(A ** 2, axis=1).reshape(-1, 1)
        B_norm = cp.sum(B ** 2, axis=1).reshape(1, -1)
        dists_squared = A_norm + B_norm - 2 * cp.dot(A, B.T)
        return cp.exp(-dists_squared / (2 * bw ** 2))
    K_x1x1 = rbf(X1, X1)
    K_y1y1 = rbf(Y1, Y1)
    K_y2x2 = rbf(Y2, X2)
    H = K_x1x1 + K_y1y1 - K_y2x2 - K_y2x2.T
    upper_sum = cp.sum(cp.triu(H, k=1))
    return 2 * upper_sum / (m * (m - 1))


def stand_h_vec(X, Y, bw):
    return kern_k(X, X, bw=bw) + kern_k(Y, Y, bw=bw) - kern_k(X, Y, bw=bw) - kern_k(X, Y, bw=bw).T


def sigma_spec_a(d, n, X, Y, bw, mmd_hat):
    H = stand_h_vec(X, Y, bw=bw)
    h1 = cp.mean(H, axis=1) - mmd_hat
    s_a_squared = 4.0 * cp.var(h1, ddof=1)
    return {"s.a": cp.sqrt(s_a_squared), "h1.a": h1}


def new_h_vec(Z, bw):
    d = Z.shape[1] // 4
    x1 = Z[:, :d]
    y1 = Z[:, d:2*d]
    x2 = Z[:, 2*d:3*d]
    y2 = Z[:, 3*d:]
    return kern_k(x1, x1, bw=bw) + kern_k(y1, y1, bw=bw) - kern_k(x2, y2, bw=bw) - kern_k(x2, y2, bw=bw).T


def sigma_spec_q(d, n, X, Y, bw, mmd_hat):
    idx = cp.arange(0, n, 2)
    Z = cp.hstack([X[idx], Y[idx], X[idx+1], Y[idx+1]])
    H = new_h_vec(Z, bw=bw)
    h1 = cp.mean(H, axis=1) - mmd_hat
    s_q_squared = 8.0 * cp.var(h1, ddof=1)
    return {"s.q": cp.sqrt(s_q_squared), "h1.q": h1}


def BFM_test(d, n, epsilon, X_np, Y_np, Y2_np=None, bw=1.0, model_selection=False, level=0.05):
    def _run_test(eps):
        X = cp.asarray(X_np, dtype=cp.float32)
        Y = cp.asarray(Y_np, dtype=cp.float32)
        Y2 = cp.asarray(Y2_np, dtype=cp.float32) if model_selection else None

        if not model_selection:
            MMD_hat = MMD_stand(X, Y, bw, unbiased=True)
            MMD_q = u_stat(X, Y, bw)
            MMD_eps = MMD_hat + eps * MMD_q
            out_a = sigma_spec_a(d, n, X, Y, bw, MMD_hat)
            out_q = sigma_spec_q(d, n, X, Y, bw, MMD_hat)
            s_a = out_a["s.a"]
            s_q = out_q["s.q"]
            sd_mmd_eps = s_a + eps * s_q

            return {
                "spec.test.stat.1": float(cp.sqrt(n) * MMD_eps / sd_mmd_eps),
                "spec.reject.1": float(cp.sqrt(n) * MMD_eps / sd_mmd_eps) > norm.ppf(1 - level / 2),
                "MMD.eps.1": float(MMD_eps),
                "MMD.hat.1": float(MMD_hat),
                "MMD.q.1": float(MMD_q),
                "spec.test.stat.q.1": float(cp.sqrt(n) * MMD_q / s_q),
                "spec.reject.q.1": float(cp.sqrt(n) * MMD_q / s_q) > norm.ppf(1 - level / 2),
                "s.a.1": float(s_a),
                "s.q.1": float(s_q),
                "selec.test.stat": None,
                "selec.reject": None,
                "selec.test.stat.q": None,
                "selec.reject.q": None,
                "sd.mmd.sel": None,
                "sd.mmd.sel.q": None,
                "spec.test.stat.2": None,
                "spec.reject.2": None,
                "MMD.eps.2": None,
                "MMD.hat.2": None,
                "MMD.q.2": None,
                "spec.test.stat.q.2": None,
                "spec.reject.q.2": None,
                "s.a.2": None,
                "s.q.2": None,
            }

        # model selection
        MMD_hat_1 = MMD_stand(X, Y, bw, unbiased=True)
        MMD_q_1 = u_stat(X, Y, bw)
        MMD_eps_1 = MMD_hat_1 + eps * MMD_q_1
        out_a1 = sigma_spec_a(d, n, X, Y, bw, MMD_hat_1)
        out_q1 = sigma_spec_q(d, n, X, Y, bw, MMD_hat_1)

        MMD_hat_2 = MMD_stand(X, Y2, bw, unbiased=True)
        MMD_q_2 = u_stat(X, Y2, bw)
        MMD_eps_2 = MMD_hat_2 + eps * MMD_q_2
        out_a2 = sigma_spec_a(d, n, X, Y2, bw, MMD_hat_2)
        out_q2 = sigma_spec_q(d, n, X, Y2, bw, MMD_hat_2)

        s_a_squared = 4 * cp.var(out_a1["h1.a"] - out_a2["h1.a"], ddof=1)
        s_q_squared = 8 * cp.var(out_q1["h1.q"] - out_q2["h1.q"], ddof=1)

        sd_mmd_sel = cp.sqrt(s_a_squared) + eps * cp.sqrt(s_q_squared)
        selec_test_stat = cp.sqrt(n) * (MMD_eps_1 - MMD_eps_2) / sd_mmd_sel
        selec_test_stat_q = cp.sqrt(n) * (MMD_q_1 - MMD_q_2) / cp.sqrt(s_q_squared)

        return {
            "selec.test.stat": float(selec_test_stat),
            "selec.reject": float(cp.abs(selec_test_stat)) > norm.ppf(1 - level / 2),
            "selec.test.stat.q": float(selec_test_stat_q),
            "selec.reject.q": float(cp.abs(selec_test_stat_q)) > norm.ppf(1 - level / 2),
            "sd.mmd.sel": float(sd_mmd_sel),
            "sd.mmd.sel.q": float(cp.sqrt(s_q_squared)),
            "spec.test.stat.1": float(cp.sqrt(n) * MMD_eps_1 / (out_a1["s.a"] + eps * out_q1["s.q"])),
            "spec.reject.1": float(cp.sqrt(n) * MMD_eps_1 / (out_a1["s.a"] + eps * out_q1["s.q"])) > norm.ppf(1 - level / 2),
            "MMD.eps.1": float(MMD_eps_1),
            "MMD.hat.1": float(MMD_hat_1),
            "MMD.q.1": float(MMD_q_1),
            "spec.test.stat.q.1": float(cp.sqrt(n) * MMD_q_1 / out_q1["s.q"]),
            "spec.reject.q.1": float(cp.sqrt(n) * MMD_q_1 / out_q1["s.q"]) > norm.ppf(1 - level / 2),
            "s.a.1": float(out_a1["s.a"]),
            "s.q.1": float(out_q1["s.q"]),
            "spec.test.stat.2": float(cp.sqrt(n) * MMD_eps_2 / (out_a2["s.a"] + eps * out_q2["s.q"])),
            "spec.reject.2": float(cp.sqrt(n) * MMD_eps_2 / (out_a2["s.a"] + eps * out_q2["s.q"])) > norm.ppf(1 - level / 2),
            "MMD.eps.2": float(MMD_eps_2),
            "MMD.hat.2": float(MMD_hat_2),
            "MMD.q.2": float(MMD_q_2),
            "spec.test.stat.q.2": float(cp.sqrt(n) * MMD_q_2 / out_q2["s.q"]),
            "spec.reject.q.2": float(cp.sqrt(n) * MMD_q_2 / out_q2["s.q"]) > norm.ppf(1 - level / 2),
            "s.a.2": float(out_a2["s.a"]),
            "s.q.2": float(out_q2["s.q"]),
        }

    if np.isscalar(epsilon):
        return _run_test(float(epsilon))
    else:
        return [_run_test(float(eps)) for eps in epsilon]


n,d,theta1 = 50000, 16, 16
bw = np.sqrt(theta1 / 2)
data_x=pd.read_csv(path_x,index_col=0).to_numpy()   #since R adds an additional first column
data_y=pd.read_csv(path_y,index_col=0).to_numpy()
X = data_x[:n,:d]
Y = data_y[:n,:d]
pow = -1 / np.array([2.5, 4.5, 6.5])
epsilon = n ** pow
start_time=time.time()
bfm=BFM_test(d,n,epsilon,X,Y,bw=bw)
end_time=time.time()
elapsed_time = end_time - start_time
print(f"BFM took {elapsed_time:.6f} seconds.")