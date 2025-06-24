import cupy as cp
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def sim_mu_2sampl(d: int, n: int, SD_theor1=(1.0, 1.0), SD_theor2=(1.0, 1.0), rng: np.random.Generator | None = None):
    if d % 2 != 0:
        raise ValueError("d must be even.")
    if rng is None:
        rng = np.random.default_rng()
    X = rng.standard_normal(size=(n, d))
    dd = d // 2
    hat_mu_X = X.mean(axis=0)
    mu_mat = hat_mu_X
    Y1_block1 = rng.normal(0.0, SD_theor1[0], size=(n, dd))
    Y1_block2 = rng.normal(0.0, SD_theor1[1], size=(n, dd))
    Y_first = np.hstack([Y1_block1, Y1_block2])
    hat_Y1 = Y_first + mu_mat
    Y2_block1 = rng.normal(0.0, SD_theor2[0], size=(n, dd))
    Y2_block2 = rng.normal(0.0, SD_theor2[1], size=(n, dd))
    Y_second = np.hstack([Y2_block1, Y2_block2])
    hat_Y2 = Y_second + mu_mat
    return {"X": X, "Y": Y_second, "hat_Y1": hat_Y1, "hat_Y2": hat_Y2, "hat_mu_X": mu_mat}

def kern_k(x, y=None, bw=1.0):
    #Distance computations (cupyx.scipy.spatial.distance)  could use cdist of cupy !
    x = cp.asarray(x, dtype=cp.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = x if y is None else cp.asarray(y, dtype=cp.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    x_norm = cp.sum(x ** 2, axis=1).reshape(-1, 1)
    y_norm = cp.sum(y ** 2, axis=1).reshape(1, -1)
    dists_squared = x_norm + y_norm - 2 * cp.dot(x, y.T)
    return cp.exp(-(1 / (2 * bw * bw)) * dists_squared)

def MMD_stand(X, Y, bw, unbiased=True):
    Kxx = kern_k(X, X, bw)
    Kyy = kern_k(Y, Y, bw)
    Kxy = kern_k(X, Y, bw)
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
    n, d = X.shape
    if n % 2 != 0:
        raise ValueError("n must be even.")
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
    H = stand_h_vec(X, Y, bw)
    h1 = cp.mean(H, axis=1) - mmd_hat
    s_a_squared = 4.0 * cp.var(h1, ddof=1)
    return {"s.a": cp.sqrt(s_a_squared), "h1.a": h1}

def new_h_vec(Z, bw):
    d = Z.shape[1] // 4
    x1, y1 = Z[:, :d], Z[:, d:2*d]
    x2, y2 = Z[:, 2*d:3*d], Z[:, 3*d:]
    return kern_k(x1, x1, bw=bw) + kern_k(y1, y1, bw=bw) - kern_k(x2, y2, bw=bw) - kern_k(x2, y2, bw=bw).T

def sigma_spec_q(d, n, X, Y, bw, mmd_hat):
    nn = n // 2
    idx = cp.arange(0, n, 2)
    Z = cp.hstack([X[idx], Y[idx], X[idx + 1], Y[idx + 1]])
    H = new_h_vec(Z, bw=bw)
    h1 = cp.mean(H, axis=1) - mmd_hat
    s_q_squared = 8.0 * cp.var(h1, ddof=1)
    return {"s.q": cp.sqrt(s_q_squared), "h1.q": h1}

def BFM_test(d, n, epsilon, X_np, Y_np, Y2_np=None, bw=1.0, model_selection=False, level=0.05):
    X = cp.asarray(X_np, dtype=cp.float32)
    Y = cp.asarray(Y_np, dtype=cp.float32)
    Y2 = cp.asarray(Y2_np, dtype=cp.float32) if model_selection else None

    if not model_selection:
        MMD_hat = MMD_stand(X, Y, bw, unbiased=True)
        MMD_q = u_stat(X, Y, bw)
        MMD_eps = MMD_hat + epsilon * MMD_q
        s_a = sigma_spec_a(d, n, X, Y, bw, MMD_hat)["s.a"]
        s_q = sigma_spec_q(d, n, X, Y, bw, MMD_hat)["s.q"]
        sd_mmd_eps = s_a + epsilon * s_q
        stat_eps = float(cp.sqrt(n) * MMD_eps / sd_mmd_eps)
        stat_q = float(cp.sqrt(n) * MMD_q / s_q)
        reject_eps = abs(stat_eps) > norm.ppf(1 - level / 2)
        reject_q = abs(stat_q) > norm.ppf(1 - level / 2)
        return {
            "spec.reject.1": reject_eps,
            "spec.reject.q.1": reject_q
        }

    # model selection part
    MMD_hat_1 = MMD_stand(X, Y, bw, unbiased=True)
    MMD_q_1 = u_stat(X, Y, bw)
    MMD_eps_1 = MMD_hat_1 + epsilon * MMD_q_1
    out_spec_a_1 = sigma_spec_a(d, n, X, Y, bw, MMD_hat_1)
    out_spec_q_1 = sigma_spec_q(d, n, X, Y, bw, MMD_hat_1)

    MMD_hat_2 = MMD_stand(X, Y2, bw, unbiased=True)
    MMD_q_2 = u_stat(X, Y2, bw)
    MMD_eps_2 = MMD_hat_2 + epsilon * MMD_q_2
    out_spec_a_2 = sigma_spec_a(d, n, X, Y2, bw, MMD_hat_2)
    out_spec_q_2 = sigma_spec_q(d, n, X, Y2, bw, MMD_hat_2)

    s_a_squared = 4 * cp.var(out_spec_a_1["h1.a"] - out_spec_a_2["h1.a"], ddof=1)
    s_q_squared = 8 * cp.var(out_spec_q_1["h1.q"] - out_spec_q_2["h1.q"], ddof=1)

    sd_mmd_sel = cp.sqrt(s_a_squared) + epsilon * cp.sqrt(s_q_squared)
    stat_sel = cp.sqrt(n) * (MMD_eps_1 - MMD_eps_2) / sd_mmd_sel
    stat_sel_q = cp.sqrt(n) * (MMD_q_1 - MMD_q_2) / cp.sqrt(s_q_squared)

    reject_sel = cp.abs(stat_sel) > norm.ppf(1 - level / 2)
    reject_sel_q = cp.abs(stat_sel_q) > norm.ppf(1 - level / 2)

    return {
        "selec.reject": bool(reject_sel.get()),
        "selec.reject.q": bool(reject_sel_q.get())
    }

def run_degenerate_case_simulation():
    sample_sizes = [100,250,500,1000]
    sigmas = np.arange(1.0, 1.5, 0.1)
    d_values = [2, 16]
    nsim = 1000
    level = 0.05
    rng = np.random.default_rng(seed=42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, d in enumerate(d_values):
        ax = axes[ax_idx]
        for n in sample_sizes:
            bw = np.sqrt(d / 2)
            epsilon = n ** (-1 / 2.5)
            power_q = []
            power_eps = []
            for sigma in sigmas:
                reject_q = []
                reject_eps = []
                for _ in tqdm(range(nsim), desc=f"d={d}, n={n}, sigma={sigma:.1f}"):
                    data = sim_mu_2sampl(d, n, SD_theor1=(1.0, sigma), SD_theor2=(1.0, 1.0), rng=rng)
                    result = BFM_test(d, n, epsilon, data['X'], data['hat_Y1'], data['hat_Y2'], bw=bw, model_selection=True, level=level)
                    reject_q.append(result['selec.reject.q'])
                    reject_eps.append(result['selec.reject'])
                power_q.append(np.mean(reject_q))
                power_eps.append(np.mean(reject_eps))
            ax.plot(sigmas, power_q, marker='o', label=f"MMD_q n={n}", linestyle='-')
            ax.plot(sigmas, power_eps, marker='o', label=f"eps n={n}", linestyle='--')
        ax.axhline(y=0.05, color='k', linestyle='dashed')
        ax.set_xlabel("Stand. dev.")
        ax.set_ylabel("Emp. Level/Power")
        ax.set_title(f"p={d}")
        ax.set_ylim(0, 1.05)
        ax.legend()
    plt.tight_layout()
    plt.show()

run_degenerate_case_simulation()
