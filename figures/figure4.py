import numpy as np
# import torch
from scipy.stats import norm
from scipy.spatial.distance import cdist
from math import sqrt
import pandas as pd
import time
from tqdm import tqdm
from matplotlib import pyplot as plt

# path_y="C:\\Users\\Fabia\\Downloads\\y1.csv"
# path_x="C:\\Users\\Fabia\\Downloads\\x.csv"
# path_y2="C:\\Users\\Fabia\\Downloads\\y2.csv"

"""
    For the functions kern_k and MMD_stand I'm using code similar to https://github.com/sshekhar17/PermFreeMMD/blob/main/src/utils.py#L76, I just excluded the diagonal elements.
    For the other functions also the figure 1-4 data generating functions I'm using something similar to https://github.com/florianbrueck/MMD_tests_for_model_selection/blob/main/BFM-TEST.R.
    The code for the plots is written by myself, without using the R code Professor Min provived since I thought my approach is easier.
    All the references on which equation it represents is based on https://arxiv.org/abs/2305.07549 version 2 of Distribution free MMD tests for model selection with estimated parameters 
    by Florian Brueck, Jean-David Fermanian and Aleksey Min.

"""


def kern_k(x, y=None, bw=1.0, amp=1.0):
    """
    Gaussian RBF kernel.

    Parameters
    ----------
    x, y : array‑like (1‑D)
        Input vectors of equal length d.
    bw : float
        Kernel tuning parameter.
    amp : optional
        Included only for API compatibility; not used.

    Returns
    -------
    float
        k(x1, x2)
    """
    x = np.asarray(x, dtype=float)  # to make sure that the inputs are numpy arrays
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    y = x if y is None else np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    dists = cdist(x, y, metric="euclidean")  # we use cdist for parallelization
    squared_dists = dists * dists
    k = amp * np.exp(-(1 / (2 * bw * bw)) * squared_dists)  # changed theta_1 to bw=bandwidth, bw=sqrt(theta1/2)
    return k


def MMD_stand(X, Y, bw, unbiased=False, return_float=False):
    """
        Unbiased U‑statistic estimator of the squared MMD described in Equation (1).  (later in the BFM test function it is called MMD.hat)

        Parameters
        ----------
        d : int
            Dimension of each X‑sample (and Y‑sample) vector.
        X, Y : array‑like, shape (n, d)  or  (n, ) for d == 1
            Paired observations; rows are i.i.d.
            If d == 1 a 1‑D array is accepted and reshaped.
        bw : float
            Tuning parameter passed to `kern_k`.
        Returns
        -------
        float
            The statistic  2 / (n·(n−1))  ·  Σ_{1≤i<j≤n} h(Z_i, Z_j)
            where  Z_i = (X_i, Y_i).
        """
    X = np.asarray(X, dtype=float)  # again making sure that inputs are numpy arrays
    Y = np.asarray(Y, dtype=float)
    Kxx = kern_k(X, X, bw=bw)  # computing all the kernels for all the combinations
    Kyy = kern_k(Y, Y, bw=bw)
    Kxy = kern_k(X, Y, bw=bw)

    n, m = len(X), len(Y)  # in our study it is always m==n
    if unbiased:
        np.fill_diagonal(Kxx, 0.0)  # we just set the diagonal elements to 0 since we want to exlcude i==j in our sum
        np.fill_diagonal(Kyy, 0.0)

        # Exclude k(x_i, y_i) when i == j
        if n == m:  # here always the case!
            np.fill_diagonal(Kxy, 0.0)  # we want to exclude all cases where i==j
            term3 = 2 * Kxy.sum() / (n * (m - 1))
        else:
            term3 = 2 * Kxy.sum() / (n * m)

        term1 = Kxx.sum() / (n * (n - 1))
        term2 = Kyy.sum() / (m * (m - 1))
    else:  # in the R code there wasn't a biased part
        term1 = Kxx.sum() / (n * n)
        term2 = Kyy.sum() / (m * m)
        term3 = 2 * Kxy.sum() / (n * m)
    MMD_squared = term1 + term2 - term3
    if return_float:
        return MMD_squared
    else:
        return MMD_squared


def u_stat(X, Y, bw):
    """
    Fully vectorized implementation of the U-statistic estimator of MMD_q^2
    as in section 2.2 on page 7.

    Parameters
    ----------
    X, Y : ndarray of shape (n, d)
        Input paired samples. n MUST BE EVEN.
    bw : float
        Bandwidth for the RBF kernel.

    Returns
    -------
    float
        MMD_q^2 U-statistic estimate.
    """
    X = np.asarray(X, dtype=float)  # again making sure that input is a numpy array
    Y = np.asarray(Y, dtype=float)
    n, d = X.shape

    if n % 2 != 0:
        raise ValueError("Sample size n must be even.")

    m = n // 2
    # Forming paired blocks from X and Y:
    # These pairs (X1, X2), (Y1, Y2) represent index-based pairing of observations:
    # - X1 and Y1 take the even-indexed samples (0, 2, 4, ...)
    # - X2 and Y2 take the following odd-indexed samples (1, 3, 5, ...)
    # This mirrors the definition in the MMD_q U-statistic where we form
    # pairs (Z_i, Z_j) = ((X_i, Y_i), (X_j, Y_j)) with i < j
    X1, X2 = X[::2], X[1::2]  # x^{(1)}, x^{(2)}
    Y1, Y2 = Y[::2], Y[1::2]  # y^{(1)}, y^{(2)}

    # Compute squared pairwise distances
    def rbf(A, B):
        dists = cdist(A, B, metric="sqeuclidean")
        return np.exp(-dists / (2 * bw ** 2))

    K_x1x1 = rbf(X1, X1)  # again computing kernels
    K_y1y1 = rbf(Y1, Y1)
    K_y2x2 = rbf(Y2, X2)

    # U-statistic: remove diagonals and take upper triangle
    H = K_x1x1 + K_y1y1 - K_y2x2 - K_y2x2.T
    upper_sum = np.sum(np.triu(H, k=1))

    return 2 * upper_sum / (m * (m - 1))  # use upper triangle as in the U-statistics seminar


def sim_mu_fig1(d, n, sd_temp: float = 1.2, rng: np.random.Generator | None = None):
    """
    Simulate twod‑dimensional samples (X,Y) and the shifted Ŷ used in Fig.1.

    Parameters
    ----------
    d : int
        Dimension of each observation.
    n : int
        Sample size (number of rows in X and Y).
    sd_temp : float, default 1.2
        Standard deviation of the Y‑sample (X has unit variance).
    rng : np.random.Generator, optional
        Random‑number generator for reproducibility.
        Example: rng = np.random.default_rng(42)

    Returns
    -------
    dict with keys
        'X'        : ndarray, shape (n, d)  –standard‑normal sample
        'Y'        : ndarray, shape (n, d)  –N(0,sd_temp²) sample
        'hat.Y'    : ndarray, shape (n, d)  –Y shifted by ȳ_X (row‑wise)    #the output which is used in the figure 1
        'hat.mu.X' : ndarray, shape (d, )    –column means of X
    """
    if rng is None:
        rng = np.random.default_rng()

    X = rng.standard_normal(size=(n, d))  # just a (n,d) matrix of standard normal distributed values
    Y = rng.normal(loc=0.0, scale=sd_temp,
                   size=(n, d))  # just a (n,d) matrix of normal distributed values (with mean 0, std=sd_temp)

    hat_mu_X = X.mean(axis=0)  # column means (shape: d, )
    hat_Y = Y + hat_mu_X  # broadcast adds ȳ_X to every row, we shift Y by the column mean of X

    return {"X": X, "Y": Y, "hat.Y": hat_Y, "hat.mu.X": hat_mu_X}


def sim_mu_2sampl(d: int, n: int, SD_theor1=(1.0, 1.0), SD_theor2=(1.0, 1.0), rng: np.random.Generator | None = None):
    """
    Generate two paired samples with different per‑block standard deviations.
    Used for figure 3 and figure 4

    Parameters
    ----------
    d : int
        Total dimension (must be even; split into two blocks of length d/2).
    n : int
        Sample size (number of rows of each matrix).
    SD_theor1, SD_theor2 : tuple(float, float), default (1, 1)
        Standard deviations for the first and second halves of Y in
        sample1 and sample2, respectively.  Example:
            SD_theor1 = (1, 2)  # σ=1 for first d/2 coords, then σ=2 for the last d/2 coords in sample 1
    rng : np.random.Generator, optional
        Random‑number generator (e.g. np.random.default_rng(42)) for
        reproducibility.

    Returns
    -------
    dict with keys
        'X'        : ndarray, shape (n, d)   –baseline N(0,1) sample
        'Y'        : ndarray, shape (n, d)   –second sample's raw Y
        'hat.Y1'   : ndarray, shape (n, d)   –shifted first sample
        'hat.Y2'   : ndarray, shape (n, d)   –shifted second sample
        'hat.mu.X' : ndarray, shape (d, )     –column means of X
    """
    if d % 2 != 0:
        raise ValueError("d must be even (so it can be split into two halves).")

    if rng is None:
        rng = np.random.default_rng()

    X = rng.standard_normal(size=(n, d))

    # Column means of X
    hat_mu_X = X.mean(axis=0)  # shape (d, )
    mu_mat = hat_mu_X  # for broadcasting later
    # length of each half
    dd = d // 2

    # first sample
    Y1_block1 = rng.normal(0.0, SD_theor1[0], size=(n, dd))  # has std of the first half
    Y1_block2 = rng.normal(0.0, SD_theor1[1], size=(n, dd))  # has std of the second half
    Y_first = np.hstack([Y1_block1, Y1_block2])  # raw Y of sample1

    hat_Y1 = Y_first + mu_mat  # shift by μ_hat_X

    # second sample
    Y2_block1 = rng.normal(0.0, SD_theor2[0], size=(n, dd))  # now we use the std of sample 2 first half
    Y2_block2 = rng.normal(0.0, SD_theor2[1], size=(n, dd))
    Y_second = np.hstack([Y2_block1, Y2_block2])  # raw Y of sample2

    hat_Y2 = Y_second + mu_mat  # shift by μ_hat_X

    return {
        "X": X,
        "Y": Y_second,  # as in the R code right now
        "hat_Y1": hat_Y1,
        "hat_Y2": hat_Y2,
        "hat_mu_X": mu_mat,
    }


def stand_h_vec(X, Y, bw):
    # h(Zi, Zj) = k(xi, xj) + k(yi, yj) - k(xi, yj) - k(xj, yi)
    K_xx = kern_k(X, X, bw=bw)
    K_yy = kern_k(Y, Y, bw=bw)
    K_xy = kern_k(X, Y, bw=bw)
    return K_xx + K_yy - K_xy - K_xy.T


def sigma_spec_a(d, n, X, Y, bw, mmd_hat):  # should only be used inside of BFM-test
    """
        In the paper in section 3.2 on page 12 the estimator: sigma_tilde squared alpha_n.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.shape != (n, d) or Y.shape != (n, d):
        raise ValueError("Shapes of X and Y must both be (n, d).")

    H = stand_h_vec(X, Y, bw=bw)  # shape (n, n)
    h1 = H.mean(axis=1) - mmd_hat  # vector of h₁(i), mean over the rows of H -mmd_hat

    s_a_squared = 4.0 * np.var(h1,
                               ddof=1)  # ddof=1 means we use the sample variance, not the population variance (sample variance has a multiplier of 1/(n-1) and not 1/n  )
    s_a = np.sqrt(s_a_squared)

    return {"s.a": s_a, "h1.a": h1}


def new_h_vec(Z, bw):
    """
        Z is a Matrix where each row contains [X1, Y1, X2, Y2] concatenated.
        In section 2.2 on page 8 the kernel q.
    """
    m = Z.shape[0]
    d = Z.shape[1] // 4

    # Decompose Z into its four parts:
    x1 = Z[:, 0 * d:1 * d]
    y1 = Z[:, 1 * d:2 * d]
    x2 = Z[:, 2 * d:3 * d]
    y2 = Z[:, 3 * d:4 * d]

    k_x1x1 = kern_k(x1, x1, bw=bw)  # similarity within x1 block
    k_y1y1 = kern_k(y1, y1, bw=bw)  # similarity within y1 block
    k_x2y2 = kern_k(x2, y2, bw=bw)  # cross-similarity between x2 and y2
    k_x2y2_T = k_x2y2.T  # avoid recomputing

    return k_x1x1 + k_y1y1 - k_x2y2 - k_x2y2_T


def sigma_spec_q(d, n, X, Y, bw, mmd_hat):  # should only be used inside the BFM_test function since it uses mmd_hat
    """
        In the paper in section 3.2 on page 12 the estimator: sigma_tilde squared q,alpha_n.

        Returns dictionary with:
        - 's.q'   : standard deviation estimate of MMD_q
        - 'h1.q'  : vector of centered h1 values (first-order terms)
    """
    if n % 2 != 0:
        raise ValueError("n must be even.")

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != (n, d) or Y.shape != (n, d):
        raise ValueError("Shapes of X and Y must both be (n, d).")

    nn = n // 2
    idx = np.arange(0, n, 2)  # even indices
    # Construct Z matrix: rows contain concatenated [x1, y1, x2, y2]
    # where x1 = X[idx], x2 = X[idx+1], y1 = Y[idx], y2 = Y[idx+1]
    Z = np.hstack([X[idx], Y[idx], X[idx + 1], Y[idx + 1]])  # shape (nn, 4d)

    # Compute the kernel-based h(Z_i, Z_j) matrix
    H = new_h_vec(Z, bw=bw)
    # center each row mean by subtracting the MMD_hat estimate
    h1 = H.mean(axis=1) - mmd_hat

    s_q_squared = 8.0 * np.var(h1, ddof=1)  # again we use 1/(n-1) instead of 1/n thus ddof=1
    s_q = np.sqrt(s_q_squared)

    return {"s.q": s_q, "h1.q": h1}


def BFM_test(d, n, epsilon, X, Y, Y2=None, bw=1.0, model_selection=False, level=0.05):
    """
    Translation of the R function BFM.test into Python. The test statistic in Thm 3.1 on page 14.
    if model_selection=False: we want to test, if the underlying data X is similar to the data Y.
    if model_selection=True: we want to test, if the underlying data X is more similar to the data Y or Y2.
    """
    if not model_selection:
        # MMD.hat (Equation (1))
        MMD_hat = MMD_stand(X, Y, bw=bw, unbiased=True)

        # MMD.q
        MMD_q = u_stat(X, Y, bw=bw)

        # MMD.eps, section 2.2 on page 8
        MMD_eps = MMD_hat + epsilon * MMD_q

        # Variance estimates
        s_a = sigma_spec_a(d, n, X, Y, bw=bw, mmd_hat=MMD_hat)["s.a"]
        s_q = sigma_spec_q(d, n, X, Y, bw=bw, mmd_hat=MMD_hat)["s.q"]

        sd_mmd_eps = s_a + epsilon * s_q  # sigma_hat_n in Equation (6) on page 9. also in Thm 3.1

        # Specification test statistics
        spec_test_stat_1 = np.sqrt(n) * MMD_eps / sd_mmd_eps  # T_n in Thm 3.1 on page 14.
        spec_reject_1 = (np.abs(spec_test_stat_1) > norm.ppf(
            1 - level / 2))  # reject if the abs(spec_test_stat) > quantile_alpha of standard normal distribution

        spec_test_stat_q_1 = np.sqrt(n) * MMD_q / s_q
        spec_reject_q_1 = (np.abs(spec_test_stat_q_1) > norm.ppf(
            1 - level / 2))  # reject if the abs(spec_test_stat_q_1) > quantile_alpha of standard normal distribution

        out = {
            "selec.test.stat": None,
            "selec.reject": None,
            "selec.test.stat.q": None,
            "selec.reject.q": None,
            "sd.mmd.sel": None,
            "sd.mmd.sel.q": None,

            "spec.test.stat.1": spec_test_stat_1,
            "spec.reject.1": spec_reject_1,
            # for MMD_eps 1000 times is a vector and first column is for epsilon 1 and then count how many are 1 and divide by 1000 to get the graph (figure 1,2)
            "MMD.eps.1": MMD_eps,
            "MMD.hat.1": MMD_hat,
            "MMD.q.1": MMD_q,
            "spec.test.stat.q.1": spec_test_stat_q_1,
            "spec.reject.q.1": spec_reject_q_1,  # for MMD_q 1000 times
            "s.a.1": s_a,
            "s.q.1": s_q,

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

    else:
        ### SAMPLE 1 (Y)
        # MMD.hat (Equation (1))
        MMD_hat_1 = MMD_stand(X, Y, bw=bw, unbiased=True)

        # MMD.q
        MMD_q_1 = u_stat(X, Y, bw=bw)
        # MMD.eps, section 2.2 on page 8
        MMD_eps_1 = MMD_hat_1 + epsilon * MMD_q_1

        # Estimate variances
        out_spec_a_1 = sigma_spec_a(d, n, X, Y, bw=bw, mmd_hat=MMD_hat_1)
        out_spec_q_1 = sigma_spec_q(d, n, X, Y, bw=bw, mmd_hat=MMD_hat_1)

        s_a_1 = out_spec_a_1["s.a"]
        s_q_1 = out_spec_q_1["s.q"]

        sd_mmd_eps_1 = s_a_1 + epsilon * s_q_1  # sigma_hat_n in Equation (6) on page 9. also in Thm 3.1

        spec_test_stat_1 = np.sqrt(n) * MMD_eps_1 / sd_mmd_eps_1  # T_n in Thm 3.1 on page 14.
        spec_reject_1 = (np.abs(spec_test_stat_1) > norm.ppf(1 - level / 2))

        spec_test_stat_q_1 = np.sqrt(n) * MMD_q_1 / s_q_1
        spec_reject_q_1 = (np.abs(spec_test_stat_q_1) > norm.ppf(1 - level / 2))

        ### SAMPLE 2 (Y2)
        # MMD.hat (Equation (1))
        MMD_hat_2 = MMD_stand(X, Y2, bw=bw, unbiased=True)
        # MMD.q
        MMD_q_2 = u_stat(X, Y2, bw=bw)
        # MMD.eps, section 2.2 on page 8
        MMD_eps_2 = MMD_hat_2 + epsilon * MMD_q_2

        # Estimate variances
        out_spec_a_2 = sigma_spec_a(d, n, X, Y2, bw=bw, mmd_hat=MMD_hat_2)
        out_spec_q_2 = sigma_spec_q(d, n, X, Y2, bw=bw, mmd_hat=MMD_hat_2)

        s_a_2 = out_spec_a_2["s.a"]
        s_q_2 = out_spec_q_2["s.q"]

        sd_mmd_eps_2 = s_a_2 + epsilon * s_q_2  # sigma_hat_n in Equation (6) on page 9. also in Thm 3.1

        spec_test_stat_2 = np.sqrt(n) * MMD_eps_2 / sd_mmd_eps_2  # T_n in Thm 3.1 on page 14.
        spec_reject_2 = (np.abs(spec_test_stat_2) > norm.ppf(1 - level / 2))

        spec_test_stat_q_2 = np.sqrt(n) * MMD_q_2 / s_q_2
        spec_reject_q_2 = (np.abs(spec_test_stat_q_2) > norm.ppf(1 - level / 2))

        ### Model Selection Part
        h1_a_1 = out_spec_a_1["h1.a"]
        h1_a_2 = out_spec_a_2["h1.a"]
        s_a_squared = 4 * np.var(h1_a_1 - h1_a_2, ddof=1)

        h1_q_1 = out_spec_q_1["h1.q"]
        h1_q_2 = out_spec_q_2["h1.q"]
        s_q_squared = 8 * np.var(h1_q_1 - h1_q_2, ddof=1)

        sd_mmd_sel = np.sqrt(s_a_squared) + epsilon * np.sqrt(s_q_squared)

        selec_test_stat = np.sqrt(n) * (MMD_eps_1 - MMD_eps_2) / sd_mmd_sel
        selec_reject = (np.abs(selec_test_stat) > norm.ppf(1 - level / 2))

        selec_test_stat_q = np.sqrt(n) * (MMD_q_1 - MMD_q_2) / np.sqrt(s_q_squared)
        selec_reject_q = (np.abs(selec_test_stat_q) > norm.ppf(1 - level / 2))

        out = {
            "selec.test.stat": selec_test_stat,
            "selec.reject": selec_reject,  # for epsilon
            "selec.test.stat.q": selec_test_stat_q,
            "selec.reject.q": selec_reject_q,  # for MMD_q
            "sd.mmd.sel": sd_mmd_sel,
            "sd.mmd.sel.q": np.sqrt(s_q_squared),

            "spec.test.stat.1": spec_test_stat_1,
            "spec.reject.1": spec_reject_1,
            "MMD.eps.1": MMD_eps_1,
            "MMD.hat.1": MMD_hat_1,
            "MMD.q.1": MMD_q_1,
            "spec.test.stat.q.1": spec_test_stat_q_1,
            "spec.reject.q.1": spec_reject_q_1,
            "s.a.1": s_a_1,
            "s.q.1": s_q_1,

            "spec.test.stat.2": spec_test_stat_2,
            "spec.reject.2": spec_reject_2,
            "MMD.eps.2": MMD_eps_2,
            "MMD.hat.2": MMD_hat_2,
            "MMD.q.2": MMD_q_2,
            "spec.test.stat.q.2": spec_test_stat_q_2,
            "spec.reject.q.2": spec_reject_q_2,
            "s.a.2": s_a_2,
            "s.q.2": s_q_2,
        }

    return out

def run_non_degenerate_case():
    sample_sizes = [100, 250, 500, 1000]
    sigmas = np.arange(1.2, 1.7, 0.1)
    d_values = [2, 16]
    nsim = 100
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
                    #the data is generated in the way outlined in section 5.2 on page 21
                    data = sim_mu_2sampl(d, n, SD_theor1=(1.2, sigma), SD_theor2=(1.2, 1.2), rng=rng)   #second half has a varying std
                    X = data['X']
                    Y1 = data['hat_Y1']
                    Y2 = data['hat_Y2']

                    result = BFM_test(d, n, epsilon, X, Y1, Y2, bw=bw, model_selection=True, level=level)   #model_selection=True

                    reject_q.append(result['selec.reject.q'])
                    reject_eps.append(result['selec.reject'])

                power_q.append(np.mean(reject_q))
                power_eps.append(np.mean(reject_eps))

            # Plot both lines for each n
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

# Run the updated simulation with new std inputs for Y1 and Y2
run_non_degenerate_case()
