import numpy as np
import scipy

try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    pass


def get_sparse_plus_low_rank(A: np.ndarray, thresold_perc: float = 1, rank: int = 5):
    """
        :param A: dose influence matrix
        :param thresold_perc: thresold percentage. Default to 1% of max(A)
        :type rank: rank of L = A-S.
        :returns: S, H, W using randomized svd
    """
    tol = np.max(A) * thresold_perc * 0.01
    S = np.where(A > tol, A, 0)
    if rank == 0:
        S = scipy.sparse.csr_matrix(S)
        return S
    else:
        print('Running svd..')
        [U, svd_S, V] = randomized_svd(A - S, n_components=rank + 1, random_state=0)
        print('svd done!')
        H = U[:, :rank]
        W = np.diag(svd_S[:rank]) @ V[:rank, :]
        S = scipy.sparse.csr_matrix(S)
        return S, H, W
