import numpy as np
import scipy
import math

def Naive(matrix, threshold):
    copy_matrix = matrix.copy()
    copy_matrix[np.abs(matrix) <= threshold] = 0
    copy_matrix = scipy.sparse.csr_matrix(copy_matrix)
    return copy_matrix

def Naive_nonzeros(matrix, threshold):
    return np.sum(np.abs(matrix) > threshold)

def AHK06(matrix, threshold):
    copy_matrix = matrix.copy()
    n, d = matrix.shape
    probs = np.random.random((n, d))
    copy_matrix[np.abs(matrix) < threshold] = 0
    copy_matrix[probs < (np.abs(matrix) / threshold) * (np.abs(matrix) < threshold)] = threshold

    copy_matrix = scipy.sparse.csr_matrix(copy_matrix)
    return copy_matrix

def AHK06_nonzeros(matrix, threshold):
    n, d = matrix.shape
    indices = np.abs(matrix) < threshold
    return n * d - np.sum(indices) + np.sum(matrix[indices] / threshold)

def compute_row_distribution(matrix, s, delta, row_norms):
    m, n = matrix.shape
    z = row_norms / np.sum(row_norms)
    alpha, beta = math.sqrt(np.log((m + n) / delta) / s), np.log((m + n) / delta) / (3 * s)
    zeta = 1
    rou = (alpha * z / (2 * zeta) + ((alpha * z / (2 * zeta)) ** 2 + beta * z / zeta) ** (1 / 2)) ** 2
    sum = np.sum(rou)
    while np.abs(sum - 1) > 1e-5:
        zeta *= sum
        rou = (alpha * z / (2 * zeta) + ((alpha * z / (2 * zeta)) ** 2 + beta * z / zeta) ** (1 / 2)) ** 2
        sum = np.sum(rou)
    return rou

def AKL13(matrix, s):
    matrix = matrix.T
    s = int(s)
    n, d = matrix.shape
    row_norms = np.linalg.norm(matrix, axis=1, ord=1)
    rou = compute_row_distribution(matrix, s, 0.1, row_norms)
    nonzero_indices = matrix.nonzero()
    data = matrix[nonzero_indices]
    row_norms[row_norms == 0] = 1
    probs_matrix = rou.reshape((n, 1)) * matrix / row_norms.reshape((n, 1))
    probs = probs_matrix[nonzero_indices]
    probs /= np.sum(probs)
    indices = np.arange(len(data))
    selected = np.random.choice(indices, s, p=probs, replace=True)
    result = np.zeros((n, d))
    np.add.at(result, (nonzero_indices[0][selected], nonzero_indices[1][selected]), data[selected] / (probs[selected] * s))
    result = result.T
    matrix = matrix.T
    result = scipy.sparse.csr_matrix(result)
    return result

def AKL13_nonzeros(matrix, s):
    matrix = matrix.T
    s = int(s)
    n = matrix.shape[0]
    row_norms = np.linalg.norm(matrix, axis=1, ord=1)
    rou = compute_row_distribution(matrix, s, 0.1, row_norms)
    nonzero_indices = matrix.nonzero()
    data = matrix[nonzero_indices]
    row_norms[row_norms == 0] = 1
    probs_matrix = rou.reshape((n, 1)) * matrix / row_norms.reshape((n, 1))
    probs = probs_matrix[nonzero_indices]
    probs /= np.sum(probs)
    indices = np.arange(len(data))
    selected = np.random.choice(indices, s, p=probs, replace=True)
    matrix = matrix.T
    return len(np.unique(selected))

def row_operation(copy_row, threshold):
    argzero = np.argwhere((np.abs(copy_row) <= threshold) * (copy_row != 0))
    argzero = argzero.reshape(len(argzero),)
    argzero_copy = copy_row[argzero]
    copy_row[argzero] = 0
    sum = np.sum(argzero_copy)
    if sum != 0:
        k = math.ceil(sum / threshold)
            
        indices = np.random.choice(argzero, k, p=argzero_copy/sum, replace=True)
        np.add.at(copy_row, indices, sum / k)

def RMR(matrix, threshold):
    copy_matrix = matrix.copy()
    np.apply_along_axis(row_operation, 1, copy_matrix, threshold)
    copy_matrix = scipy.sparse.csr_matrix(copy_matrix)
    return copy_matrix

def RMR_nonzeros(matrix, threshold):
    n, d = matrix.shape
    sum = 0
    for i in range(n):
        argzero = np.argwhere(np.abs(matrix[i, :]) <= threshold)
        sum2 = np.sum(np.abs(matrix[i, argzero]))
        if sum2 != 0:
            k = math.ceil(sum2 / threshold)
            sum += d - len(argzero) + np.sum(1 - (1 - np.abs(matrix[i, argzero]) / sum2) ** k)
        else: 
            sum += d - len(argzero)
    return sum

def DZ11(matrix, threshold):
    copy_matrix = matrix.copy()
    n, d = matrix.shape
    norm_fro = np.linalg.norm(matrix, ord="fro")
    copy_matrix[np.abs(matrix) <= threshold / (n + d)] = 0
    s = int(14 * (n + d) * np.log(np.sqrt(2) / 2 * (n + d)) * (norm_fro / threshold) ** 2)
    nonzero_indices = copy_matrix.nonzero()
    data = copy_matrix[nonzero_indices]
    probs_matrix = copy_matrix * copy_matrix
    probs = probs_matrix[nonzero_indices]
    probs /= np.sum(probs)
    indices = np.arange(len(data))
    selected = np.random.choice(indices, s, p=probs, replace=True)
    result = np.zeros((n, d))
    np.add.at(result, (nonzero_indices[0][selected], nonzero_indices[1][selected]), data[selected] / (probs[selected] * s))
    result = scipy.sparse.csr_matrix(result)
    return result

def DZ11_nonzeros(matrix, threshold):
    copy_matrix = matrix.copy()
    n, d = matrix.shape
    norm_fro = np.linalg.norm(matrix, ord="fro")
    copy_matrix[np.abs(matrix) <= threshold / (n + d)] = 0
    s = int(14 * (n + d) * np.log(np.sqrt(2) / 2 * (n + d)) * (norm_fro / threshold) ** 2)
    nonzero_indices = copy_matrix.nonzero()
    data = copy_matrix[nonzero_indices]
    probs_matrix = copy_matrix * copy_matrix
    probs = probs_matrix[nonzero_indices]
    probs /= np.sum(probs)
    indices = np.arange(len(data))
    selected = np.random.choice(indices, s, p=probs, replace=True)
    return len(np.unique(selected))

def BKKS21(matrix, s):
    n, d = matrix.shape
    probs = np.random.random((n, d))
    row_norms = np.linalg.norm(matrix, axis=1, ord=1)
    col_norms = np.linalg.norm(matrix, axis=0, ord=1)
    p1 = np.abs(matrix) / np.sum(np.abs(matrix))
    p2 = np.abs(matrix) * (row_norms / np.sum(row_norms ** 2)).reshape(-1, 1)
    p3 = np.abs(matrix) * (col_norms / np.sum(col_norms ** 2)).reshape(1, -1)
    p = np.minimum(1, s * np.maximum(p1, np.maximum(p2, p3)))
    probs[p == 0] = 1
    p[p == 0] = 1
    result = (matrix / p) * (probs < p)

    result = scipy.sparse.csr_matrix(result)
    return result

def BKKS21_nonzeros(matrix, s):
    row_norms = np.linalg.norm(matrix, axis=1, ord=1)
    col_norms = np.linalg.norm(matrix, axis=0, ord=1)
    p1 = np.abs(matrix) / np.sum(np.abs(matrix))
    p2 = np.abs(matrix) * (row_norms / np.sum(row_norms ** 2)).reshape(-1, 1)
    p3 = np.abs(matrix) * (col_norms / np.sum(col_norms ** 2)).reshape(1, -1)
    p = np.minimum(1, s * np.maximum(p1, np.maximum(p2, p3)))
    return np.sum(p)