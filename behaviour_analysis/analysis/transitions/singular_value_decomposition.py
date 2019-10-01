import numpy as np


def SVD(M):
    """Finds the singular value decomposition of the symmetric and antisymmetric components of matrix M.

    Parameters
    ----------
    M : np.ndarray

    Returns
    -------
    USVs, USVa : np.ndarray
    """
    # Compute symmetric and antisymmetric components of matrix, M
    S = 0.5 * (M + M.T)
    A = 0.5 * (M - M.T)
    # Singular value decomposition
    Us, ss, VsT = np.linalg.svd(S)
    Ua, sa, VaT = np.linalg.svd(A)
    # Transpose input vectors
    Vs = VsT.T
    Va = VaT.T
    # Singular values of symmetric component
    Ds = np.zeros(Us.shape)
    Ds[np.diag_indices(len(Ds))] = ss
    # Singular values of antisymmetric component
    Da = np.zeros(Ua.shape)
    Da[np.diag_indices(len(Da))] = sa
    # Combine matrices
    USVs = np.array([Us, Ds, Vs])
    USVa = np.array([Ua, Da, Va])
    return USVs, USVa
