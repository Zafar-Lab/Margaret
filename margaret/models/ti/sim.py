# This module implements some quasi-local methods for similarity computation

import numpy as np

from copy import deepcopy
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def compute_lpi(adj_conn, beta=1.0, n_steps=3):
    # Local Path Index computation
    conn = deepcopy(adj_conn)

    if not isinstance(conn, csr_matrix):
        conn = csr_matrix(conn)

    S = conn
    prev = conn
    for i in range(n_steps):
        prev = prev @ conn
        S += np.power(beta, i) * prev

    return S


def compute_lrw(adj_conn, n_steps=2):
    # Local Random Walk computation
    conn = deepcopy(adj_conn)

    if not isinstance(conn, csr_matrix):
        conn = csr_matrix(conn)

    T = normalize(conn, norm="l1", axis=1)
    prev = T
    for _ in range(n_steps):
        T = T @ prev
        prev = T

    return T + T.transpose()
