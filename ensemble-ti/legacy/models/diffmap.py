import numpy as np
import scanpy as sc

from scipy.sparse import csr_matrix, find
from scipy.sparse.linalg import eigs


class DiffusionMap:
    """This Diffusion Map implementation is inspired from the implementation of
    https://github.com/dpeerlab/Palantir/blob/master/src/palantir/utils.py
    """

    def __init__(self, n_components=10, n_neighbors=30, alpha=0, **kwargs):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.kwargs = kwargs

    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("The input data must be a numpy array!")
        print("Determing nearest neighbor graph...")
        temp = sc.AnnData(data)
        sc.pp.neighbors(temp, n_neighbors=self.n_neighbors, n_pcs=0, **self.kwargs)
        N = temp.shape[0]
        kNN = temp.obsp["distances"]

        # Adaptive k
        # This gives the lth neighbor as described in the Palantir paper
        adaptive_k = int(np.floor(self.n_neighbors / 10))
        adaptive_std = np.zeros(N)

        for i in np.arange(len(adaptive_std)):
            # Take the distance to lth nearest neighbor as the sigm value
            adaptive_std[i] = np.sort(kNN.data[kNN.indptr[i] : kNN.indptr[i + 1]])[
                adaptive_k - 1
            ]

        # Kernel Construction (Anisotropic Scaling)
        x, y, dists = find(kNN)
        dists = dists / adaptive_std[x]
        W = csr_matrix((np.exp(-dists), (x, y)), shape=[N, N])

        # Diffusion components (Make the kernel symmetric for better performance)
        kernel = W + W.T

        # Row-stochastic Normalization
        D = np.ravel(kernel.sum(axis=1))
        if self.alpha > 0:
            D[D != 0] = D[D != 0] ** (-alpha)
            mat = csr_matrix((D, (range(N), range(N))), shape=[N, N])
            kernel = mat.dot(kernel).dot(mat)
            D = np.ravel(kernel.sum(axis=1))

        D[D != 0] = 1 / D[D != 0]
        T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)

        # Eigen value dcomposition
        # Taking the n + 1 components cause the first eigenvector is trivial
        # and will be removed
        D, V = eigs(T, self.n_components, tol=1e-4, maxiter=1000)
        D = np.real(D)
        V = np.real(V)
        inds = np.argsort(D)[::-1]
        D = D[inds]
        V = V[:, inds]

        # Account for the multi-scale distance computation
        # which avoids the selection of an additional t parameter
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

        # Create are results dictionary
        return {"T": T, "eigenvectors": V, "eigenvalues": D, "kernel": kernel}

    def determine_multiscale_space(self, eigenvalues, eigenvectors, n_eigs=None):
        # Perform eigen gap analysis to select eigenvectors
        n_eigs = eigenvalues.shape[-1]
        if n_eigs is None:
            vals = np.ravel(eigenvalues)
            n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-1] + 1
            if n_eigs < 3:
                n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-2] + 1

        # Select eigenvalues
        use_eigs = list(range(1, n_eigs))
        eig_vals = np.ravel(eigenvalues[use_eigs])
        # Scale the data
        scaled_eigenvectors = eigenvectors[:, use_eigs] * (eig_vals / (1 - eig_vals))
        return scaled_eigenvectors


class IterativeDiffusionMap:
    # Does not Work
    def __init__(self, iterations=10, n_components=10, **kwargs):
        self.iterations = iterations
        self.kwargs = kwargs
        self.n_components = n_components
        self.map = DiffusionMap(n_components=self.n_components, **self.kwargs)

    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("The input data must be a numpy array!")
        print(f"Running Iterative Diffusion Maps for {self.iterations} iterations")
        ev = data
        for _ in range(self.iterations):
            res = self.map(ev)
            ev = res["eigenvectors"]
        return res


class IterativeDiffusionMapv2:
    # Does not Work
    def __init__(self, inter, n_components=10, **kwargs):
        self.inter = inter
        self.kwargs = kwargs

    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("The input data must be a numpy array!")
        print(f"Running Iterative Diffusion Maps for {self.iterations} iterations")
        ev = data
        for d in self.inter:
            map_ = DiffusionMap(n_components=d, **self.kwargs)
            res = map_(ev)
            ev = res["eigenvectors"]
        return res
