# CREDITS: Adapted for graphs with unequal number of nodes from NetRD: https://netrd.readthedocs.io/en/latest/index.html


import numpy as np
import networkx as nx
import warnings

from functools import wraps
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from scipy.integrate import quad


def ensure_undirected(G):
    """Ensure the graph G is undirected.

    If it is not, coerce it to undirected and warn the user.

    Parameters
    ----------
    G (networkx graph)
        The graph to be checked

    Returns
    -------

    G (nx.Graph)
        Undirected version of the input graph

    """
    if nx.is_directed(G):
        G = G.to_undirected(as_view=False)
        warnings.warn("Coercing directed graph to undirected.", RuntimeWarning)
    return G


def undirected(func):
    """
    Decorator applying ``ensure_undirected()`` to all ``nx.Graph``-subclassed
    arguments of ``func``.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [
            ensure_undirected(arg) if issubclass(arg.__class__, nx.Graph) else arg
            for arg in args
        ]
        return func(*args, **kwargs)

    return wrapper


def ensure_unweighted(G):
    """Ensure the graph G is unweighted.

    If it is not, coerce it to unweighted and warn the user.

    Parameters
    ----------
    G (networkx graph)
        The graph to be checked

    Returns
    -------

    G (nx.Graph)
        Unweighted version of the input graph

    """

    for _, _, attr in G.edges(data=True):
        if not np.isclose(attr.get("weight", 1.0), 1.0):
            H = G.__class__()
            H.add_nodes_from(G)
            H.add_edges_from(G.edges)
            warnings.warn("Coercing weighted graph to unweighted.", RuntimeWarning)
            return H

    return G


def unweighted(func):
    """
    Decorator applying ``ensure_unweighted()`` to all ``nx.Graph``-subclassed
    arguments of ``func``.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [
            ensure_unweighted(arg) if issubclass(arg.__class__, nx.Graph) else arg
            for arg in args
        ]
        return func(*args, **kwargs)

    return wrapper


class BaseDistance:
    """Base class for all distance algorithms.

    The basic usage of a distance algorithm is as follows:

    >>> dist_obj = DistanceAlgorithm()
    >>> distance = dist_obj.dist(G1, G2, <some_params>)
    >>> # or alternatively: distance = dist_obj.results['dist']

    Here, `G1` and `G2` are ``nx.Graph`` objects (or subclasses such as
    ``nx.DiGraph``). The results dictionary holds the distance value, as
    well as any other values that were computed as a side effect.

    """

    def __init__(self):
        self.results = {}

    def __call__(self, *args, **kwargs):
        return self.dist(*args, **kwargs)

    def dist(self, G1, G2):
        """Compute distance between two graphs.

        Values computed as side effects of the distance method can be foun
        in self.results.

        Parameters
        ----------

        G1, G2 (nx.Graph): two graphs.

        Returns
        -----------

        distance (float).

        """
        dist = -1  # compute the distance
        self.results["dist"] = dist  # store dist in self.results
        # self.results[..] = ..     # also store other values if needed
        return dist  # return only one value!


class IpsenMikhailov(BaseDistance):
    """Compares the spectrum of the Laplacian matrices."""

    @unweighted
    def dist(self, G1, G2, hwhm=0.08):
        """Compare the spectrum ot the associated Laplacian matrices.

        The results dictionary also stores a 2-tuple of the underlying
        adjacency matrices in the key `'adjacency_matrices'`.

        Parameters
        ----------

        G1, G2 (nx.Graph)
            two networkx graphs to be compared.

        hwhm (float)
            half with at half maximum of the lorentzian kernel.

        Returns
        -------

        dist (float)
            the distance between G1 and G2.

        Notes
        -----

        Requires undirected networks.

        References
        ----------

        .. [1] https://journals.aps.org/pre/abstract/10.1103/PhysRevE.66.046109

        """
        # get the adjacency matrices
        adj1 = nx.to_numpy_array(G1)
        adj2 = nx.to_numpy_array(G2)
        self.results["adjacency_matrices"] = adj1, adj2

        # get the IM distance
        dist = _im_distance(adj1, adj2, hwhm)

        self.results["dist"] = dist

        return dist


def _im_distance(adj1, adj2, hwhm):
    """Computes the Ipsen-Mikhailov distance for two symmetric adjacency
    matrices

    Base on this paper :
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.66.046109

    Note : this is also used by the file hamming_ipsen_mikhailov.py

    Parameters
    ----------

    adj1, adj2 (array): adjacency matrices.

    hwhm (float) : hwhm of the lorentzian distribution.

    Returns
    -------

    dist (float) : Ipsen-Mikhailov distance.

    """
    N_1 = len(adj1)
    N_2 = len(adj2)

    # get laplacian matrix
    L1 = laplacian(adj1, normed=False)
    L2 = laplacian(adj2, normed=False)

    # get the modes for the positive-semidefinite laplacian
    w1 = np.sqrt(np.abs(eigh(L1)[0][1:]))
    w2 = np.sqrt(np.abs(eigh(L2)[0][1:]))

    # we calculate the norm for both spectrum
    norm1 = (N_1 - 1) * np.pi / 2 - np.sum(np.arctan(-w1 / hwhm))
    norm2 = (N_2 - 1) * np.pi / 2 - np.sum(np.arctan(-w2 / hwhm))

    # define both spectral densities
    density1 = lambda w: np.sum(hwhm / ((w - w1) ** 2 + hwhm ** 2)) / norm1
    density2 = lambda w: np.sum(hwhm / ((w - w2) ** 2 + hwhm ** 2)) / norm2

    func = lambda w: (density1(w) - density2(w)) ** 2

    return np.sqrt(quad(func, 0, np.inf, limit=100)[0])


if __name__ == "__main__":
    g1 = nx.fast_gnp_random_graph(10, 0.6)
    g2 = nx.fast_gnp_random_graph(5, 0.4)
    im = IpsenMikhailov()
    dist1 = im(g1, g2)
    dist2 = im(g2, g1)
    assert dist1 == dist2
