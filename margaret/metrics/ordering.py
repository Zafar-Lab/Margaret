import scipy.stats as ss


def compute_ranking_correlation(pseudotime1, pseudotime2):
    """Computes the ranking correlation between two pseudotime series.
    It is upto the user to ensure that the index of the two pseudotime series match

    Args:
        pseudotime1 ([np.ndarray, pd.Series]): Pseudotime series 1
        pseudotime2 ([np.ndarray, pd.Series]): Pseudotime series 2

    Returns:
        [dict]: A dictionary containing KT, Weighted KT and SR correlations.
    """
    kt = ss.kendalltau(pseudotime1, pseudotime2)
    weighted_kt = ss.weightedtau(pseudotime1, pseudotime2)
    sr = ss.spearmanr(pseudotime1, pseudotime2)
    return {"kendall": kt, "weighted_kendall": weighted_kt, "spearman": sr}
