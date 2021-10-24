import scipy.stats as ss


def compute_ranking_correlation(pseudotime1, pseudotime2):
    kt = ss.kendalltau(pseudotime1, pseudotime2)
    weighted_kt = ss.weightedtau(pseudotime1, pseudotime2)
    sr = ss.spearmanr(pseudotime1, pseudotime2)
    return {"kendall": kt, "weighted_kendall": weighted_kt, "spearman": sr}
