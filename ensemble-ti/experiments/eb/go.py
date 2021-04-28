import os
import numpy as np
import pandas as pd

from gprofiler import GProfiler


# Create a GOProfiler object
gp = GProfiler(return_dataframe=True)


def generate_go_terms(
    ad, de_key='rank_genes_groups', clusters_key='metric_clusters', lfc_cutoff=1.0, pval_cutoff=0.05,
    n_top=500, go_clusters=None, save_dir=None, **kwargs
):
    de_res = ad.uns[de_key]
    communities = ad.obs[clusters_key]
    cluster_ids = np.unique(communities)
    query_cluster_ids = cluster_ids if go_clusters is None else go_clusters

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for idx in query_cluster_ids:
        print(f'Computing GO terms for cluster: {idx}')

        # Read DE results
        names = [res[idx] for res in de_res['names']]
        scores = pd.Series([res[idx] for res in de_res['scores']], index=names)
        lfc_vals = pd.Series([res[idx] for res in de_res['logfoldchanges']], index=names)
        adjp_vals = pd.Series([res[idx] for res in de_res['pvals_adj']], index=names)
        gene_index = lfc_vals.index

        # Keep values above threshold
        valid_lfc_inds = lfc_vals >= lfc_cutoff
        valid_pval_inds = adjp_vals <= pval_cutoff
        mask = valid_lfc_inds.multiply(valid_pval_inds)

        # Compute filtered genes
        remaining_genes = gene_index[mask]

        # Take the top genes from the filtered pool based on scores
        filtered_scores = scores.loc[remaining_genes]
        scores_ = filtered_scores.sort_values(ascending=False).iloc[:n_top]
        remaining_genes = scores_.index

        # GO query
        go_df = gp.profile(organism='hsapiens', query=list(remaining_genes), **kwargs)
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"GO_{idx}.csv")
            go_df.to_csv(save_path, index=False)
            print(f'GO terms for cluster: {idx} written at: {save_path}')
