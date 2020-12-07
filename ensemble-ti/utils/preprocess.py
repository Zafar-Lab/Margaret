import numpy as np
import scanpy as sc


def preprocess_recipe(adata, min_expr_level=50, min_cells=10, use_hvg=True, n_top_genes=1500):
    preprocessed_data = adata.copy()
    print('Preprocessing....')
    sc.pp.filter_cells(preprocessed_data, min_counts=min_expr_level)
    print(f'\t->Removed cells with expression level<{min_expr_level}')

    sc.pp.filter_genes(preprocessed_data, min_cells=min_cells)
    print(f'\t->Removed genes expressed in <{min_cells} cells')

    sc.pp.normalize_total(preprocessed_data)
    log_transform(preprocessed_data)
    print('\t->Normalized data')

    if use_hvg:
        sc.pp.highly_variable_genes(preprocessed_data, n_top_genes=n_top_genes, flavor='cell_ranger')
        print(f'\t->Selected the top {n_top_genes} genes')
    print(f'Pre-processing complete. Updated data shape: {preprocessed_data.shape}')
    return preprocessed_data


def log_transform(data, pseudo_count=0.1):
    # Code taken from Palantir
    if type(data) is sc.AnnData:
        data.X = np.log2(data.X + pseudo_count) - np.log2(pseudo_count)
    else:
        return np.log2(data + pseudo_count)


def run_pca(data, n_components=300, use_hvg=True, variance=0.85):
    if type(data) is sc.AnnData:
        ad = data
    else:
        ad = sc.AnnData(data.values)

    # Run PCA
    if not use_hvg:
        n_comps = n_components
    else:
        sc.pp.pca(ad, n_comps=1000, use_highly_variable=True, zero_center=False)
        try:
            n_comps = np.where(np.cumsum(ad.uns['pca']['variance_ratio']) > variance)[0][0]
        except IndexError:
            n_comps = n_components

    # Re-run with selected number of components (Either n_comps=n_components or
    # n_comps = minimum number of components required to explain variance)
    sc.pp.pca(ad, n_comps=n_comps, use_highly_variable=use_hvg, zero_center=False)

    # Return PCA projections if it is a dataframe
    X_pca = ad.obsm['X_pca']
    return X_pca, ad.uns['pca']['variance_ratio'], n_comps
