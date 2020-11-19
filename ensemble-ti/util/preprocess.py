import numpy as np
import scanpy as sc


def preprocess_recipe(data, min_expr_level=1000, min_cells=10, use_hvg=True, n_top_genes=1500):
    """A collection of preprocessing steps for this project

    Args:
        data ([anndata]): Takes a numpy array as input
        min_cells([int]): Minimum number of cells in which a gene must be expressed in
        min_counts([int]): Minimum expression level needed for each cell
    """
    print('Preprocessing....')
    # Remove cells with low library size (maybe dead cells)
    sc.pp.filter_cells(data, min_counts=min_expr_level)
    print(f'\t->Removed cells with expression level<{min_expr_level}')

    # Filter genes which do not express in enough cells
    sc.pp.filter_genes(data, min_cells=min_cells)
    print(f'\t->Removed genes expressed in <{min_cells} cells')

    # Library size normalization and Log counts
    # sc.pp.normalize_total(data)
    # log_transform(data)
    print('\t->Normalized data')

    # Select highly variable genes (hvg)
    if use_hvg:
        sc.pp.highly_variable_genes(data, n_top_genes=n_top_genes, flavor='seurat_v3', subset=True)
        # Re-Normalization to account for hvg selection
        sc.pp.normalize_total(data)
        # log_transform(data)
        print(f'\t->Selected the top {n_top_genes} genes and re-normalized')
    
    # sc.pp.scale(data)
    # print(f'\t->Scaled features to have zero mean and unit variance')
    print(f'Pre-processing complete. Updated data shape: {data.shape}')
    return data


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
    return X_pca, ad.uns['pca']['variance_ratio']
