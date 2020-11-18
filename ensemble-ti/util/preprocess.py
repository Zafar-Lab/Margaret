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
    sc.pp.normalize_total(data)
    log_transform(data)
    print('\t->Normalized data')

    # Select highly variable genes (hvg)
    if use_hvg:
        sc.pp.highly_variable_genes(data, n_top_genes=n_top_genes, flavor='cell_ranger', subset=True)
        # Re-Normalization to account for hvg selection
        sc.pp.normalize_total(data)
        log_transform(data)
        print(f'\t->Selected the top {n_top_genes} genes and re-normalized')
    
    sc.pp.scale(data)
    print(f'\t->Scaled features to have zero mean and unit variance')
    print(f'Pre-processing complete. Updated data shape: {data.shape}')
    return data


def log_transform(data, pseudo_count=0.1):
    # Code taken from Palantir
    if type(data) is sc.AnnData:
        data.X = np.log2(data.X + pseudo_count) - np.log2(pseudo_count)
    else:
        return np.log2(data + pseudo_count)
