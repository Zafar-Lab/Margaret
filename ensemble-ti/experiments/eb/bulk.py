import mygene
import numpy as np
import pandas as pd
import scipy.stats as ss

from tqdm import tqdm


def compute_bulk_correlations(ad, bulk_expr_path, mapping_file_path, sc_gene_list=None):
    ad.var_names_make_unique()

    # Read the mapping file which maps Ensembl gene IDs to symbol IDs
    mapping_df = pd.read_csv(mapping_file_path)
    mapping_df.index = mapping_df['EnsemblId']
    mapping_df = mapping_df.drop_duplicates(subset='SymbolId')

    # Read the bulk-expression data
    bulk_expr_df = pd.read_csv(bulk_expr_path)
    bulk_expr_df.index = bulk_expr_df['name']

    # Get the bulk expression values and gene IDs for all mapped genes
    bulk_expr_vals = bulk_expr_df.loc[mapping_df.index]['rpkm_900']
    bulk_expr_genes = mapping_df.loc[mapping_df.index]['SymbolId']
    bulk_expr_vals.index = bulk_expr_genes

    sc_expr_genes = []
    if sc_gene_list is None:
        sc_expr_genes = ad.var_names
    # Compute the set of genes which are common in bulk and scRNA data
    common_genes = list(set(sc_expr_genes).intersection(set(bulk_expr_genes)))

    # Compute the correlation of the expression of each cell with the bulk expr data
    p = []
    common_bulk_expr_val = bulk_expr_vals.loc[common_genes]
    ad_df = ad.to_df()
    for cell in tqdm(ad.obs_names):
        sc_expr_val = ad_df.loc[cell, common_genes]
        p.append(ss.pearsonr(common_bulk_expr_val, sc_expr_val)[0])
    return p


def generate_mapping_file(bulk_expr_path, mapping_path):
    bulk_df = pd.read_csv(bulk_expr_path)
    mg = mygene.MyGeneInfo()
    notfound = []
    expr_dict = {}

    # Create the Ensembl ID to rpkm value mapping (excluding the file headers)
    for ensembl_gene_id, rpkm_val in zip(bulk_df['name'][1:], bulk_df['rpkm_1'][1:]):
        expr_dict[ensembl_gene_id] = rpkm_val

    # Get the gene IDs for Ensembl gene IDs
    ensembl_gene_list = list(expr_dict.keys())
    results = mg.querymany(ensembl_gene_list, scopes='ensembl.gene', fields='symbol')

    queries = []
    symbols = []
    for result in results:
        query = result['query']
        symbol = result.get('symbol', None)
        # If symbol for a query was not found, skip row
        if symbol is None:
            notfound.append(query)
            continue
        queries.append(query)
        symbols.append(symbol)

    # Create mapping file
    df_ = pd.DataFrame(symbols, columns=['SymbolId'], index=queries)
    df_.to_csv(mapping_path)

    print('Mapping file generation complete. Generated file path: {mapping_path}.')
    print('The symbol IDs for the following Ensembl Ids were not found: {notfound}')
    return df_, notfound
