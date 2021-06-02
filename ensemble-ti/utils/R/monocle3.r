library(ggplot2)
library(VGAM)
library(monocle3)


# Add more filepaths here based on how many datasets we want to analyze
filepaths = c(
    '/home/lexent/Downloads/disconnected_4.rds'
)

datasets = c(
    'dyntoy_disconnected_4'
)

root_dir = '/home/lexent/Desktop/monocle3_results/'
results = data.frame()


for(i in 1:length(filepaths)){
    dataset = readRDS(filepaths[i])
    start_cell_ids = dataset$prior_information$start_id
    expression_matrix = dataset$counts
    cell_metadata = dataset$cell_ids
    gene_annotation = dataset$prior_information$features_id
    
    cell_m = matrix(unlist(cell_metadata), ncol = 1, byrow = TRUE)
    row.names(cell_m) = cell_m
    
    gene_m = matrix(unlist(gene_annotation), ncol = 1, byrow = TRUE)
    row.names(gene_m) = gene_m
    
    cds = new_cell_data_set(
        t(expression_matrix),
        cell_metadata = cell_m,
        gene_metadata = gene_m
    )
#     Preprocess
    cds <- preprocess_cds(cds,norm_method='log', pseudo_count=1)
    
#     Dimensionality reduction
    cds <- reduce_dimension(cds)
    
#     Cluster generation
    cds <- cluster_cells(cds, cluster_method='louvain', verbose=TRUE, random_seed=0)
    
#     Learn trajectory
    cds <- learn_graph(cds)
    
#     Plot graph
    plot_cells(cds,
       color_cells_by = "cluster",
       label_groups_by_cluster=TRUE,
       label_leaves=FALSE,
       label_branch_points=FALSE
    )
    ggsave(paste(datasets[i], '.png', sep=""), path='/home/lexent/Desktop/monocle3_results')
    
#     Pseudotime computation
    cds = order_cells(cds, root_cells=start_cell_ids)
    plot_cells(cds,
       color_cells_by = "pseudotime",
       label_cell_groups=FALSE,
       label_leaves=FALSE,
       label_branch_points=FALSE,
       graph_label_size=1.5
    )
    ggsave(paste(datasets[i], '_pseudotime.png', sep=""), path='/home/lexent/Desktop/monocle3_results')
    
#     Metric computation (KT, SR etc.)
    monocle3_pseudotime = cds@principal_graph_aux[['UMAP']]$pseudotime
    gt_pseudotime = dataset$prior_information$timecourse_continuous
    gt_pseudotime[which(!is.finite(gt_pseudotime))] = 0
    monocle3_pseudotime[which(!is.finite(monocle3_pseudotime))] = 0
    
    gt_df = data.frame(gt_pseudotime, row.names=names(gt_pseudotime))
    m_df = data.frame(monocle3_pseudotime, row.names=names(monocle3_pseudotime))
    reorder_idx = match(rownames(m_df),rownames(gt_df))
    gt_df = data.frame(gt_df[reorder_idx,], row.names=names(monocle3_pseudotime))

    kt = cor.test(x=as.vector(t(gt_df)), y=as.vector(t(m_df)), method = 'kendall')
    sr = cor.test(x=as.vector(t(gt_df)), y=as.vector(t(m_df)), method = 'spearman')
    results[datasets[i], 'KT'] = kt$estimate
    results[datasets[i], 'SR'] = sr$estimate
}

print(results)

write.csv(results, '/home/lexent/Desktop/monocle3_results/results.csv')
