library(reticulate)

py_anndata = import("anndata", convert=FALSE)

args = commandArgs(trailingOnly=TRUE)
if(length(args) == 0 || length(args) == 1) {
    stop("Read and write paths must be provided")
}
read_path = args[1]
write_path = args[2]
dataset = readRDS(read_path)

# Raw counts
counts = dataset$counts

# Obs and Var names
cell_ids = dataset$cell_ids
feature_ids = dataset$prior_information$features_id

# Timecourse (pseudotime) information
pt = dataset$prior_information$timecourse_continuous

# Unstructured info (like prior knowledge of start cells etc.)
uns_info = dict(
    milestone_percentages = dataset$milestone_percentages,
    milestone_network = dataset$milestone_network,
    start_id = dataset$prior_information$start_id,
    start_milestones = dataset$prior_information$start_milestones,
    end_milestones = dataset$prior_information$end_milestones,
    end_id = dataset$prior_information$end_id,
    timecourse = data.frame(pt, row.names=names(pt))
)

# Construct the AnnData object
ad = py_anndata$AnnData(X=counts, uns=uns_info)
ad$obs_names = cell_ids
ad$var_names = feature_ids

# Write the AnnData object (loom format)
ad$write(write_path)

print("AnnData file written successfully!")
