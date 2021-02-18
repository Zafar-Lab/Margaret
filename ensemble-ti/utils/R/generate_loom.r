library(anndata)

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

# Unstructured info (like prior knowledge of start cells etc.)
uns_info = list(
    milestone_percentages = dataset$milestone_percentages,
    milestone_network = dataset$milestone_network,
    start_id = dataset$prior_information$start_id,
    start_milestones = dataset$prior_information$start_milestones,
    end_milestones = dataset$prior_information$end_milestones,
    end_id = dataset$prior_information$end_id
)

# Construct the AnnData object
ad = AnnData(X=counts)
ad$obs_names = cell_ids
ad$var_names = feature_ids
ad$uns = uns_info

# Write the AnnData object (loom format)
write_loom(ad, write_path)

print("AnnData file written successfully!")
