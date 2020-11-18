import numpy as np
import scanpy as sc

from util.preprocess import preprocess_recipe


random_seed = 0

np.random.seed(random_seed)

# Load data
data = sc.read('/home/lexent/ensemble-ti/ensemble-ti/data/marrow_sample_scseq_counts.csv', first_column_names=True)

# Preprocess data
preprocessed_data = preprocess_recipe(data)
