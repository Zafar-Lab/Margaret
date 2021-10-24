import magic
import numpy as np
import scanpy as sc
import torch

from datasets.np import NpDataset
from models.ae import AE
from models.api import *
from utils.plot import plot_gene_expression, generate_plot_embeddings
from utils.preprocess import preprocess_recipe, run_pca
from utils.trainer import AETrainer, AEMixupTrainer
from utils.config import *


random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)


# Load data
data_path = "/home/lexent/ensemble-ti/ensemble-ti/data/marrow_sample_scseq_counts.csv"
adata = sc.read(data_path, first_column_names=True)

# Preprocessing
min_expr_level = 50
min_cells = 10
use_hvg = False
n_top_genes = 1500
preprocessed_data = preprocess_recipe(
    adata,
    min_expr_level=min_expr_level,
    min_cells=min_cells,
    use_hvg=use_hvg,
    n_top_genes=n_top_genes,
)

# Apply MAGIC on the data
magic_op = magic.MAGIC(random_state=random_seed, solver="exact")
X_magic = magic_op.fit_transform(preprocessed_data.X, genes="all_genes")
preprocessed_data.obsm["X_magic"] = X_magic

# Apply PCA
print("Computing PCA...")
_, _, n_comps = run_pca(preprocessed_data, use_hvg=use_hvg)
print(f"Components computed: {n_comps}")

# Train scvi on the data
train_scvi(adata, save_path="/home/lexent/ensemble_data/", n_epochs=400)
X_scvi = adata.obsm["X_scVI"]
preprocessed_data["X_scVI"] = X_scvi

# Apply different Non-Linear manifold learning embeddings
# Can also apply isomap, lle etc. here
embedding = Embedding(n_comps=10)
embedding.fit_transform(preprocessed_data, method="diffmap", metric="euclidean")
X_diffusion = preprocessed_data.obsm["diffusion_eigenvectors"]

embedding.fit_transform(preprocessed_data, method="lle")
X_lle = preprocessed_data.obsm["X_lle"]

embedding.fit_transform(preprocessed_data, method="isomap")
X_isomap = preprocessed_data.obsm["X_isomap"]

# Train the AutoEncoder to get the shared latent space
X_train = np.concatenate([X_diffusion, X_scvi], 1)
infeatures = X_scvi.shape[-1] + X_diffusion.shape[-1]
code_size = 10
dataset = NpDataset(X_train)
model = AE(infeatures, code_size=code_size)
train_loss = get_loss("mse")
trainer = AEMixupTrainer(dataset, model, train_loss, num_epochs=150)
trainer.train("/home/lexent/ensemble_data/ae/")

# FIXME: Refactor code to move this part in the code for AE
embedding = []
model = model.cpu()
model.eval()
with torch.no_grad():
    for data in dataset:
        embedding.append(model.encode(data.unsqueeze(0)).squeeze().numpy())
X_embedding = np.array(embedding)

# Compute the 2d embedding and plot
X_embedded = generate_plot_embeddings(X_diffusion, method="tsne", perplexity=150)
preprocessed_data.obsm["X_embedded"] = X_embedded
plot_gene_expression(
    preprocessed_data,
    genes=["CD34", "MPO", "GATA1", "IRF8"],
    figsize=(16, 4),
    cmap="plasma",
)

# Save this anndata object
preprocessed_data.write_h5ad(filename="./analysis.h5ad")
