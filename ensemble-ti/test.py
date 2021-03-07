import matplotlib.pyplot as plt
import numpy as np
# import scanpy as sc
# import scanpy.external as sce
# import torch
# import torchvision
# import torchvision.transforms as T
# import scvi

# from sklearn.manifold import TSNE
# # from dca.api import dca

# from datasets.np import NpDataset
# # from models.vae import VAE
# # from models.dae import DAE
# from utils.preprocess import preprocess_recipe, run_pca

import scanpy as sc

from utils.plot import plot_gt_milestone_network

# random_seed = 0
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)


# Load data
data_path = '/home/lexent/Desktop/dyntoy_disconnected_gen_1.h5ad'
adata = sc.read(data_path)

# adata = anndata.read_h5ad(path_to_anndata)
# scvi.data.setup_anndata(adata)
# vae = scvi.model.SCVI(adata)
# vae.train()
# adata.obsm["X_scVI"] = vae.get_latent_representation()
# adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()
# print(adata.obsm["X_scVI"])


# # Preprocessing and MAGIC Denoising
# min_expr_level = 1000
# min_cells = 10
# use_hvg = True
# n_top_genes = 1500
# preprocessed_data = preprocess_recipe(
#     data, 
#     min_expr_level=min_expr_level, 
#     min_cells=min_cells,
#     use_hvg=use_hvg,
#     n_top_genes=n_top_genes
# )

# print('Computing PCA...')
# X_pca, _, n_comps = run_pca(preprocessed_data, use_hvg=use_hvg)
# print(f'PCA computed: {X_pca.shape}')

# dca(adata, mode='latent')

# # Perform MAGIC Imputation
# print('Resolving Dropouts. Magic Imputation')
# sce.pp.magic(preprocessed_data, random_state=random_seed, n_pca=300, name_list='pca_only')

# X_magic = preprocessed_data.obsm['X_magic']
# # X_magic = preprocessed_data.X
# print(X_magic.shape)

# X_train = X_magic
# train_dataset = NpDataset(X_train)
# transforms = T.Compose([
#     T.ToTensor()
# ])

# state_dict = torch.load('/home/lexent/ensemble_data/chkpt_140.pt')

# model = VAE(X_train.shape[-1], code_size=5)
# model.load_state_dict(state_dict['model'])
# model.eval()
# embeddings = []
# with torch.no_grad():
#     for data in train_dataset:
#         embedding, _, _, _ = model(data.unsqueeze(0))
#         embeddings.append(embedding.squeeze().numpy())

# embeddings = np.array(embeddings)

# X_embedded = TSNE(n_components=2, perplexity=150).fit_transform(adata.obsm["X_scVI"])
# plt.scatter(embeddings[:, 0], embeddings[:, 1])
# plt.show()

plot_gt_milestone_network(adata)
plt.show()
