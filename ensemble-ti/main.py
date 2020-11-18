import numpy as np
import scanpy as sc
import torch

from datasets.np import NpDataset
from models.vae import VAE
from util.preprocess import preprocess_recipe, run_pca
from util.criterion import VAELoss
from util.trainer import VAETrainer


random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)


# Load data
data_path = '/home/lexent/ensemble-ti/ensemble-ti/data/marrow_sample_scseq_counts.csv'
data = sc.read(data_path, first_column_names=True)

# Preprocess data
min_expr_level = 1000
min_cells = 10
use_hvg = True
n_top_genes = 1500
preprocessed_data = preprocess_recipe(
    data, 
    min_expr_level=min_expr_level, 
    min_cells=min_cells, 
    use_hvg=use_hvg, 
    n_top_genes=n_top_genes
)

# Apply PCA as a preprocessing step
variance = 0.85
X_pca, variance = run_pca(preprocessed_data, use_hvg=True, variance=variance)

# Train a VAE on the input data
train_dataset = NpDataset(X_pca)
save_path = '/home/lexent/ensemble_data/'
train_loss = VAELoss(use_bce=False)
eval_loss = None
in_features = X_pca.shape[1]
code_size = 50
batch_size = 32
lr = 0.0001
optim = 'Adam'
epochs = 100
model = VAE(in_features, code_size=code_size)

trainer = VAETrainer(
    train_dataset, model, train_loss,
    random_state=random_seed, batch_size=batch_size,
    lr=lr, optimizer=optim, num_epochs=epochs
)

trainer.train()
