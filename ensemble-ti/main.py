import numpy as np
import scanpy as sc
import torch
import torchvision
import torchvision.transforms as T

from datasets.np import NpDataset
from models.vae import VAE
from models.dae import DAE
from util.preprocess import preprocess_recipe, run_pca
from util.criterion import VAELoss
from util.trainer import VAETrainer, DAETrainer
from util.config import *


random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)


# Load data
# data_path = '/home/lexent/ensemble-ti/ensemble-ti/data/marrow_sample_scseq_counts.csv'
# data = sc.read(data_path, first_column_names=True)

# # Preprocess data
# min_expr_level = 1000
# min_cells = 50
# use_hvg = True
# n_top_genes = 500
# preprocessed_data = preprocess_recipe(
#     data, 
#     min_expr_level=min_expr_level, 
#     min_cells=min_cells,
#     use_hvg=use_hvg,
#     n_top_genes=n_top_genes
# )

# Apply PCA as a preprocessing step
# variance = 0.85
# X_pca, variance = run_pca(preprocessed_data, use_hvg=True, variance=variance)

# Train a VAE on the input data
# X_pca = preprocessed_data.X
# train_dataset = NpDataset(X_pca)
transforms = T.Compose([
    T.ToTensor()
])
mnist_data = torchvision.datasets.MNIST('data/', transform=transforms, download=True)
save_path = '/home/lexent/ensemble_data/'
train_loss = VAELoss(use_bce=True)
dae_train_loss = get_loss('mse')
eval_loss = None
# in_features = X_pca.shape[1]
in_features = 784
code_size = 5
batch_size = 128
lr = 0.001
optim = 'Adam'
# optimizer_kwargs = {
    # 'momentum': 0.9
# }

epochs = 200
model = VAE(in_features, code_size=code_size)
dae = DAE(in_features, code_size=code_size)

trainer = VAETrainer(
    mnist_data, model, train_loss,
    random_state=random_seed, batch_size=batch_size,
    lr=lr, optimizer=optim, num_epochs=epochs, 
    # optimizer_kwargs=optimizer_kwargs
)

trainer_2 = DAETrainer(
    mnist_data, dae, dae_train_loss,
    random_state=random_seed, batch_size=batch_size,
    lr=lr, optimizer=optim, num_epochs=epochs,
    # optimizer_kwargs=optimizer_kwargs
)

trainer.train(save_path)
