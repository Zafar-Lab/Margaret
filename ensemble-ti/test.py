import numpy as np
import scanpy as sc
import torch
import torchvision
import torchvision.transforms as T

from datasets.np import NpDataset
from models.vae import VAE
from models.dae import DAE
from util.preprocess import preprocess_recipe, run_pca
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)


# Load data
data_path = '/home/lexent/ensemble-ti/ensemble-ti/data/marrow_sample_scseq_counts.csv'
data = sc.read(data_path, first_column_names=True)

# Preprocess data
min_expr_level = 1000
min_cells = 50
use_hvg = True
n_top_genes = 500
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
X_pca = preprocessed_data.X
# Train a VAE on the input data
# train_dataset = NpDataset(X_pca)

transforms = T.Compose([
    T.ToTensor()
])
mnist_data = torchvision.datasets.MNIST('data/', transform=transforms, download=True)

state_dict = torch.load('/home/lexent/ensemble_data/chkpt_20.pt')

model = VAE(784, code_size=5)
model.load_state_dict(state_dict['model'])
model.eval()
embeddings = []
with torch.no_grad():
    for data, _ in mnist_data:
        data = data.reshape(784, )
        embedding = model.encode(data.unsqueeze(0))
        embeddings.append(embedding.squeeze().numpy())

embeddings = np.array(embeddings)

# X_embedded = TSNE(n_components=2).fit_transform(embeddings)
plt.scatter(embeddings[:, 0], embeddings[:, 1])
plt.show()
