import numpy as np
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets.metric import MetricDataset
from datasets.np import NpDataset
from models.metric import MetricEncoder
from utils.trainer import MetricTrainer
from utils.util import determine_cell_clusters


def train_metric_learner(
    adata, n_episodes=10, n_metric_epochs=10, code_size=10, obsm_data_key='X_pca',
    random_state=0, save_path=os.getcwd(), backend='kmeans', cluster_kwargs={}
):
    X = adata.obsm[obsm_data_key]
    clustering_scores = []
    cluster_record = []

    # Generate initial clusters
    print('Generating initial clusters')
    communities, score = determine_cell_clusters(
        adata, obsm_key=obsm_data_key, backend=backend, cluster_key='metric_clusters', **cluster_kwargs
    )
    clustering_scores.append(score)

    # Dataset
    dataset = MetricDataset(adata, obsm_data_key=obsm_data_key, obsm_cluster_key='metric_clusters')
    cluster_record.append(dataset.num_clusters)

    # Train Loss
    train_loss = nn.TripletMarginLoss()

    # Model
    infeatures = X.shape[-1]
    model = MetricEncoder(infeatures, code_size=code_size).cuda()

    # Trainer
    trainer = MetricTrainer(dataset, model, train_loss, random_state=random_state)

    for episode_idx in range(n_episodes):
        epoch_start_time = time.time()
        trainer.train(n_metric_epochs, save_path)

        # Generate embeddings
        embedding = []
        embedding_dataset = NpDataset(X)

        model.eval()
        with torch.no_grad():
            for data in tqdm(embedding_dataset):
                data = data.cuda()
                embedding.append(model(data.unsqueeze(0)).squeeze().cpu().numpy())
        X_embedding = np.array(embedding)

        adata.obsm['X_embedding'] = X_embedding

        # Generate new cluster assignments using the obtained embedding
        communities, score = determine_cell_clusters(
            adata, obsm_key='X_embedding', backend=backend, cluster_key='metric_clusters', **cluster_kwargs
        )
        clustering_scores.append(score)

        # Update the dataset
        dataset = MetricDataset(adata, obsm_data_key=obsm_data_key, obsm_cluster_key='metric_clusters')
        cluster_record.append(dataset.num_clusters)
        trainer.update_dataset(dataset)
        print(f'Time Elapsed: {time.time() - epoch_start_time}s')

    # Add the modularity score estimates to the adata
    adata.uns['clustering_scores'] = clustering_scores
    adata.uns['n_cluster_records'] = cluster_record
