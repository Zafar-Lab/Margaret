import numpy as np
import phenograph
import time
import torch
import torch.nn as nn

from datasets.metric import MetricDataset
from models.metric import MetricEncoder
from utils.trainer import MetricTrainer


def train_metric_learner(
    adata, n_episodes=10, n_metric_epochs=20, code_size=10, obsm_data_key='X_pca',
    knn=50, random_state=0, save_path=os.getcwd(), **kwargs
):
    X = adata.obsm[obsm_data_key]
    modularity_scores = []
    cluster_record = []

    communities, _, Q0 = phenograph.cluster(X, k=knn, seed=random_state, **kwargs)
    modularity_scores.append(Q0)
    adata.obsm['phenograph_communities'] = communities

    # Dataset
    dataset = MetricDataset(adata, obsm_data_key=obsm_data_key)
    cluster_record.append(dataset.num_clusters)

    # Train Loss
    train_loss = nn.TripletMarginLoss()

    # Model
    infeatures = X.shape[-1]
    model = MetricEncoder(infeatures, code_size=code_size).cuda()

    # Trainer
    trainer = MetricTrainer(dataset, model, train_loss)

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

        # Generate new cluster assignments
        communities, _, Q = phenograph.cluster(X_embedding, k=knn, seed=random_state, **kwargs)
        adata.obsm['phenograph_communities'] = communities
        modularity_scores.append(Q)

        # Update the dataset
        dataset = MetricDataset(preprocessed_data, obsm_data_key=obsm_data_key)
        cluster_record.append(dataset.num_clusters)
        trainer.update_dataset(dataset)
        print(f'Time Elapsed: {time.time() - epoch_start_time}s')

    # Add the modularity score estimates to the adata
    adata.uns['mod_scores'] = modularity_scores
    adata.uns['cluster_records'] = cluster_record
