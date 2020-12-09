import phenograph
import torch.nn as nn

from datasets.metric import MetricDataset
from models.metric import MetricEncoder
from utils.trainer import MetricTrainer


def train_metric_learner(adata, num_epochs=10, code_size=10, obsm_data_key='X_pca', knn=50):
    X = adata.obsm[obsm_data_key]
    print('Generating initial clusters..')
    communities, _, _ = phenograph.cluster(X, k=knn)
    adata.obsm['phenograph_communities'] = communities

    # Train params
    dataset = MetricDataset(adata, obsm_data_key=obsm_data_key)
    train_loss = nn.TripletMarginLoss()
    infeatures = X.shape[-1]
    model = MetricEncoder(infeatures, code_size=code_size).cuda()
    trainer = MetricTrainer(dataset, model, train_loss)

    for epoch_idx in range(num_epochs):
        print(f'Training for epoch : {epoch_idx + 1}')
        trainer.train(10, '/content/metric/')

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
        communities, _, _ = phenograph.cluster(X_embedding, k=knn)
        adata.obsm['phenograph_communities'] = communities

        # Update the dataset
        dataset = MetricDataset(preprocessed_data, obsm_data_key=obsm_data_key)
        trainer.update_dataset(dataset)
