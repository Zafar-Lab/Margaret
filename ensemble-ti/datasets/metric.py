import numpy as np
import pandas as pd
import scanpy as sc
import torch

from torch.utils.data import Dataset


# TODO: Support testing triplets
class MetricDataset(Dataset):
    def __init__(self, data, obsm_cluster_key='phenograph_communities', obsm_data_key='X_pca', transform=None):
        if not isinstance(data, sc.AnnData):
            raise Exception(f'Expected data to be of type sc.AnnData found : {type(data)}')
        self.data = data
        try:
            self.cluster_inds = data.obs[obsm_cluster_key]
        except KeyError:
            raise Exception(f'`{obsm_cluster_key}` must be set in the data')
        self.X = self.data.obsm[obsm_data_key]
        self.unique_clusters = np.unique(self.cluster_inds)

        self.indices = np.arange(self.data.shape[0])
        self.num_clusters = len(self.unique_clusters)
        self.transform = transform

    def __getitem__(self, idx):
        # Sample the anchor and the positive class
        anchor, pos_class = torch.Tensor(self.X[idx]), self.cluster_inds[idx]
        positive_idx = idx
        
        while positive_idx == idx:
            positive_idx = np.random.choice(self.indices[self.cluster_inds == pos_class])
        pos_sample = self.X[positive_idx]

        # Sample the negative label and sample
        neg_class = np.random.choice(list(set(self.unique_clusters) - set([pos_class])))
        neg_class_choices = self.cluster_inds == neg_class
        neg_sample = self.X[np.random.choice(self.indices[neg_class_choices])]

        if self.transform is not None:
            anchor = self.transform(anchor)
            pos_sample = self.transform(pos_sample)
            neg_sample = self.transform(neg_sample)
        return anchor, pos_sample, neg_sample

    def __len__(self):
        return self.data.shape[0]
