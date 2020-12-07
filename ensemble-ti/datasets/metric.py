import numpy as np
import pandas as pd
import scanpy as sc

from torch.utils.data import Dataset


class MetricDataset(Dataset):
    def __init__(self, data, obsm_key='phenograph_communities', transform=None):
        if not isinstance(data, sc.AnnData):
            raise Exception(f'Expected data to be of type sc.AnnData found : {type(data)}')
        self.data = data
        try:
            self.cluster_inds = data.obsm[obsm_key]
        except KeyError:
            raise Exception(f'`{obsm_key}` must be set in the data')
        unique_clusters = np.unique(self.cluster_inds)

        self.indices = np.arange(self.data.shape[0])
        self.min_cluster_idx = unique_clusters[0]
        self.max_cluster_idx = unique_clusters[-1]
        self.num_clusters = len(unique_clusters)
        self.transform = transform

    def __getitem__(self, idx):
        # Sample two classes
        cls1 = np.random.randint(self.min_cluster_idx, self.max_cluster_idx + 1)
        cls2 = np.random.randint(self.min_cluster_idx, self.max_cluster_idx + 1)
        
        # Sample until both classes are different
        while (cls1 == cls2):
            cls2 = np.random.randint(self.min_cluster_idx, self.max_cluster_idx + 1)
        
        cls1_indices = self.cluster_inds == cls1
        cls2_indices = self.cluster_inds == cls2

        # Sample Anchor, positive and negative class samples
        anchor, pos_sample = self.data.X[np.random.choice(self.indices[cls2_indices]), size=2]
        neg_sample = self.data.X[np.random.choice(self.indices[cls2_indices])]

        if self.transform is not None:
            anchor = self.transform(anchor)
            pos_sample = self.transform(pos_sample)
            neg_sample = self.transform(neg_sample)
        return anchor, pos_sample, neg_sample

    def __len__(self):
        return self.data.shape[0]
