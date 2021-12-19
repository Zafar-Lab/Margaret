import torch
from torch.utils.data import Dataset


class NpDataset(Dataset):
    def __init__(self, X):
        """Creates a torch dataset from  a numpy array

        Args:
            X ([np.ndarray]): Numpy array which needs to be mapped as a tensor
        """
        self.X = torch.Tensor(X)
        self.shape = self.X.shape

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return self.X.shape[0]
