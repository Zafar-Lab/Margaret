from numba import jit
import numpy as np
import torch
import torch.nn as nn


class VAELoss(nn.Module):
    def __init__(self, reduction="mean", alpha=1.0, use_bce=True):
        super(VAELoss, self).__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError("Valid values for the reduction param are `mean`, `sum`")
        self.alpha = alpha
        self.use_bce = use_bce
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction="mean")
        self.bce = nn.BCELoss(reduction="mean")

    def forward(self, x, decoder_out, mu, logvar):
        # Reconstruction Loss:
        # TODO: Try out the bce loss for reconstruction
        if self.use_bce:
            reconstruction_loss = self.bce(decoder_out, x)
        else:
            reconstruction_loss = self.mse(decoder_out, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + self.alpha * kl_loss
        if self.reduction == "mean":
            return torch.mean(loss)
        else:
            return loss


@jit(nopython=True)
def fractional_norm(x, y, order=0.5):
    return np.sum(np.power(np.abs(x - y), order)) ** (1 / order)


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (
            (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        )  # .pow(.5)
        an_distances = (
            (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        )  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
