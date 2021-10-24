import numpy as np
import scanpy as sc
import scvi

from models.diffmap import DiffusionMap
from sklearn.manifold import LocallyLinearEmbedding, Isomap


def train_scvi(adata, save_path=None, model_kwargs={}, **train_kwargs):
    """Runs the scVI model on the annotated dataset
    with the raw counts. This model requires raw
    counts matrix as input
    """
    if not isinstance(adata, sc.AnnData):
        raise ValueError(f"Expected data to be of type sc.AnnData got {type(adata)}")
    scvi.data.setup_anndata(adata)
    vae = scvi.model.SCVI(adata, **model_kwargs)
    vae.train(**train_kwargs)
    adata.obsm["X_scVI"] = vae.get_latent_representation()
    adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()

    if save_path is not None:
        vae.save(save_path, overwrite=True)


def train_dca(adata, save_path=None, return_model=True, **kwargs):
    from dca.api import dca

    # The dca training does not work as their github code is broken
    # TODO: We might need to write our own implementation of DCA
    # Confirm with the project lead.
    if not isinstance(adata, sc.AnnData):
        raise ValueError(f"Expected data to be of type sc.AnnData got {type(adata)}")
    anndata, model = dca(adata, mode="latent", **kwargs)


class Embedding:
    def __init__(self, random_state=0, **kwargs):
        self.kwargs = kwargs
        self.random_state = random_state

    def fit_transform(
        self, adata, obsm_key="X_pca", backend="diffmap", n_comps=10, **kwargs
    ):
        if not isinstance(adata, sc.AnnData):
            raise ValueError(
                f"Expected data to be of type sc.AnnData got {type(adata)}"
            )
        try:
            X = adata.obsm[obsm_key]
        except KeyError:
            raise Exception(
                f"{obsm_key} must be present before computing the embedding"
            )

        if backend == "diffmap":
            diffmap = DiffusionMap(
                n_components=n_comps, random_state=self.random_state, **kwargs
            )
            res = diffmap(X)
            eigenvectors = diffmap.determine_multiscale_space(
                res["eigenvalues"], res["eigenvectors"]
            )
            adata.obsm["diffusion_T"] = res["T"]
            adata.obsm["X_diffmap"] = eigenvectors
            adata.uns["diffusion_eigenvalues"] = res["eigenvalues"]
            adata.obsm["diffusion_kernel"] = res["kernel"]
        elif backend == "lle":
            lle = LocallyLinearEmbedding(
                n_components=n_comps, random_state=self.random_state, **kwargs
            )
            X_lle = lle.fit_transform(X)
            adata.obsm["X_lle"] = X_lle
        elif backend == "isomap":
            isomap = Isomap(n_components=n_comps, **kwargs)
            X_isomap = isomap.fit_transform(X)
            adata.obsm["X_isomap"] = X_isomap
        else:
            raise ValueError(f"Unsupported method param value: {backend}")
