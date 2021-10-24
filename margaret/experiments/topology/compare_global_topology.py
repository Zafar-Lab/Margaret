import csv
import os
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

from IPython.display import clear_output

from core import run_metti, run_paga, run_palantir, run_metti_v2
from metrics.ipsen import IpsenMikhailov
from metrics.ordering import compute_ranking_correlation
from models.ti.graph import compute_gt_milestone_network
from utils.util import (
    log_transform,
    preprocess_recipe,
    run_pca,
    get_start_cell_cluster_id,
)
from utils.plot import (
    plot_embeddings,
    plot_clusters,
    plot_connectivity_graph,
    plot_trajectory_graph,
    plot_pseudotime,
    generate_plot_embeddings,
)


def evaluate_metric_topology(
    dataset_file_path,
    results_dir=os.getcwd(),
    resolutions=[0.4, 0.6, 0.8, 1.0],
    c_backends=["louvain", "leiden"],
    threshold=0.5,
    dry_run=False,
    device="cuda",
    metti_version="v2",
):
    # Read the dataset file
    datasets = {}
    with open(dataset_file_path, "r") as fp:
        reader = csv.DictReader(fp)
        datasets = {row["name"]: row["path"] for row in reader}

    for backend in c_backends:
        r = pd.DataFrame(index=datasets.keys())
        for name, path in datasets.items():
            # Setup directory per dataset for the experiment
            dataset_path = os.path.join(results_dir, name)
            chkpt_save_path = os.path.join(dataset_path, "checkpoint")

            os.makedirs(dataset_path, exist_ok=True)

            print(f"Evaluating dataset: {name} at path: {path}...")

            for resolution in resolutions:
                print(
                    f"\nRunning {backend} for dataset: {name} at resolution: {resolution}"
                )

                ad = sc.read(path)

                # Preprocessing using Seurat like parameters
                min_expr_level = 0
                min_cells = 3
                use_hvg = False
                n_top_genes = 720
                preprocessed_data = preprocess_recipe(
                    ad,
                    min_expr_level=min_expr_level,
                    min_cells=min_cells,
                    use_hvg=use_hvg,
                    n_top_genes=n_top_genes,
                    scale=True,
                )

                # Run PCA
                print("\nComputing PCA...")
                X_pca, _, n_comps = run_pca(
                    preprocessed_data, use_hvg=False, n_components=10
                )
                preprocessed_data.obsm["X_pca"] = X_pca

                # Run method
                n_episodes = 1 if dry_run else 10
                n_metric_epochs = 1 if dry_run else 10
                if metti_version == "v2":
                    run_metti_v2(
                        preprocessed_data,
                        n_episodes=n_episodes,
                        n_metric_epochs=n_metric_epochs,
                        chkpt_save_path=chkpt_save_path,
                        random_state=0,
                        cluster_kwargs={"random_state": 0, "resolution": resolution},
                        neighbor_kwargs={"random_state": 0, "n_neighbors": 50},
                        trainer_kwargs={
                            "optimizer": "SGD",
                            "lr": 0.01,
                            "batch_size": 32,
                        },
                        c_backend=backend,
                        threshold=threshold,
                        device=device,
                    )
                else:
                    run_metti(
                        preprocessed_data,
                        n_episodes=n_episodes,
                        n_metric_epochs=n_metric_epochs,
                        chkpt_save_path=chkpt_save_path,
                        random_state=0,
                        cluster_kwargs={"random_state": 0, "resolution": resolution},
                        neighbor_kwargs={"random_state": 0, "n_neighbors": 50},
                        trainer_kwargs={
                            "optimizer": "SGD",
                            "lr": 0.01,
                            "batch_size": 32,
                        },
                        c_backend=backend,
                        threshold=threshold,
                        device=device,
                    )

                # Plot embeddings
                plot_path = os.path.join(
                    dataset_path, backend, str(resolution), "plots"
                )
                os.makedirs(plot_path, exist_ok=True)

                save_kwargs = {"dpi": 300, "bbox_inches": "tight", "transparent": True}
                plot_embeddings(
                    preprocessed_data.obsm["metric_viz_embedding"],
                    save_path=os.path.join(plot_path, "embedding.png"),
                    title=f"embedding_{backend}_{resolution}",
                    save_kwargs=save_kwargs,
                )
                # Plot clusters
                plot_clusters(
                    preprocessed_data,
                    cluster_key="metric_clusters",
                    embedding_key="metric_viz_embedding",
                    cmap="plasma",
                    title=f"clusters_{backend}_{resolution}",
                    save_path=os.path.join(plot_path, "clusters.png"),
                    save_kwargs=save_kwargs,
                )

                # Plot graphs
                communities = preprocessed_data.obs["metric_clusters"]
                start_cell_ids = preprocessed_data.uns["start_id"]
                start_cell_ids = (
                    [start_cell_ids]
                    if isinstance(start_cell_ids, str)
                    else list(start_cell_ids)
                )
                start_cluster_ids = get_start_cell_cluster_id(
                    preprocessed_data, start_cell_ids, communities
                )
                un_connectivity = preprocessed_data.uns[
                    "metric_undirected_connectivities"
                ]
                plot_connectivity_graph(
                    preprocessed_data.obsm["metric_viz_embedding"],
                    communities,
                    un_connectivity,
                    mode="undirected",
                    title=f"undirected_{backend}_{resolution}",
                    save_path=os.path.join(plot_path, "undirected.png"),
                    save_kwargs=save_kwargs,
                )

                if metti_version == "v1":
                    connectivity = preprocessed_data.uns[
                        "metric_directed_connectivities"
                    ]
                    plot_trajectory_graph(
                        preprocessed_data.obsm["metric_viz_embedding"],
                        communities,
                        connectivity,
                        start_cluster_ids,
                        title=f"directed_{backend}_{resolution}",
                        save_path=os.path.join(plot_path, "directed.png"),
                        save_kwargs=save_kwargs,
                    )

                    # Plot pseudotime
                    plot_pseudotime(
                        preprocessed_data,
                        embedding_key="metric_viz_embedding",
                        pseudotime_key="metric_pseudotime",
                        cmap="plasma",
                        title=f"pseudotime_{backend}_{resolution}",
                        save_path=os.path.join(plot_path, "pseudotime.png"),
                        save_kwargs=save_kwargs,
                    )
                else:
                    plot_pseudotime(
                        preprocessed_data,
                        embedding_key="metric_viz_embedding",
                        pseudotime_key="metric_pseudotime_v2",
                        cmap="plasma",
                        title=f"pseudotime_{backend}_{resolution}",
                        save_path=os.path.join(plot_path, "pseudotime.png"),
                        save_kwargs=save_kwargs,
                    )

                    # plot_trajectory_graph_v2(
                    #     preprocessed_data.obs["metric_pseudotime_v2"],
                    #     nx.to_pandas_adjacency(
                    #         preprocessed_data.uns["metric_undirected_graph"]
                    #     ),
                    #     preprocessed_data.obs["metric_clusters"],
                    #     preprocessed_data.uns["metric_undirected_node_positions"],
                    #     title=f"directed_{backend}_{resolution}",
                    #     save_path=os.path.join(plot_path, "directed.png"),
                    #     save_kwargs=save_kwargs,
                    # )

                # Compute IM distance
                im = IpsenMikhailov()
                net1 = compute_gt_milestone_network(
                    preprocessed_data, mode="undirected"
                )
                net2 = preprocessed_data.uns["metric_undirected_graph"]
                r.loc[name, f"IM@{resolution}"] = round(im(net1, net2), 3)

                # Compute pseudotime
                gt_pseudotime = preprocessed_data.uns["timecourse"].reindex(
                    preprocessed_data.obs_names
                )
                pseudotime_key = (
                    "metric_pseudotime"
                    if metti_version == "v1"
                    else "metric_pseudotime_v2"
                )
                res = compute_ranking_correlation(
                    gt_pseudotime, preprocessed_data.obs[pseudotime_key]
                )
                r.loc[name, f"KT@{resolution}"] = round(res["kendall"][0], 3)
                r.loc[name, f"SR@{resolution}"] = round(res["spearman"][0], 3)
                clear_output(wait=True)
        r.to_csv(os.path.join(results_dir, f"metric_{backend}_results.csv"))


def evaluate_paga_topology(
    dataset_file_path,
    results_dir=os.getcwd(),
    resolutions=[0.4, 0.6, 0.8, 1.0],
    c_backends=["louvain", "leiden"],
):
    # Read the dataset file
    datasets = {}
    with open(dataset_file_path, "r") as fp:
        reader = csv.DictReader(fp)
        datasets = {row["name"]: row["path"] for row in reader}

    for backend in c_backends:
        r = pd.DataFrame(index=datasets.keys())
        for name, path in datasets.items():
            # Setup directory per dataset for the experiment
            dataset_path = os.path.join(results_dir, name)
            os.makedirs(dataset_path, exist_ok=True)

            print(f"Evaluating dataset: {name} at path: {path}...")

            for resolution in resolutions:
                print(
                    f"\nRunning {backend} for dataset: {name} at resolution: {resolution}"
                )
                ad = sc.read(path)

                # Preprocessing using Seurat like recipe
                min_expr_level = 0
                min_cells = 3
                use_hvg = False
                n_top_genes = 720
                preprocessed_data = preprocess_recipe(
                    ad,
                    min_expr_level=min_expr_level,
                    min_cells=min_cells,
                    use_hvg=use_hvg,
                    n_top_genes=n_top_genes,
                    scale=True,
                )

                # Run PAGA
                start_cell_ids = ad.uns["start_id"]
                start_cell_ids = (
                    [start_cell_ids]
                    if isinstance(start_cell_ids, str)
                    else list(start_cell_ids)
                )
                # Added here as sometimes PAGA plot throws error when applied on preprocessed data
                try:
                    run_paga(
                        preprocessed_data,
                        start_cell_ids[-1],
                        c_backend=backend,
                        neighbor_kwargs={"random_state": 0, "n_neighbors": 50},
                        cluster_kwargs={"random_state": 0, "resolution": resolution},
                    )
                    # Plot the PAGA graph
                    plot_path = os.path.join(dataset_path, backend, str(resolution))
                    os.makedirs(plot_path, exist_ok=True)
                    os.chdir(plot_path)
                    sc.pl.paga(preprocessed_data, save="_graph.png")
                except:
                    print(
                        f"PAGA run failed for dataset: {name}@{resolution}. Skipping writing results for this conf"
                    )
                    continue

                # Compute IM distance
                im = IpsenMikhailov()
                net1 = compute_gt_milestone_network(ad, mode="undirected")
                net2 = nx.from_scipy_sparse_matrix(
                    preprocessed_data.uns["paga"]["connectivities"]
                )
                r.loc[name, f"IM@{resolution}"] = round(im(net1, net2), 3)

                # Compute pseudotime
                gt_pseudotime = ad.uns["timecourse"].reindex(ad.obs_names)
                res = compute_ranking_correlation(
                    gt_pseudotime, preprocessed_data.obs["dpt_pseudotime"]
                )
                r.loc[name, f"KT@{resolution}"] = round(res["kendall"][0], 3)
                r.loc[name, f"SR@{resolution}"] = round(res["spearman"][0], 3)
                clear_output(wait=True)
        r.to_csv(os.path.join(results_dir, f"PAGA_{backend}_results.csv"))


def evaluate_palantir(dataset_file_path, results_dir=os.getcwd(), save_kwargs={}):
    # Read the dataset file
    datasets = {}
    with open(dataset_file_path, "r") as fp:
        reader = csv.DictReader(fp)
        datasets = {row["name"]: row["path"] for row in reader}

        r = pd.DataFrame(index=datasets.keys())
        for name, path in datasets.items():
            # Setup directory per dataset for the experiment
            dataset_path = os.path.join(results_dir, name)

            os.makedirs(dataset_path, exist_ok=True)

            print(f"Evaluating dataset: {name} at path: {path}...")

            # Read and Preprocess
            ad = sc.read(path)
            try:
                # In case the anndata object has scipy.sparse graphs
                ad.X = ad.X.todense()
            except:
                pass
            sc.pp.normalize_per_cell(ad)
            log_transform(ad, pseudo_count=0.1)
            sc.pp.highly_variable_genes(ad, n_top_genes=1500, flavor="cell_ranger")

            # Run Palantir
            start_cell_ids = ad.uns["start_id"]
            start_cell_ids = (
                [start_cell_ids]
                if isinstance(start_cell_ids, str)
                else list(start_cell_ids)
            )
            try:
                presults, ms_data = run_palantir(ad, start_cell_ids[-1])
            except np.linalg.LinAlgError:
                print(
                    f"Palantir run failed for dataset: {name}. Skipping writing results for this dataset"
                )
                continue

            # Generate plots for tsne embedding and pseudotime
            print("Generating plots!")
            embed = generate_plot_embeddings(
                ms_data, method="tsne", perplexity=150, random_state=0
            )
            embed_save_path = os.path.join(dataset_path, "embedding.png")
            pt_save_path = os.path.join(dataset_path, "pt.png")
            plot_embeddings(
                embed, save_path=embed_save_path, s=1, save_kwargs=save_kwargs
            )
            plot_embeddings(
                embed,
                c=presults.pseudotime,
                s=1,
                save_path=pt_save_path,
                cmap="plasma",
                show_colorbar=True,
                save_kwargs=save_kwargs,
            )

            # Compute pseudotime related metrics
            gt_pseudotime = ad.uns["timecourse"].reindex(ad.obs_names)
            res = compute_ranking_correlation(gt_pseudotime, presults.pseudotime)
            r.loc[name, "KT"] = round(res["kendall"][0], 3)
            r.loc[name, "SR"] = round(res["spearman"][0], 3)
            clear_output(wait=True)
        r.to_csv(os.path.join(results_dir, "palantir_pseudotime_results.csv"))
