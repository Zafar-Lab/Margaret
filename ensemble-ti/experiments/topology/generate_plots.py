import matplotlib.pyplot as plt
import os
import pandas as pd


def create_box_plots(
    metric_file,
    paga_file,
    save_dir=os.getcwd(),
    save_kwargs={},
    box_kwargs={},
    summary_kwargs={},
    figsize=None,
):
    # This code expects the results generated from `compare_global_topology` and is used
    # for comparing IM scores between PAGA and our approach
    metric_results = pd.read_pickle(metric_file)
    paga_results = pd.read_pickle(paga_file)

    assert metric_results.shape == paga_results.shape

    datasets = metric_results.index
    x_metric = []
    x_paga = []
    y = []
    for dataset in datasets:
        metric_res = metric_results.loc[
            dataset, ["IM@0.4", "IM@0.6", "IM@0.8", "IM@1.0"]
        ]
        paga_res = paga_results.loc[dataset, [0.4, 0.6, 0.8, 1.0]]

        # Store average results for a final plot
        x_metric.append(metric_res.mean())
        x_paga.append(paga_res.mean())
        y.append(dataset)

        # Generate boxplots
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dataset}.png")
        plt.figure(figsize=figsize)
        plt.title(f"{dataset}")
        plt.boxplot([metric_res, paga_res], labels=["Metric", "PAGA"], **box_kwargs)
        plt.savefig(save_path, bbox_inches="tight", **save_kwargs)

    plt.figure()
    plt.plot(
        datasets, x_metric, color="green", marker="o", label="Metric", **summary_kwargs
    )
    plt.plot(datasets, x_paga, color="blue", marker="o", label="PAGA", **summary_kwargs)
    plt.ylabel("Average IM Score")
    plt.legend()

    # Dataset names can be big so rotate!
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment="right")

    save_path = os.path.join(save_dir, f"summary.png")
    plt.savefig(save_path, bbox_inches="tight", **save_kwargs)
