import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_im_box_plots(
    metric_file,
    paga_file,
    colors=None,
    show_labels=False,
    save_path=None,
    save_kwargs={},
    figsize=None,
    **kwargs,
):
    # This code expects the results generated from `compare_global_topology` and is used
    # for comparing IM scores between PAGA and our approach
    metric_results = pd.read_csv(metric_file, index_col=0)
    paga_results = pd.read_csv(paga_file, index_col=0)

    assert metric_results.shape == paga_results.shape

    datasets = metric_results.index
    x_metric = []
    x_paga = []
    y = []
    for dataset in datasets:
        metric_res = metric_results.loc[
            dataset, ["IM@0.4", "IM@0.6", "IM@0.8", "IM@1.0"]
        ]
        paga_res = paga_results.loc[dataset, ["IM@0.4", "IM@0.6", "IM@0.8", "IM@1.0"]]

        # Store average results for a final plot
        x_metric.append(metric_res)
        x_paga.append(paga_res)
        y.append(dataset)

    # Generate boxplots
    plt.figure(figsize=figsize)
    metric_bp = plt.boxplot(
        x_metric,
        # labels=y,
        # labels=["Metric"] * len(x_metric),
        patch_artist=True,
        positions=np.arange(1, 3 * len(x_metric) + 1, step=3),
        **kwargs,
    )
    paga_bp = plt.boxplot(
        x_paga,
        # labels=y,
        # labels=["PAGA"] * len(x_paga),
        patch_artist=True,
        positions=np.arange(2, 3 * len(x_paga) + 2, step=3),
        **kwargs,
    )

    if colors is not None:
        assert len(colors) == 2
        for mp, bp in zip(metric_bp["boxes"], paga_bp["boxes"]):
            mp.set(facecolor=colors[0])
            bp.set(facecolor=colors[1])

    if show_labels:
        plt.gca().set_ylabel("IM Score")
        plt.gca().set_xlabel("Datasets")

    # Remove the right and top axes
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    # Dataset names can be big so rotate!
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment="right")

    # Legend
    plt.gca().legend(
        [metric_bp["boxes"][0], paga_bp["boxes"][0]],
        ["Metric", "PAGA"],
        # loc="upper right",
    )

    # Save
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)

    plt.show()


def generate_im_line_plots(
    metric_file,
    paga_file,
    colors=None,
    show_labels=False,
    save_path=None,
    save_kwargs={},
    figsize=None,
    **kwargs,
):
    # This code expects the results generated from `compare_global_topology` and is used
    # for comparing IM scores between PAGA and our approach
    metric_results = pd.read_csv(metric_file, index_col=0)
    paga_results = pd.read_csv(paga_file, index_col=0)

    assert metric_results.shape == paga_results.shape

    datasets = metric_results.index
    x_metric = []
    x_paga = []
    y = []
    for dataset in datasets:
        metric_res = metric_results.loc[
            dataset, ["IM@0.4", "IM@0.6", "IM@0.8", "IM@1.0"]
        ]
        paga_res = paga_results.loc[dataset, ["IM@0.4", "IM@0.6", "IM@0.8", "IM@1.0"]]

        # Store average results for a final plot
        x_metric.append(metric_res.mean())
        x_paga.append(paga_res.mean())
        y.append(dataset)

    # Generate line plots
    plt.figure(figsize=figsize)
    plt.plot(
        y,
        x_metric,
        color=colors[0] if colors is not None else None,
        marker="+",
        label="Metric",
        **kwargs,
    )
    plt.plot(
        y,
        x_paga,
        color=colors[1] if colors is not None else None,
        marker="o",
        label="PAGA",
        **kwargs,
    )

    if show_labels:
        plt.gca().set_ylabel("Average IM Score")
        plt.gca().set_xlabel("Datasets")

    # Remove the right and top axes
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    # Dataset names can be big so rotate!
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment="right")

    # Legend
    plt.gca().legend()

    # Save
    if save_path is not None:
        plt.savefig(save_path, **save_kwargs)

    plt.show()
