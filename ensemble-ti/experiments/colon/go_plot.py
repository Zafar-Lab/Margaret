import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import plot_annotated_heatmap


def transform_pval(p):
    return np.sqrt(-np.log(p))


go_ids = [
    # HC (mature goblets)
    "GO:0002483",
    "GO:0001913",
    "GO:0060337",
    "GO:0002283",
    "GO:0034340",
    "GO:0002444",
    "GO:0002446",
    # IC (Mature goblets)
    "GO:0034330",
    "GO:0016477",
    "GO:0009653",
    "GO:0009611",
    "GO:0097435",
    "GO:0030855",
    "GO:0042060",
    "GO:0006950",
    "GO:0008219",
    "GO:0051707",
    # HC (immature goblets)
    "GO:0006614",
    "GO:0006613",
    "GO:0045047",
    "GO:0000956",
    "GO:0072594",
]

term_dict = {
    "HC (Mature)": "experiments/colon/files/go_goblet_normal/GO_4.csv",
    "IC (Mature)": "experiments/colon/files/go_goblet_inflammed/GO_2.csv",
    "HC (Immature)": "experiments/colon/files/go_goblet_normal/GO_0.csv",
    "IC (Immature)": "experiments/colon/files/go_goblet_inflammed/GO_0.csv",
}

n_terms = len(go_ids)
n_cases = len(term_dict)

mat = pd.DataFrame(np.zeros((n_terms, n_cases)), index=go_ids, columns=term_dict.keys())
descriptions = {t: "" for t in go_ids}

for case, path in term_dict.items():
    term_df = pd.read_csv(path, index_col=1)
    term_pvals = [
        transform_pval(term_df.loc[t, "p_value"]) if t in term_df.index else 0
        for t in go_ids
    ]
    descriptions.update(
        {t: term_df.loc[t, "name"] for t in go_ids if t in term_df.index}
    )
    mat.loc[:, case] = term_pvals

print(descriptions)

plot_annotated_heatmap(
    mat.to_numpy(),
    cmap="PuBu",
    col_labels=term_dict.keys(),
    row_labels=descriptions.values(),
    figsize=(7, 8),
    fontsize=8,
    annotate_text=False,
    aspect="equal",
    save_kwargs={
        "transparent": True,
        "pad_inches": 0,
        "dpi": 300,
    },
    save_path="/home/lexent/go_plot.png",
    cb_kwargs={"shrink": 0.3},
)
plt.show()
