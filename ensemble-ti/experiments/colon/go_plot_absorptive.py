import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import plot_annotated_heatmap


def transform_pval(p):
    return np.sqrt(-np.log(p))


go_ids = [
    # BEST4/OTOP2
    "GO:1990169",
    "GO:0010273",
    "GO:0061687",
    "GO:0097501",
    "GO:0071276",
    "GO:0071294",
    "GO:0071280",
    "GO:0055069",
    "GO:0055076",
    "GO:0046916",
    # CT COLONOCYTES
    "GO:0002446",
    "GO:0002283",
    "GO:0002444",
    "GO:0042119",
    "GO:0043312",
    "GO:0051234",
    "GO:0006887",
    # COLONOCYTES
    "GO:0006820",
    "GO:0046943",
    "GO:0015711",
    "GO:0032532",
    "GO:0032528",
    "GO:0032787",
    # AP'S
    "GO:0044281",
    "GO:0006629",
    "GO:1901615",
    "GO:0019752",
    "GO:0019637",
    "GO:0044255",
]

term_dict = {
    "BEST4/OTOP2": "experiments/colon/files/go_goblet_absorptive/GO_13.csv",
    "CT Colonocytes": "experiments/colon/files/go_goblet_absorptive/GO_3.csv",
    "Colonocytes": "experiments/colon/files/go_goblet_absorptive/GO_8.csv",
    "AP's": "experiments/colon/files/go_goblet_absorptive/GO_1.csv",
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

print(len(descriptions))

plot_annotated_heatmap(
    mat.to_numpy(),
    cmap="cividis",
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
    save_path="/home/lexent/go_plot_absorptive.png",
    cb_kwargs={"shrink": 0.3},
)
plt.show()
