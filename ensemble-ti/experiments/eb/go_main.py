from experiments.eb.go import generate_go_heatmap


# GO analysis for 5 major lineages
term_paths = {
    "EN": "/home/lexent/Downloads/go_terms_5_new/go_terms_new/GO_1.csv",
    "NC": "/home/lexent/Downloads/go_terms_5_new/go_terms_new/GO_2.csv",
    "NE": "/home/lexent/Downloads/go_terms_5_new/go_terms_new/GO_3.csv",
    "ME": "/home/lexent/Downloads/go_terms_5_new/go_terms_new/GO_4.csv",
}

pattern_paths = {
    "EN": "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_5/en_regex.txt",
    "NC": "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_5/nc_ne_regex.txt",
    "NE": "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_5/nc_ne_regex.txt",
    "ME": "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_5/me_regex.txt",
}

order = ["ME", "NE", "NC", "EN"]
color_map = {
    "ME": "#E76F51",
    "NE": "#00386d",
    "NC": "#f99c37",
    "EN": "#2A9D8F",
}
generate_go_heatmap(
    term_paths,
    pattern_paths,
    order=order,
    cmap="Greens",
    color_map=color_map,
    save_path="/home/lexent/go_5.png",
    save_kwargs={"dpi": 200, "transparent": True, "bbox_inches": "tight"},
)


# GO analysis for 26 clusters
term_paths = {
    0: "/home/lexent/Downloads/go_26/content/go_26/GO_0.csv",
    1: "/home/lexent/Downloads/go_26/content/go_26/GO_1.csv",
    2: "/home/lexent/Downloads/go_26/content/go_26/GO_2.csv",
    3: "/home/lexent/Downloads/go_26/content/go_26/GO_3.csv",
    4: "/home/lexent/Downloads/go_26/content/go_26/GO_4.csv",
    5: "/home/lexent/Downloads/go_26/content/go_26/GO_5.csv",
    7: "/home/lexent/Downloads/go_26/content/go_26/GO_7.csv",
    8: "/home/lexent/Downloads/go_26/content/go_26/GO_8.csv",
    9: "/home/lexent/Downloads/go_26/content/go_26/GO_9.csv",
    10: "/home/lexent/Downloads/go_26/content/go_26/GO_10.csv",
    11: "/home/lexent/Downloads/go_26/content/go_26/GO_11.csv",
    12: "/home/lexent/Downloads/go_26/content/go_26/GO_12.csv",
    14: "/home/lexent/Downloads/go_26/content/go_26/GO_14.csv",
    15: "/home/lexent/Downloads/go_26/content/go_26/GO_15.csv",
    16: "/home/lexent/Downloads/go_26/content/go_26/GO_16.csv",
    17: "/home/lexent/Downloads/go_26/content/go_26/GO_17.csv",
    18: "/home/lexent/Downloads/go_26/content/go_26/GO_18.csv",
    19: "/home/lexent/Downloads/go_26/content/go_26/GO_19.csv",
    20: "/home/lexent/Downloads/go_26/content/go_26/GO_20.csv",
    21: "/home/lexent/Downloads/go_26/content/go_26/GO_21.csv",
    23: "/home/lexent/Downloads/go_26/content/go_26/GO_23.csv",
    24: "/home/lexent/Downloads/go_26/content/go_26/GO_24.csv",
}

pattern_paths = {
    0: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/me_regex.txt",
    1: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/nc_regex.txt",
    2: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/en_regex.txt",
    3: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/nc_regex.txt",
    4: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/me_regex.txt",
    5: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/nc_regex.txt",
    7: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/nc_regex.txt",
    8: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/me_regex.txt",
    9: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/nc_regex.txt",
    10: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/me_regex.txt",
    11: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/me_regex.txt",
    12: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/nc_regex.txt",
    14: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/me_regex.txt",
    15: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/me_regex.txt",
    16: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/nc_regex.txt",
    17: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/me_regex.txt",
    18: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/nc_regex.txt",
    19: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/me_regex.txt",
    20: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/me_regex.txt",
    21: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/me_regex.txt",
    23: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/nc_regex.txt",
    24: "/home/lexent/ensemble-ti/ensemble-ti/experiments/eb/files/go_26/me_regex.txt",
}

order = [2, 3, 5, 16, 12, 23, 9, 1, 7, 18, 8, 14, 17, 10, 20, 19, 11, 21, 0, 15, 4, 24]
color_map = {
    0: "#95a0b3",
    1: "#f8aa81",
    2: "#2A9D8F",
    3: "#c2e2ff",
    4: "#424b5c",
    5: "#79beff",
    7: "#f99c37",
    8: "#e06250",
    9: "#d48c90",
    10: "#fecfc3",
    11: "#e6ead7",
    12: "#005eb6",
    14: "#841c01",
    15: "#6a7a95",
    16: "#319bff",
    17: "#fe967b",
    18: "#af8e80",
    19: "#c8d1a7",
    20: "#95a857",
    21: "#6b783e",
    23: "#00386d",
    24: "#E76F51",
}
generate_go_heatmap(
    term_paths,
    pattern_paths,
    order=order,
    cmap="Greens",
    figsize=(24, 6),
    color_map=color_map,
    var_group_rotation=90,
    save_path="/home/lexent/go_26.png",
    save_kwargs={"dpi": 200, "transparent": True, "bbox_inches": "tight"},
)
