# individual tables for all fidelity, utility and privacy metrics

# fidelity table:
# - separate table for each metric
# - columns are datasets
# - rows are models
# -> then in Latex, stack the different tables on top of each other!

import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def format_table(df, save_path="results_formatted"):

    # Group the DataFrame by 'name'
    grouped = df.groupby("name")

    # Create a dictionary to hold the DataFrames
    dfs = {}

    # Process each group
    for name, group in grouped:
        # Pivot the table
        pivot = group.pivot(index="model", columns="ds", values=["mean", "stddev"])

        # Round the values to three decimal places
        mean_rounded = pivot["mean"].round(3)
        stddev_rounded = pivot["stddev"].round(3)

        # Format for LaTeX
        formatted_df = (
            "$"
            + mean_rounded.applymap(lambda x: f"{x:.3f}")
            + "_{\pm "
            + stddev_rounded.applymap(lambda x: f"{x:.3f}")
            + "}$"
        )

        # Store in the dictionary
        dfs[name] = formatted_df

    result_path = f"{save_path}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for metric, table in dfs.items():
        table.to_csv(f"{result_path}/{metric}.csv")


def format_plot(df):
    # remove metrics which are unwanted in the plot
    remove = [
        "stats.alpha_precision.delta_precision_alpha_naive",
        "stats.alpha_precision.delta_coverage_beta_naive",
        "stats.alpha_precision.authenticity_naive",
        "performance.xgb.gt",
        "performance.xgb.syn_id",
        "privacy.k-anonymization.gt",
        "privacy.distinct l-diversity.gt",
        "privacy.identifiability_score.score",
    ]  # all metrics ending in these words are unwanted
    df = df[df["name"].apply(lambda x: x not in remove)]

    # map metric names to interpretable names
    metric_mapper = {
        "stats.jensenshannon_dist.marginal": "JS",
        "stats.max_mean_discrepancy.joint": "MMD",
        "stats.alpha_precision.delta_precision_alpha_OC": "a_prc",
        "stats.alpha_precision.delta_coverage_beta_OC": "B_rec",
        "stats.alpha_precision.authenticity_OC": "Auth",
        "performance.xgb.syn_ood": "TSTR",
        "detection.detection_xgb.mean": "Detc",
        "privacy.delta-presence.score": "d_pres",
        "privacy.k-anonymization.syn": "k_ano",
        "privacy.k-map.score": "k_map",
        "privacy.distinct l-diversity.syn": "l_div",
        "privacy.identifiability_score.score_OC": "idtf",
    }
    # attach dataset specific AIA metric
    for idx, row in df[
        df["name"].str.contains("leakage", case=False, na=False)
    ].iterrows():
        metric_mapper[row["name"]] = "AIA_" + row["name"].split(".")[-1]
    df["name"] = df["name"].map(metric_mapper)

    metric_info = {
        "utility": ["TSTR"],
        "privacy": [
            "d_pres",
            "k_ano",
            "k_map",
            "l_div",
            "idtf",
            "Auth",
        ],
        "fidelity": ["JS", "MMD", "a_prc", "B_rec", "Detc"],
    }
    # attach dataset specific AIA metric
    metric_info["privacy"].extend(
        [value for value in metric_mapper.values() if "AIA_" in value]
    )

    # add metric ranges
    df["metric_range"] = [(0, 1)] * len(df)
    special_ranges = {
        "d_pres": (
            0,
            df["mean"][df["name"] == "d_pres"].max(),
        ),
        "k_ano": (
            1,
            df["mean"][df["name"] == "k_ano"].max(),
        ),
        "k_map": (
            1,
            df["mean"][df["name"] == "k_map"].max(),
        ),
        "l_div": (
            0,
            df["mean"][df["name"] == "l_div"].max(),
        ),
    }  # all non (0,1) ranges
    df["metric_range"] = df.apply(
        lambda row: special_ranges.get(row["name"], row["metric_range"]), axis=1
    )

    # scale all to (0,1)
    df["mean"] = df.apply(
        lambda row: (row["mean"] - row["metric_range"][0])
        / (row["metric_range"][1] - row["metric_range"][0]),
        axis=1,
    )

    # for minimize rows, perform (1-x)
    df["mean"][df["direction"] == "minimize"] = (
        1 - df["mean"][df["direction"] == "minimize"]
    )

    for dataset in df.ds.unique():
        df = df[df.ds == dataset]
        fig, axes = plt.subplots(3, 1, figsize=(10, 7))
        bar_width = 0.2
        for j, (cat, metrics) in enumerate(metric_info.items()):
            data = df[df["name"].isin(metrics)]
            ax = sns.barplot(
                data=data,
                x="name",
                y="mean",
                hue="model",
                ax=axes[j],
                width=bar_width,
                palette="viridis",
            )

            # Highlight the bar in bright red
            for patch, model_name in zip(ax.patches, data["model"]):
                if model_name == "FASD":
                    patch.set_facecolor("red")

            axes[j].set_xlabel("")
            axes[j].set_ylabel("")
            axes[j].set_ylim((0, 1))
            axes[j].set_title(cat)
            axes[j].legend(loc="upper right")

            # Customize the legend
            handles, labels = ax.get_legend_handles_labels()
            new_handles = []
            for handle, label in zip(handles, labels):
                if label == "FASD":
                    handle.set_color("red")  # Set legend color to red for 'TVAE'
                new_handles.append(handle)
            axes[j].legend(handles=new_handles, loc="upper right")

        plt.suptitle(dataset)
        plt.tight_layout()
        plt.show()


# TBD:
# scaling per dataset instead of over all datasets?

if __name__ == "__main__":

    # loop through the datasets in results
    results = []
    for ds_dir in os.listdir("results"):
        path = f"results/{ds_dir}"
        for filename in os.listdir(path):
            results.append(
                pd.read_csv(f"{path}/{filename}").assign(
                    model=filename.split(".")[0], ds=ds_dir
                )
            )

    results = pd.concat(results)
    results = results.rename({"Unnamed: 0": "name"}, axis=1)
    results = results[["name", "mean", "direction", "stddev", "model", "ds"]]
    df = results

    format_table(df)
    format_plot(df)
