import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn.objects as so
from scipy.stats import mannwhitneyu, norm
import matplotlib.patches as patches


def format_table(df, metrics, save_path="UIAYN_experiments/results_formatted"):
    # save_path = save_path + f"/{ds}"
    # Group the DataFrame by 'name'
    grouped = df.groupby("name")

    df.loc[df["name"] == "performance.feat_rank_distance.corr", "direction"] = (
        "maximize"
    )

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

    fid_table = []
    ut_table = []
    priv_table = []
    for metric, table in dfs.items():

        # cannot attach AIA metrics since these rows do not exist for each column in the table
        # they are dataset specific
        if "leakage" in metric:
            metrics.update(
                {
                    metric: {
                        "name": "AIA-" + metric.split(".")[-1],
                        "category": "privacy",
                    }
                }
            )

        # only use the metrics which are contained in the dictionary
        z = df.groupby("name")["direction"].first()  # .reset_index(drop=False)
        z = z.map({"minimize": "\u2193", "maximize": "\u2191"}).to_dict()

        if metric in metrics.keys():
            dss = list(table.columns)
            table = table.reset_index()
            # add the statistic to the table
            table["stat"] = [""] * len(table)
            cols = ["stat", "model"]
            cols.extend(dss)
            table = table[cols]
            empty_row = pd.DataFrame([[""] * len(table.columns)], columns=table.columns)
            table = pd.concat([empty_row, table], ignore_index=True)
            table.iloc[0, 0] = metrics[metric]["name"] + f" {z[metric]}"

            # attach to own metric category table
            if metrics[metric]["category"] == "fidelity":
                fid_table.append(table)
            elif metrics[metric]["category"] == "utility":
                ut_table.append(table)
            else:
                priv_table.append(table)

    for tab, cat in zip(
        [fid_table, ut_table, priv_table], ["fidelity", "utility", "privacy"]
    ):
        fin_tab = pd.concat(tab, ignore_index=True)
        fin_tab = fin_tab[["stat", "model", "adult", "credit", "student", "heart"]]
        fin_tab.to_csv(f"{result_path}/{cat}_table.csv", index=False)


def format_plot(df, ds, metrics):
    # select data for current dataset
    data = df[df["ds"] == ds]

    # attach dataset specific AIA metric
    for _, row in data[
        data["name"].str.contains("leakage", case=False, na=False)
    ].iterrows():
        metrics.update(
            {
                row["name"]: {
                    "name": "AIA-" + row["name"].split(".")[-1],
                    "category": "privacy",
                }
            }
        )

    # remove all non used metrics
    data = data[data["name"].apply(lambda x: x in metrics.keys())]

    # add metrics on real set as Real model
    # real_metrics = (
    #     data[data["name"].str.endswith(".gt")]
    #     .groupby("name")
    #     .agg(
    #         {
    #             "mean": "mean",
    #             "direction": "first",
    #             "stddev": "mean",  # For numerical columns, take the mean
    #             "model": "first",
    #             "ds": "first",  # For non-numericals, take first value
    #         }
    #     )
    # )
    # real_metrics = real_metrics.reset_index()
    # real_metrics["model"] = "real"
    # real_metrics["name"] = real_metrics["name"].str.replace(
    #     r"\.[^.]+$", ".syn_ood", regex=True
    # )
    # data = pd.concat([data, real_metrics], ignore_index=True)
    # # dont forget to remove the original metrics so it doesnt get plotted
    # metrics = {k: v for k, v in metrics.items() if not k.endswith(".gt")}

    # map the names to interpretable names
    data["name"] = data["name"].map(lambda x: metrics.get(x, {}).get("name", x))

    # revert wrong direction of feature importance (for utility, higher is better)
    data.loc[data["name"] == "Feature Importance", "direction"] = "maximize"

    # Categories (x-axis labels)
    categories = list(data["model"].unique())
    colors = sns.color_palette("Greys", len(categories))

    # Create the grid layout
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

    # create the subplots for fidelity, utility, and privacy within a single figure
    for j, cat in enumerate(["fidelity", "utility", "privacy"]):
        cur_metrics = [
            value["name"] for value in metrics.values() if value["category"] == cat
        ]
        cur_data = data[data["name"].isin(cur_metrics)]
        subplot(
            fig=fig,
            grid=gs,
            cur_data=cur_data,
            cur_metrics=cur_metrics,
            n_subplot=j,
            categories=categories,
            colors=colors,
        )

    plt.suptitle(ds)
    plt.tight_layout()
    # plt.savefig(f"UIAYN_experiments/results_formatted/{ds}/plot_{ds}.png")
    plt.show()


def subplot(fig, grid, cur_data, cur_metrics, n_subplot, categories, colors):
    # this function creates a subplot for each metric category
    gs_ = gridspec.GridSpecFromSubplotSpec(
        1, len(cur_metrics), subplot_spec=grid[n_subplot], wspace=0.4
    )

    for i, metric in enumerate(cur_metrics):
        # delete the Real as category if not relevant for cur metric
        cats = categories.copy()
        if "real" in categories and (
            not "real" in cur_data[cur_data["name"] == metric]["model"].to_list()
        ):
            cats.remove("real")

        ax = plt.Subplot(fig, gs_[i])
        y = (cur_data[cur_data["name"] == metric]["mean"]).reset_index(drop=True)
        sns.barplot(
            x=cats,
            y=y,
            ax=ax,
            palette=colors,
        )

        # set barcolor of best and worst bar to highlight best performers
        if (cur_data[cur_data["name"] == metric]["direction"] == "maximize").any():
            best_bar = cats[y.idxmax()]
            worst_bar = cats[y.idxmin()]
            tick = "\u2191"
        else:
            best_bar = cats[y.idxmin()]
            worst_bar = cats[y.idxmax()]
            tick = "\u2193"
        for bar, cat in zip(ax.patches, cats):

            # set best and worst bar to green and red
            if cat == best_bar:
                bar.set_facecolor("green")
            elif cat == worst_bar:
                bar.set_facecolor("red")

            # highlight FASD
            if cat == "fasd":
                bar.set_hatch("//")

        ax.set_title(f"{metric} {tick}", fontsize=10)  # Upward arrow
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # set y bounds for AUC metrics
        if any(
            string in metric
            for string in ["Distinguishability", "XGB", "LR", "MLP", "AIA"]
        ):
            # additional check, only set bounds if all values are in these bounds
            if not (any(y < 0.5) or any(y > 1)):
                ax.set_ybound(0.5, 1)
        ax.set_ylabel("")
        fig.add_subplot(ax)


def stripplot_new(df, metric):
    # select data for current dataset
    data = df

    # attach dataset specific AIA metric
    for _, row in data[
        data["name"].str.contains("leakage", case=False, na=False)
    ].iterrows():
        metrics.update(
            {
                row["name"]: {
                    "name": "AIA-" + row["name"].split(".")[-1],
                    "category": "privacy",
                }
            }
        )

    # remove all non used metrics
    data = data[data["name"].apply(lambda x: x in metrics.keys())]

    # add metric category as column
    data["category"] = data["name"].map(lambda x: metrics.get(x, {}).get("category", x))

    # map the names to interpretable names
    data["name"] = data["name"].map(lambda x: metrics.get(x, {}).get("name", x))

    data = data[data["name"] != "Feature Importance"]

    # for each metric in each dataset rank the mean score
    data["score_rank"] = data.groupby(["ds", "name"]).rank()["mean"]
    # for direction==maximize invert the ranking so rank of 1 is the best
    data["score_rank"] = np.where(
        data["direction"] == "maximize",
        data.groupby(["ds", "name"])["score_rank"].transform(lambda x: x.max() - x + 1),
        data["score_rank"],
    )
    # plot
    palette = {
        "fidelity": "gray",
        "utility": "black",
        "privacy": "red",
    }
    data = data.rename({"ds": "dataset"}, axis=1)

    def fun(x: float, pos: int) -> str:
        if x == 1:
            return str(int(x)) + " (Best)"
        elif x == 5:
            return str(int(x)) + " (Worst)"
        else:
            return str(int(x))

    markdict = {
        "adult": "o",
        "credit": "v",
        "student": "^",
        "heart": "<",
    }

    bar_data = data.groupby(["category", "model"], as_index=False)["score_rank"].mean()
    bar_data = bar_data.rename(columns={"score_rank": "mean_score"})

    sns.barplot(
        data=bar_data,
        x="model",
        y="mean_score",
        hue="category",
        palette=palette,
        order=data["category"][::-1],
        alpha=0.5,
    )

    plt.gca().invert_yaxis()

    plt.show()


def stripplot(df, metrics):
    # select data for current dataset
    data = df

    # attach dataset specific AIA metric
    for _, row in data[
        data["name"].str.contains("leakage", case=False, na=False)
    ].iterrows():
        metrics.update(
            {
                row["name"]: {
                    "name": "AIA-" + row["name"].split(".")[-1],
                    "category": "privacy",
                }
            }
        )

    # remove all non used metrics
    data = data[data["name"].apply(lambda x: x in metrics.keys())]

    # add metric category as column
    data["category"] = data["name"].map(lambda x: metrics.get(x, {}).get("category", x))

    # map the names to interpretable names
    data["name"] = data["name"].map(lambda x: metrics.get(x, {}).get("name", x))

    data = data[data["name"] != "Feature Importance"]

    # for each metric in each dataset rank the mean score
    data["score_rank"] = data.groupby(["ds", "name"]).rank()["mean"]
    # for direction==maximize invert the ranking so rank of 1 is the best
    data["score_rank"] = np.where(
        data["direction"] == "maximize",
        data.groupby(["ds", "name"])["score_rank"].transform(lambda x: x.max() - x + 1),
        data["score_rank"],
    )
    # plot
    palette = {
        "fidelity": "gray",
        "utility": "green",
        "privacy": "red",
    }
    data = data.rename({"ds": "dataset"}, axis=1)

    data["model"] = data["model"].map(
        {
            "fasd": "FASD",
            "pategan": "PATE-GAN",
            "adsgan": "AdsGAN",
            "tvae": "TVAE",
            "ctgan": "CTGAN",
        }
    )

    def fun(x: float, pos: int) -> str:
        if x == 1:
            return str(int(x)) + " (Best)"
        elif x == 5:
            return str(int(x)) + " (Worst)"
        else:
            return str(int(x))

    markdict = {
        "adult": "o",
        "credit": "v",
        "student": "^",
        "heart": "<",
    }

    bar_data = data.groupby(["category", "model"], as_index=False)["score_rank"].mean()
    bar_data = bar_data.rename(columns={"score_rank": "mean_score"})
    b = (
        so.Plot(
            bar_data,
            x="model",
            y="mean_score",
            color="category",
        )
        .add(so.Bar(alpha=0.3, edgewidth=0, baseline=6), so.Dodge(), legend=False)
        .scale(
            color=so.Nominal(palette),
        )
    )

    p = (
        so.Plot(
            data,
            x="model",
            y="score_rank",
            color="category",
            marker="dataset",
        )
        .add(
            so.Dot(pointsize=5, edgewidth=0.05, edgecolor="gray"),
            so.Dodge(),
            so.Jitter(
                x=0,
                y=0.3,
                seed=0,
            ),
        )
        .scale(
            y=so.Continuous().tick(every=1).label(like=fun),
            color=so.Nominal(palette),
            marker=so.Nominal(markdict),
        )
        .label(x="model", y="score_rank", color="Metric")
        .label(x="model", y="score_rank", marker="Dataset")
    )

    fig, ax = plt.subplots()

    # plot results
    # l.on(ax).plot()
    b.on(ax).plot()
    p = p.on(ax).plot()
    plt.ylim((0.5, 5.5))
    plt.xlabel("")
    plt.ylabel("")
    plt.gca().invert_yaxis()

    # Find the coordinates for "FASD"
    x_fasd = bar_data["model"].unique().tolist().index("FASD")
    y_fasd = (
        data[data["model"] == "FASD"]["score_rank"].min() - 0.2
    )  # Adjust as necessary

    # Annotate the "FASD" model with an arrow
    ax.annotate(
        "FASD",
        xy=(x_fasd, y_fasd),  # Point to the position of "FASD"
        xytext=(x_fasd, y_fasd - 0.5),  # Position of the text
        arrowprops=dict(facecolor="black", arrowstyle="->"),  # Arrow properties
        fontsize=10,
        ha="center",  # Horizontal alignment
        color="red",  # Color of the text
    )

    p.save(
        "UIAYN_experiments/results_formatted/rank_fig.pdf",
        bbox_inches="tight",
        # pad_inches=0.5,
    )


def mann_whitney_tests(df, metrics):
    data = df

    # attach dataset specific AIA metric
    for _, row in data[
        data["name"].str.contains("leakage", case=False, na=False)
    ].iterrows():
        metrics.update(
            {
                row["name"]: {
                    "name": "AIA-" + row["name"].split(".")[-1],
                    "category": "privacy",
                }
            }
        )

    # remove all non used metrics
    data = data[data["name"].apply(lambda x: x in metrics.keys())]

    # add metric category as column
    data["category"] = data["name"].map(lambda x: metrics.get(x, {}).get("category", x))

    # map the names to interpretable names
    data["name"] = data["name"].map(lambda x: metrics.get(x, {}).get("name", x))

    data = data[data["name"] != "Feature Importance"]

    # for each metric in each dataset rank the mean score
    data["score_rank"] = data.groupby(["ds", "name"]).rank()["mean"]
    # for direction==maximize invert the ranking so rank of 1 is the best
    data["score_rank"] = np.where(
        data["direction"] == "maximize",
        data.groupby(["ds", "name"])["score_rank"].transform(lambda x: x.max() - x + 1),
        data["score_rank"],
    )

    mw_ind = {}
    for cat in ["fidelity", "utility", "privacy", "privacy_twosided"]:
        mw_ind[cat] = {}
        cat_ = cat
        alternative = "less"
        if cat == "privacy_twosided":
            cat_ = "privacy"
            alternative = "two-sided"

        for model_ in ["pategan", "ctgan", "adsgan", "tvae"]:

            data1 = data[(data["model"] == "fasd") & (data["category"] == cat_)][
                "score_rank"
            ]
            data2 = data[(data["model"] == model_) & (data["category"] == cat_)][
                "score_rank"
            ]
            stat, pval = mannwhitneyu(data1, data2, alternative=alternative)
            mw_ind[cat][model_] = str(stat) + f" ({np.round(pval, 3)})"

    mw_ind = pd.DataFrame(mw_ind).reset_index(drop=False)
    mw_ind = mw_ind.rename({"index": "model"}, axis=1)
    mw_ind.to_csv("UIAYN_experiments/results_formatted/MWU.csv")


def weighted_z_test(df, metrics):

    data = df

    # attach dataset specific AIA metric
    for _, row in data[
        data["name"].str.contains("leakage", case=False, na=False)
    ].iterrows():
        metrics.update(
            {
                row["name"]: {
                    "name": "AIA-" + row["name"].split(".")[-1],
                    "category": "privacy",
                }
            }
        )

    # remove all non used metrics
    data = data[data["name"].apply(lambda x: x in metrics.keys())]

    # add metric category as column
    data["category"] = data["name"].map(lambda x: metrics.get(x, {}).get("category", x))

    # map the names to interpretable names
    data["name"] = data["name"].map(lambda x: metrics.get(x, {}).get("name", x))

    data = data[data["name"] != "Feature Importance"]

    # for each metric in each dataset rank the mean score
    data["score_rank"] = data.groupby(["ds", "name"]).rank()["mean"]
    # for direction==maximize invert the ranking so rank of 1 is the best
    data["score_rank"] = np.where(
        data["direction"] == "maximize",
        data.groupby(["ds", "name"])["score_rank"].transform(lambda x: x.max() - x + 1),
        data["score_rank"],
    )

    mw = {}
    for model_ in ["pategan", "ctgan", "adsgan", "tvae"]:
        z_scores = []
        weights = []
        for cat in ["utility", "privacy"]:
            data1 = data[(data["model"] == "fasd") & (data["category"] == cat)][
                "score_rank"
            ]
            data2 = data[(data["model"] == model_) & (data["category"] == cat)][
                "score_rank"
            ]
            stat, pval = mannwhitneyu(data1, data2, alternative="less")
            n1 = len(data1)
            n2 = len(data2)
            mu = (n1 * n2) / 2
            sd = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
            z_scores.append((stat - mu) / sd)
            weights.append(n1 + n2)
        z_scores, weights = np.array(z_scores), np.array(weights)
        z_weighted = np.sum(z_scores * weights) / (np.sqrt(np.sum(weights**2)))
        pval_weighted = 2 * (1 - norm.cdf(abs(z_weighted)))
        mw[model_] = str(z_weighted) + f" ({np.round(pval_weighted,3)})"

    mw = pd.DataFrame(mw, index=["test"], columns=mw.keys())  # .reset_index(drop=False)
    mw.to_csv("UIAYN_experiments/results_formatted/Z_test.csv")


if __name__ == "__main__":

    ds = "heart"

    metrics = {
        "stats.jensenshannon_dist.marginal": {"name": "JS", "category": "fidelity"},
        # "stats.max_mean_discrepancy.joint": {"name": "MMD", "category": "fidelity"},
        "stats.wasserstein_dist.joint": {"name": "Wasserstein", "category": "fidelity"},
        "stats.alpha_precision.delta_precision_alpha_OC": {
            "name": "a-precision",
            "category": "fidelity",
        },
        "stats.alpha_precision.delta_coverage_beta_OC": {
            "name": "B-recall",
            "category": "fidelity",
        },
        "detection.detection_xgb.mean": {
            "name": "Distinguishability",
            "category": "fidelity",
        },
        "performance.xgb.syn_ood": {"name": "XGB", "category": "utility"},
        # "performance.xgb.gt": {"name": "XGB_real", "category": "utility"},
        "performance.mlp.syn_ood": {"name": "MLP", "category": "utility"},
        # "performance.mlp.gt": {"name": "MLP_real", "category": "utility"},
        "performance.linear_model.syn_ood": {"name": "LR", "category": "utility"},
        # "performance.linear_model.gt": {"name": "LR_real", "category": "utility"},
        "performance.feat_rank_distance.corr": {
            "name": "Feature Importance",
            "category": "utility",
        },
        "privacy.delta-presence.score": {
            "name": "d-presence",
            "category": "privacy",
        },
        "privacy.k-anonymization.syn": {
            "name": "k-anonymization",
            "category": "privacy",
        },
        "privacy.k-map.score": {"name": "k-map", "category": "privacy"},
        "privacy.distinct l-diversity.syn": {
            "name": "l-diversity",
            "category": "privacy",
        },
        "privacy.identifiability_score.score_OC": {
            "name": "Identifiability",
            "category": "privacy",
        },
        "stats.alpha_precision.authenticity_OC": {
            "name": "Authenticity",
            "category": "privacy",
        },
    }

    # loop through the datasets in results
    results = []
    curwd = "UIAYN_experiments/results"
    for ds_dir in os.listdir(curwd):
        path = f"{curwd}/{ds_dir}"
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

    # format_table(df, metrics=metrics)
    # format_plot(df, ds, metrics=metrics)
    stripplot(df, metrics)
    # stripplot_new(df, metrics)
    # mann_whitney_tests(df, metrics)
    # weighted_z_test(df, metrics)
