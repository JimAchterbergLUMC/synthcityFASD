import pandas as pd
import math
from matplotlib import pyplot as plt

# make sure that discrete features have max. k features, set in tabular_vae.py
# or change code of tabular_vae.py such that it takes discrete features as input


def preprocess(X: pd.DataFrame, y: pd.DataFrame, config: dict):
    # drop features
    X = X.drop(config["drop"], axis=1)

    # cast to correct data types
    X[config["numerical"]] = X[config["numerical"]].astype(float)
    X[config["discrete"]] = X[config["discrete"]].astype(str)

    # dataset specific preprocessing
    if config["name"] == "adult":
        X.loc[
            ~X["native-country"].isin(["United-States", "Mexico"]), "native-country"
        ] = "Other"
        y = y.squeeze()
        y = y.replace({"<=50K": 0.0, "<=50K.": 0.0, ">50K": 1.0, ">50K.": 1.0})

    # add y as target to df
    y.name = "target"
    df = pd.concat([X, y], axis=1)
    return df


def plot_df(df: pd.DataFrame):
    # Determine the number of features
    num_features = len(df.columns)

    # Determine grid size (rows and columns)
    grid_size = math.ceil(math.sqrt(num_features))

    # Create a figure and axes with a grid layout
    fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(15, 15))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each feature
    for i, col in enumerate(df.columns):
        if df[col].dtype == "object":  # Discrete feature
            df[col].value_counts().plot(kind="bar", ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
        else:  # Numerical feature
            df[col].plot(kind="hist", ax=axes[i], bins=10)
            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_xlabel(col)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()
