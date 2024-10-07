import pandas as pd
import math
from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.impute import SimpleImputer


def preprocess(X, y, config: dict):
    # load data if no data supplied
    if not type(X) == pd.DataFrame:
        if config["name"] == "thyroid":
            df = pd.read_csv(
                "UIAYN_experiments/data/thyroid0387.data", delimiter=",", header=None
            )
            cols = [
                "age",
                "sex",
                "thyroxine",
                "q_thyroxine",
                "antithyroid_med",
                "sick",
                "pregnant",
                "surgery",
                "I131",
                "q_hypothyroid",
                "q_hyperthyroid",
                "lithium",
                "goitre",
                "tumor",
                "hypopituitary",
                "psych",
                "TSH_m",
                "TSH",
                "T3_m",
                "T3",
                "TT4_m",
                "TT4",
                "T4U_m",
                "T4U",
                "FTI_m",
                "FTI",
                "TBG_m",
                "TBG",
                "ref",
                "diagnosis",
            ]
            df.columns = cols
            y = df[["diagnosis"]]
            X = df.drop("diagnosis", axis=1)

    # drop features
    X = X.drop(config["drop"], axis=1)
    y = y.squeeze()
    # dataset specific preprocessing
    if config["name"] == "adult":
        X.loc[
            ~X["native-country"].isin(["United-States", "Mexico"]), "native-country"
        ] = "Other"
        y = y.replace({"<=50K": 0.0, "<=50K.": 0.0, ">50K": 1.0, ">50K.": 1.0})
    elif config["name"] == "credit":
        X = X.rename({"X2": "sex", "X5": "age"}, axis=1)
    elif config["name"] == "heart":
        y = y.apply(lambda x: 1 if x > 0 else 0)
    elif config["name"] == "student":
        X = X.rename({"Age at enrollment": "age", "Gender": "sex"}, axis=1)
    elif config["name"] == "obesity":
        X = X.rename({"Gender": "sex", "Age": "age"}, axis=1)
        # y = y.str.startswith("Obesity").map({True: "obesity", False: "no_obese"})
        pass
    elif config["name"] == "diabetes":
        y = y.apply(lambda x: x if x == "<30" else "No")
        X = X.rename({"gender": "sex"}, axis=1)

        for col in config["discrete"]:
            # fill missing categories (no missing numericals exist in the dataset)
            X[col] = X[col].fillna("missing")
            # frequency encode high cardinality features
            if X[col].nunique() > 15:
                X[col] = frequency_encode(X[col])

    elif config["name"] == "thyroid":
        y = y.apply(encode_diagnosis)
        X = X.replace({"?": np.nan})
        for col in config["numerical"]:
            data = X[[col]].astype(float)
            imputer = SimpleImputer()
            data = imputer.fit_transform(data)
            data = pd.DataFrame(data, columns=imputer.get_feature_names_out([col]))
            data = data.squeeze()
            X[col] = data

    # cast numericals
    X[config["numerical"]] = X[config["numerical"]].astype(float)
    # add y as target to df
    y.name = "target"
    df = pd.concat([X, y], axis=1)
    return df


def encode_diagnosis(row):
    if row[0] in ["A", "B", "C", "D"]:
        return "hyperthyroid"
    elif row[0] in ["E", "F", "G", "H"]:
        return "hypothyroid"
    elif row[0] in ["I", "J"]:
        return "binding protein"
    elif row[0] in ["K"]:
        return "general health"
    elif row[0] in ["L", "M", "N"]:
        return "replacement therapy"
    elif row[0] in ["R"]:
        return "discordant results"
    else:
        return "No condition"


def frequency_encode(series):
    freq_map = series.value_counts().to_dict()
    return series.map(freq_map)


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


def clear_dir(directory):
    # remove all files from specified directory
    for item in os.listdir(directory):
        filepath = os.path.join(directory, item)
        if os.path.isfile(filepath):
            os.remove(filepath)
