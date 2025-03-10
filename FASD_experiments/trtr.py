import json
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    LabelBinarizer,
    LabelEncoder,
    OrdinalEncoder,
)
from utils import preprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# simple ML script which determines Train Real Test Real (TRTR) utility.
# there seem to be some issues with the SynthCity implementation of this, taking the same training data each iteration.


def train_real_test_real(
    df,
    target_column,
    discrete_features,
    model_type,
    n_folds=10,
    test_size=0.2,
    **model_kwargs,
):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if model_type == XGBClassifier:
        ohe = OrdinalEncoder()
    else:
        # one hot encode for regression and neural net
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    ohe_features = (
        ohe.fit_transform(X[discrete_features]) if discrete_features else np.array([])
    )
    ohe_feature_names = (
        ohe.get_feature_names_out(discrete_features) if discrete_features else []
    )

    # Non-discrete features
    continuous_features = [col for col in X.columns if col not in discrete_features]
    X_continuous = X[continuous_features]

    # Combine transformed features
    if discrete_features:
        X_transformed = pd.DataFrame(
            ohe_features, columns=ohe_feature_names, index=X.index
        )
        X_transformed = pd.concat([X_transformed, X_continuous], axis=1)
    else:
        X_transformed = X_continuous

    # Binarize the target for multiclass AUC computation
    multiclass = True if len(np.unique(y)) > 2 else False

    if model_type == XGBClassifier:
        y = LabelEncoder().fit_transform(y)

    if multiclass:
        lb = LabelBinarizer()
        y_binarized = lb.fit_transform(y)

    auroc_scores = []

    # Perform random splits
    for fold in range(n_folds):
        # Split the data randomly into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=test_size, random_state=fold
        )

        # Scale continuous features separately for train and test
        if continuous_features:
            if model_type == XGBClassifier:
                X_train_continuous = X_train[continuous_features]
                X_test_continuous = X_test[continuous_features]
            else:
                # do minmax scaling for regression and neural net
                X_train_continuous = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
                    X_train[continuous_features]
                )
                X_test_continuous = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
                    X_test[continuous_features]
                )

            X_train.update(
                pd.DataFrame(
                    X_train_continuous, columns=continuous_features, index=X_train.index
                )
            )
            X_test.update(
                pd.DataFrame(
                    X_test_continuous, columns=continuous_features, index=X_test.index
                )
            )

        # Train a model (using Random Forest as an example)
        model = model_type(**model_kwargs)
        model.fit(X_train, y_train)

        # Predict probabilities
        y_pred_prob = model.predict_proba(X_test)

        # Compute AUC
        if multiclass:
            # Convert test labels to binarized form
            y_test_binarized = lb.transform(y_test)

            # Compute micro-averaged AUC
            auroc = roc_auc_score(
                y_test_binarized, y_pred_prob, average="micro", multi_class="ovr"
            )
        else:
            # Binary case: compute AUC for positive class
            auroc = roc_auc_score(y_test, y_pred_prob[:, 1])

        auroc_scores.append(auroc)

    # Report average and standard deviation of AUROC scores
    print(f"Average AUROC: {np.mean(auroc_scores):.4f}")
    print(f"Std of AUROC: {np.std(auroc_scores):.4f}")


# load data
ds = "heart"
with open("UIAYN_experiments/datasets.json", "r") as f:
    config = json.load(f)
config = config[ds]
dataset = fetch_ucirepo(id=config["id"])
X = dataset.data.features
y = dataset.data.targets

df = preprocess(X=X, y=y, config=config)

# required for MLP
# model_kwargs = {
#     "hidden_layer_sizes": (100,),
#     "batch_size": 32 if ds == "heart" else 200,
#     "max_iter": 1000,
#     "early_stopping": True,
#     "validation_fraction": 0.2,
#     "n_iter_no_change": 50,
#     "verbose": True,
# }

# required for XGBoost
model_kwargs = {
    "tree_method": "approx",
    "n_jobs": 2,
    "max_depth": 3,
}
train_real_test_real(
    df=df,
    target_column="target",
    discrete_features=config["discrete"],
    model_type=XGBClassifier,
    n_folds=10,
    test_size=0.2,
    **model_kwargs,
)
