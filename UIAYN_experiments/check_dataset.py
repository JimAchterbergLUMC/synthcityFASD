import json
from ucimlrepo import fetch_ucirepo
from utils import preprocess
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


def skf_classification(X, y, config, model_template, k, model_args):
    model = model_template(**model_args)
    X_ = []
    for col in X.columns:
        if col in config["discrete"]:
            encoder = OneHotEncoder(sparse_output=False)
            data = encoder.fit_transform(X[[col]])
            columns = encoder.get_feature_names_out([col])
            data = pd.DataFrame(data, columns=columns)
        else:
            data = X[[col]]

        data.columns = data.columns.str.replace(r"[<>\[\]]", "_", regex=True)
        X_.append(data)
    X = pd.concat(X_, axis=1)
    y = pd.Series(LabelEncoder().fit_transform(y.squeeze()))

    skf = StratifiedKFold(n_splits=k, random_state=0, shuffle=True)
    scores = []

    for tr, te in skf.split(X, y):
        X_tr = X.loc[tr, :].reset_index(drop=True)
        X_te = X.loc[te, :].reset_index(drop=True)
        y_tr = y[tr].reset_index(drop=True)
        y_te = y[te].reset_index(drop=True)

        # scale numericals
        for x in [X_tr, X_te]:
            encoder = MinMaxScaler((-1, 1))
            data = encoder.fit_transform(x[config["numerical"]])
            data = pd.DataFrame(
                data, columns=encoder.get_feature_names_out(config["numerical"])
            )
            x[config["numerical"]] = data

        model = XGBClassifier()
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_te)

        if y.nunique() > 2:
            score = roc_auc_score(
                y_te,
                preds,
                multi_class="ovr",
                average="micro",
            )
        else:
            score = roc_auc_score(y_te, preds[:, 1])
        scores.append(score)
    return scores


ds = "heart"
with open("UIAYN_experiments/datasets.json", "r") as f:
    config = json.load(f)
config = config[ds]

if config["loadable"] == "yes":
    dataset = fetch_ucirepo(id=config["id"])
    X = dataset.data.features
    y = dataset.data.targets
else:
    X = None
    y = None

df = preprocess(X, y, config)

y = df.target
X = df.drop("target", axis=1)

# print(X.age.describe())
# print(X.sex.value_counts(normalize=True))


print(y.value_counts())

exit()

scores = skf_classification(X, y, config, LogisticRegression, k=5, model_args={})
print(scores)
