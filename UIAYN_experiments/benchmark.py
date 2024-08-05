from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
import os
import json
from ucimlrepo import fetch_ucirepo
from utils import preprocess, plot_df

# TBD:
# - should MIA be added?


ds = "adult"
with open(f"UIAYN_experiments/datasets.json", "r") as f:
    config = json.load(f)
config = config[ds]
dataset = fetch_ucirepo(id=config["id"])
X = dataset.data.features
y = dataset.data.targets

df = preprocess(X=X, y=y, config=config)
df = df[:100]

# we have to make sure that the categorical limit corresponds to what we find discrete features in the dataset
# print(df.nunique())
# plot_df(df)


# setup dataloader
X_r = GenericDataLoader(
    data=df,
    sensitive_features=config["sensitive"],
    target_column="target",
    random_state=0,
    train_size=0.8,
)


# tvae_kwargs = {
#     "n_iter": 300,
#     "n_units_embedding": 128,
#     "lr": 0.001,
#     "weight_decay": 1e-05,
#     "batch_size": 500,
#     "random_state": 0,
#     "decoder_n_layers_hidden": 2,
#     "decoder_n_units_hidden": 128,
#     "decoder_nonlin": "relu",
#     "decoder_dropout": 0,
#     "encoder_n_layers_hidden": 2,
#     "encoder_n_units_hidden": 128,
#     "encoder_nonlin": "relu",
#     "encoder_dropout": 0,
#     "loss_factor": 2,
#     "data_encoder_max_clusters": 10,
#     "clipping_value": 1,
#     "n_iter_print": 50,
#     "n_iter_min": 100,
#     "patience": 5,
#     # "device": device(type="cpu"),
#     "workspace": "workspace",
#     "compress_dataset": False,
#     "sampling_patience": 500,
# }
tvae_kwargs = {}
fasd_args = {
    "hidden_dim": 64,
    "num_epochs": 300,
    "batch_size": 500,
}
score = Benchmarks.evaluate(
    [
        (
            "FASD",
            "tvae",
            {
                "fasd": True,
                "fasd_args": fasd_args,
                **tvae_kwargs,
            },
        ),
        (
            "ARF",
            "arf",
            {},
        ),
        (
            "TVAE",
            "tvae",
            {"fasd": False, **tvae_kwargs},
        ),
        (
            "CTGAN",
            "ctgan",
            {},
        ),
        (
            "BN",
            "bayesian_network",
            {},
        ),
        (
            "NFLOW",
            "nflow",
            {},
        ),
    ],
    X_r,
    task_type="classification",
    metrics={
        # "sanity": [
        #     "data_mismatch",
        #     "common_rows_proportion",
        #     "nearest_syn_neighbor_distance",
        #     "close_values_probability",
        #     "distant_values_probability",
        # ],
        "stats": [
            "jensenshannon_dist",
            # "chi_squared_test",
            # "feature_corr",
            # "inv_kl_divergence",
            # "ks_test",
            "max_mean_discrepancy",
            # "wasserstein_dist",
            # "prdc",
            "alpha_precision",
            # "survival_km_distance",
        ],
        "performance": [  # "linear_model", "mlp",
            "xgb",
            # "feat_rank_distance"
        ],
        "detection": [
            "detection_xgb",
            # "detection_mlp",
            # "detection_gmm",
            # "detection_linear",
        ],
        "privacy": [
            "delta-presence",
            "k-anonymization",
            "k-map",
            "distinct l-diversity",
            "identifiability_score",
            # "DomiasMIA_BNAF",
            # "DomiasMIA_KDE",
            # "DomiasMIA_prior",
        ],
        "attack": [  # "data_leakage_linear",
            "data_leakage_xgb",
            # "data_leakage_mlp"
        ],
    },
    synthetic_size=len(df),
    repeats=2,
    synthetic_cache=False,
    synthetic_reuse_if_exists=False,
    use_metric_cache=False,
)

result_path = f"results/{ds}"
if not os.path.exists(result_path):
    os.makedirs(result_path)

for model in ["FASD", "ARF", "TVAE", "CTGAN", "BN", "NFLOW"]:
    score[model].to_csv(result_path + f"/{model}.csv")
