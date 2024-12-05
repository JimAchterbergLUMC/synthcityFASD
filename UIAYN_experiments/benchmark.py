from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.benchmark import Benchmarks
import os
import json
from ucimlrepo import fetch_ucirepo
from utils import preprocess, plot_df, clear_dir
from sklearn.model_selection import train_test_split

from synthcity.utils.optuna_sample import suggest_all
from synthcity.plugins import Plugins
import optuna


def tune(X_r, models, tune_params, path):
    # save hparams in separate files
    hparam_path = path
    clear_dir(hparam_path)

    # find approppriate hparam spaces
    hp_desired = {
        "fasd": [
            "fasd_n_units_embedding",
            "n_units_embedding",
            "decoder_n_layers_hidden",
            "decoder_n_units_hidden",
            "encoder_n_layers_hidden",
            "encoder_n_units_hidden",
        ],
        "adsgan": [
            "discriminator_n_layers_hidden",
            "discriminator_n_units_hidden",
            "generator_n_layers_hidden",
            "generator_n_units_hidden",
        ],
        "pategan": [
            "discriminator_n_layers_hidden",
            "discriminator_n_units_hidden",
            "generator_n_layers_hidden",
            "generator_n_units_hidden",
        ],
        "ctgan": [
            "discriminator_n_layers_hidden",
            "discriminator_n_units_hidden",
            "generator_n_layers_hidden",
            "generator_n_units_hidden",
        ],
        "tvae": [
            "n_units_embedding",
            "decoder_n_layers_hidden",
            "decoder_n_units_hidden",
            "encoder_n_layers_hidden",
            "encoder_n_units_hidden",
        ],
    }

    hp_desired = {key: value for key, value in hp_desired.items() if key in models}

    # make empty hparam files if not tuning to get default parameters
    if not tune_params:
        models = hp_desired.keys()
        for m in models:
            with open(f"{hparam_path}/hparams_{m}.json", "w") as file:
                json.dump({m: {}}, file)
        return None

    hp_space = {}
    for plugin, params in hp_desired.items():
        hp_ini = Plugins().get(plugin).hyperparameter_space()
        for hp_ in hp_ini:
            if hp_.name not in params:
                hp_ini = [x for x in hp_ini if x != hp_]

        hp_space[plugin] = hp_ini

    best_params = {}
    for plugin, hparams in hp_space.items():

        def objective(trial: optuna.Trial):
            params = suggest_all(trial, hparams)
            ID = f"trial_{trial.number}"

            if len(X_r) < 1000:
                params["batch_size"] = 32
                if plugin in ["tvae", "fasd", "ctgan", "adsgan"]:
                    params["n_iter_min"] = 10
                if plugin in ["pategan"]:
                    params["n_teachers"] = 5

            try:
                report = Benchmarks.evaluate(
                    [(ID, plugin, params)],
                    X_r,
                    task_type="classification",
                    repeats=1,
                    metrics={
                        "performance": [
                            "linear_model",
                            "mlp",
                            "xgb",
                        ],
                    },  # DELETE THIS LINE FOR ALL METRICS
                )
            except Exception as e:  # invalid set of params
                print(f"{type(e).__name__}: {e}")
                print(params)
                raise optuna.TrialPruned()
            score = report[ID].query('direction == "maximize"')["mean"].mean()
            # average score across all metrics with direction="minimize"
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=32)
        best_params[plugin] = study.best_params
        with open(f"{hparam_path}/hparams_{plugin}.json", "w") as file:
            json.dump({plugin: best_params[plugin]}, file)
        # after tuning a model, clear the workspace to free up space for the next one
        clear_dir("workspace")


def benchmark(ds, models, tune_params, metrics, repeats, split=1):
    # load data
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
    df = preprocess(X=X, y=y, config=config)

    # for testing purposes
    if split < 1:
        df, _ = train_test_split(
            df, stratify=df["target"], train_size=split, random_state=1
        )
    train_size = 0.8
    X_r = GenericDataLoader(
        data=df,
        sensitive_features=config["sensitive"],
        discrete_features=config["discrete"],
        target_column="target",
        random_state=0,
        train_size=train_size,
    )

    hparam_path = f"UIAYN_experiments/hparams/{ds}"
    if not os.path.exists(hparam_path):
        os.makedirs(hparam_path)

    if len(os.listdir(hparam_path)) == 0:
        # create hparam files, either through tuning or not, if there are no files yet
        tune(X_r, models, tune_params, hparam_path)

    # load plugin params from json files
    hparams = {}
    for file in os.listdir(hparam_path):
        with open(f"{hparam_path}/{file}", "r") as f:
            hparams.update(json.load(f))

    evaluate = [(plugin, plugin, params) for (plugin, params) in hparams.items()]

    for name, plugin, params in evaluate:
        if len(X_r) < 1000:
            params["batch_size"] = 32
            if plugin in ["tvae", "fasd", "ctgan", "adsgan"]:
                params["n_iter_min"] = 10
            if plugin in ["pategan"]:
                params["n_teachers"] = 5

    # perform benchmarking
    score = Benchmarks.evaluate(
        evaluate,
        X_r,
        task_type="classification",
        metrics=metrics,
        synthetic_size=len(X_r),  # int(len(X_r) * train_size)
        repeats=repeats,
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
    )

    # save benchmark results
    result_path = f"UIAYN_experiments/results/{ds}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for name, plugin, params in evaluate:
        score[name].to_csv(result_path + f"/{name}.csv")
