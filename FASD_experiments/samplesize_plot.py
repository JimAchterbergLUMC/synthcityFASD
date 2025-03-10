from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np
from synthcity.metrics.eval_performance import PerformanceEvaluatorXGB
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from utils import preprocess
from ucimlrepo import fetch_ucirepo
import json
from utils import clear_dir
import random
from synthcity.utils.reproducibility import enable_reproducible_results


def _get_results(ds, sample_sizes, models, n_runs):

    with open("UIAYN_experiments/datasets.json", "r") as f:
        config = json.load(f)
    config = config[ds]
    dataset = fetch_ucirepo(id=config["id"])
    X = dataset.data.features
    y = dataset.data.targets

    df = preprocess(X=X, y=y, config=config)

    # Experiment settings
    train_size = 0.8
    all_results = {model: {str(N): [] for N in sample_sizes} for model in models}

    # Run experiment for multiple models and random states
    hparam_path = f"UIAYN_experiments/hparams/{ds}"
    for model_name in models:
        hparams = {}
        with open(f"{hparam_path}/hparams_{model_name}.json", "r") as f:
            hparams.update(json.load(f))
        hparams = hparams[model_name]
        repeats_list = list(range(n_runs))
        random.shuffle(repeats_list)
        for random_state in repeats_list:
            enable_reproducible_results(random_state)
            X = GenericDataLoader(
                data=df,
                sensitive_features=config["sensitive"],
                discrete_features=config["discrete"],
                target_column="target",
                random_state=random_state,
                train_size=train_size,
            )
            hparams["random_state"] = random_state

            # Fit SD generator
            model = Plugins().get(model_name, **hparams)
            model.fit(X.train())

            # Generate SD and evaluate TSTR
            for N in sample_sizes:
                syn = model.generate(N)
                X_syn = GenericDataLoader(
                    data=syn.dataframe(),
                    sensitive_features=config["sensitive"],
                    discrete_features=config["discrete"],
                    target_column="target",
                    random_state=random_state,
                    train_size=train_size,
                )
                result = PerformanceEvaluatorXGB().evaluate(X.test(), X_syn)
                all_results[model_name][str(N)].append(result["syn_ood"])
            clear_dir("workspace")
    return all_results


ds = "adult"
n_runs = 10
models = [
    # "dpgan",
    "fasd",
    "tvae",
    "ctgan",
    "adsgan",
]


# Generate 10 values in log space between 5k-250k
sample_sizes = np.logspace(np.log10(5e3), np.log10(250e3), num=10, base=10, dtype=int)

all_results = _get_results(
    ds=ds,
    sample_sizes=sample_sizes,
    models=models,
    n_runs=n_runs,
)

with open("UIAYN_experiments/samplesize_plot.json", "w") as f:
    json.dump(all_results, f)


with open("UIAYN_experiments/samplesize_plot.json", "r") as f:
    all_results = json.load(f)

# map model names and sample sizes to correct names for plot
mapper = {"fasd": "FASD", "adsgan": "AdsGAN", "ctgan": "CTGAN", "tvae": "TVAE"}
models = [mapper[k] for k in models]
all_results = {mapper.get(k, k): v for k, v in all_results.items()}
sample_sizes_mapper = {}
for s in sample_sizes:
    sample_sizes_mapper.update({str(s): str(s / 9768)})
sample_sizes = [(sample_sizes_mapper.get(str(item), item)) for item in sample_sizes]
all_results = {
    outer_key: {
        sample_sizes_mapper.get(inner_key, inner_key): value
        for inner_key, value in inner_dict.items()
    }
    for outer_key, inner_dict in all_results.items()
}

# Plot results
fig, axs = plt.subplots(figsize=(10, 5))

for model_name in models:
    means = [np.mean(all_results[model_name][str(N)]) for N in sample_sizes]
    stds = [np.std(all_results[model_name][str(N)]) for N in sample_sizes]
    sns.lineplot(
        x=np.array(sample_sizes, dtype=float),
        y=means,
        label=model_name,
        ax=axs,
        marker="o",
    )
    axs.fill_between(
        np.array(sample_sizes, dtype=float),
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        alpha=0.3,
    )

plt.axhline(y=0.927, color="b", linestyle="--", label="Train Real Test Real")
plt.axvline(x=1, color="gray", linestyle="--", label="Real Data Test Size")

# Put x-axis on log scale

plt.xscale("log")
ticks = [0.5, 1, 2, 4, 8, 16, 32]
plt.xticks(ticks, ticks)
plt.minorticks_off()
plt.xlabel(r"$N_{Synthetic}/N_{Real}$", fontsize=12)
plt.ylabel("Train Synthetic Test Real (AUROC)", fontsize=12)


# plt.ylim((0.83, 0.95))
plt.legend(loc="upper right", fontsize=12, ncols=3)
plt.savefig("UIAYN_experiments/results/samplesplot.pdf")
