from benchmark import benchmark

if __name__ == "__main__":
    ds = "heart"
    models = [
        "fasd",
        "pategan",
        "tvae",
        "ctgan",
        "adsgan",
    ]

    tune_params = True
    repeats = 10
    metrics = {
        "stats": [
            "jensenshannon_dist",
            "max_mean_discrepancy",
            "wasserstein_dist",
            "alpha_precision",
        ],
        "performance": ["linear_model", "mlp", "xgb", "feat_rank_distance"],
        "detection": ["detection_xgb"],
        "privacy": [
            "delta-presence",
            "k-anonymization",
            "k-map",
            "distinct l-diversity",
            "identifiability_score",
        ],
        "attack": ["data_leakage_xgb"],
    }
    benchmark(ds, models, tune_params, metrics, repeats, split=1)
