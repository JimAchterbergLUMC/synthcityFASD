from benchmark import benchmark
from utils import clear_dir

if __name__ == "__main__":
    for ds in ["heart", "student", "credit", "adult"]:
        models = [
            "dpgan",
            "fasd",
            "tvae",
            "ctgan",
            "adsgan",
        ]

        tune_params = False
        repeats = 10
        metrics = {
            "stats": [
                "jensenshannon_dist",
                "alpha_precision",
            ],
            "performance": ["linear_model", "mlp", "xgb", "feat_rank_distance"],
            "detection": ["detection_xgb"],
            "privacy": [
                "delta-presence",
                "k-map",
                "identifiability_score",
                "DomiasMIA_BNAF",
            ],
            "attack": ["data_leakage_xgb"],
        }

        benchmark(ds, models, tune_params, metrics, repeats, split=1)
        clear_dir("workspace")
