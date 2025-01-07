# stdlib
import platform
from typing import Any, Dict

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.dataset import NumpyDataset
from synthcity.plugins.core.models.convnet import suggest_image_classifier_arch
from synthcity.plugins.core.models.mlp import MLP
from synthcity.utils.reproducibility import clear_cache
from synthcity.utils.serialization import load_from_file, save_to_file
from synthcity.plugins.core.models.tabular_encoder import preprocess_prediction


class DetectionEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.DetectionEvaluator
        :parts: 1


    Train a SKLearn classifier to detect the synthetic data from real data.

    Synthetic and real data are combined to form a new dataset.
    K-fold cross validation is performed to see how well a classifier can distinguish real from synthetic.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "detection"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @staticmethod
    def name() -> str:
        raise NotImplementedError()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        raise NotImplementedError()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_detection_generic(
        self,
        model_template: Any,
        X_gt: DataLoader,
        X_syn: DataLoader,
        **kwargs: Any,  # required to specify whether preprocessing is required
    ) -> Dict:
        if "intermediate" in kwargs.keys():
            intermediate = kwargs["intermediate"]
            del kwargs["intermediate"]
        else:
            intermediate = False
        model_args = kwargs

        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            results = load_from_file(cache_file)
            log.info(
                f" Synthetic-real data discrimination using {self.name()}. AUCROC : {results}"
            )
            return results

        # if target column is discrete, we need to add it to the list of discrete columns for preprocessing
        discrete = X_gt.discrete_features.copy()
        if X_gt[X_gt.target_column].nunique() < 10:
            discrete.append(X_gt.target_column)

        gt, syn = X_gt.dataframe(), X_syn.dataframe()

        # ensure equal precision between numerical synthetic and real data, else XGB might find splits "between" precision discrepancies
        def count_decimals(x):
            if pd.isna(x) or isinstance(x, int):
                return 0
            text = str(x).split(".")
            if len(text) == 2:
                return len(text[1].rstrip("0"))
            return 0

        precision_dict = {}
        for col in [x for x in gt.columns if x not in discrete]:
            precision_dict[col] = gt[col].apply(count_decimals).max()
        for col, precision in precision_dict.items():
            syn[col] = syn[col].round(precision)

        # get data and labels
        data = pd.concat([gt, syn], ignore_index=True)
        labels_gt = np.asarray([0] * len(X_gt))
        labels_syn = np.asarray([1] * len(X_syn))
        labels = np.concatenate([labels_gt, labels_syn])
        res = []

        skf = StratifiedKFold(
            n_splits=self._n_folds, shuffle=True, random_state=self._random_state
        )
        for train_idx, test_idx in skf.split(data, labels):
            train_data = data.loc[train_idx]
            train_labels = labels[train_idx]
            test_data = data.loc[test_idx]
            test_labels = labels[test_idx]

            # preprocess, but not if eval detection is being used as patience metric
            if not intermediate:
                train_data, test_data = preprocess_prediction(
                    train=train_data, test=test_data, discrete_features=discrete
                )
            if "n_units_in" in model_args:
                model_args["n_units_in"] = train_data.shape[1]

            model = model_template(**model_args).fit(
                np.asarray(train_data).astype(float), train_labels
            )

            test_pred = model.predict_proba(np.asarray(test_data).astype(float))[:, 1]

            score = roc_auc_score(test_labels, test_pred)
            res.append(score)

        results = {self._reduction: float(self.reduction()(res))}
        log.info(
            f" Synthetic-real data discrimination using {self.name()}. AUCROC : {results}"
        )

        save_to_file(cache_file, results)

        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
        **kwargs,  # required to specify whether preprocessing is or isn't required
    ) -> float:
        return self.evaluate(X_gt, X_syn, **kwargs)[self._reduction]


class SyntheticDetectionXGB(DetectionEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.SyntheticDetectionXGB
        :parts: 1

    Train a XGBoostclassifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "detection_xgb"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        # TODO: investigate why XGBoost always has high AUCROC for the detection
        model_template = XGBClassifier
        model_args = {
            "n_jobs": 2,
            "verbosity": 0,
            "depth": 3,
            "random_state": self._random_state,
        }

        return self._evaluate_detection_generic(
            model_template, X_gt, X_syn, **model_args
        )


class SyntheticDetectionMLP(DetectionEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.SyntheticDetectionMLP
        :parts: 1

    Train a MLP classifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "detection_mlp"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_image_detection(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        clear_cache()

        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            results = load_from_file(cache_file)
            log.info(
                f" Synthetic-real data discrimination using {self.name()}. AUCROC : {results}"
            )
            return results

        data_gt = X_gt.numpy()
        data_syn = X_syn.numpy()
        data = np.concatenate([data_gt, data_syn], axis=0)

        labels_gt = np.asarray([0] * len(X_gt))
        labels_syn = np.asarray([1] * len(X_syn))
        labels = np.concatenate([labels_gt, labels_syn])

        skf = StratifiedKFold(
            n_splits=self._n_folds, shuffle=True, random_state=self._random_state
        )
        res = []
        for train_idx, test_idx in skf.split(data, labels):
            train_X = data[train_idx]
            train_y = labels[train_idx]
            test_X = data[test_idx]
            test_y = labels[test_idx]

            clf = suggest_image_classifier_arch(
                n_channels=X_gt.info()["channels"],
                height=X_gt.info()["height"],
                width=X_gt.info()["width"],
                classes=2,
            )
            train_dataset = NumpyDataset(train_X, train_y)

            clf.fit(train_dataset)
            test_pred = clf.predict_proba(torch.from_numpy(test_X))[:, 1].cpu().numpy()

            score = roc_auc_score(test_y, test_pred)
            res.append(score)

        results = {self._reduction: float(self.reduction()(res))}
        log.info(
            f" Synthetic-real data discrimination using {self.name()}. AUCROC : {results}"
        )

        save_to_file(cache_file, results)
        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
        **kwargs,  # required to specify whether preprocessing is required
    ) -> Dict:
        if X_gt.type() == "images":
            return self._evaluate_image_detection(X_gt, X_syn)

        model_args = {
            "task_type": "classification",
            "n_units_in": X_gt.shape[1],
            "n_units_out": 2,
            "random_state": self._random_state,
        }
        kwargs.update(model_args)
        return self._evaluate_detection_generic(
            MLP,
            X_gt,
            X_syn,
            **kwargs,  # required to specify whether preprocessing is required
        )


class SyntheticDetectionLinear(DetectionEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.SyntheticDetectionLinear
        :parts: 1

    Train a LogisticRegression classifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "detection_linear"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        model_args = {
            "random_state": self._random_state,
            "n_jobs": -1,
            "max_iter": 10000,
        }
        return self._evaluate_detection_generic(
            LogisticRegression,
            X_gt,
            X_syn,
            **model_args,
        )


class SyntheticDetectionGMM(DetectionEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.SyntheticDetectionGMM
        :parts: 1

    Train a GaussianMixture model to detect synthetic data.

    Returns:
        The average score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "detection_gmm"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        model_args = {
            "n_components": min(10, len(X_gt)),
            "random_state": self._random_state,
        }
        return self._evaluate_detection_generic(
            GaussianMixture,
            X_gt,
            X_syn,
            **model_args,
        )
