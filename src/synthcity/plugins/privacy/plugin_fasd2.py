# stdlib
from pathlib import Path
from typing import Any, List, Optional, Union

# third party
import numpy as np
import pandas as pd

# Necessary packages
from pydantic import validate_arguments
from torch.utils.data import sampler

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
from synthcity.plugins.core.models.tabular_fasd2 import TabularFASD2
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.utils.constants import DEVICE


class FASD2Plugin(Plugin):
    """
    .. inheritance-diagram:: synthcity.plugins.generic.plugin_tvae.TVAEPlugin
        :parts: 1

    Tabular VAE implementation.

    Args:
        decoder_n_layers_hidden: int
            Number of hidden layers in the decoder
        decoder_n_units_hidden: int
            Number of hidden units in each layer of the decoder
        decoder_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the decoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        decoder_dropout: float
            Dropout value. If 0, the dropout is not used.
        encoder_n_layers_hidden: int
            Number of hidden layers in the encoder
        encoder_n_units_hidden: int
            Number of hidden units in each layer of the encoder
        encoder_nonlin: string, default 'leaky_relu'
            Nonlinearity to use in the encoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        encoder_dropout: float
            Dropout value for the encoder. If 0, the dropout is not used.
        n_iter: int
            Maximum number of iterations in the encoder.
        lr: float
            learning rate for optimizer.
        weight_decay: float
            l2 (ridge) penalty for the weights.
        batch_size: int
            Batch size
        random_state: int
            random_state used
        encoder_max_clusters: int
            The max number of clusters to create for continuous columns when encoding
        # early stopping
        n_iter_print: int
            Number of iterations after which to print updates and check the validation loss.
        n_iter_min: int
            Minimum number of iterations to go through before starting early stopping
        patience: int
            Max number of iterations without any improvement before early stopping is trigged.
        # Core Plugin arguments
        workspace: Path.
            Optional Path for caching intermediary results.
        compress_dataset: bool. Default = False.
            Drop redundant features before training the generator.
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.


    Example:
        >>> from sklearn.datasets import load_iris
        >>> from synthcity.plugins import Plugins
        >>>
        >>> X, y = load_iris(as_frame = True, return_X_y = True)
        >>> X["target"] = y
        >>>
        >>> plugin = Plugins().get("tvae", n_iter = 100)
        >>> plugin.fit(X)
        >>>
        >>> plugin.generate(50)

    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_iter: int = 1000,
        task_type="classification",
        n_units_embedding: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 200,
        random_state: int = 0,
        encoder_n_layers_hidden: int = 2,
        encoder_n_units_hidden: int = 100,
        encoder_nonlin: str = "relu",
        encoder_dropout: float = 0.1,
        predictor_n_layers_hidden: int = 0,
        predictor_n_units_hidden: int = 0,
        predictor_nonlin: str = "none",
        predictor_dropout: float = 0,
        loss_factor: int = 10,
        clipping_value: int = 1,
        n_iter_print: int = 10,
        n_iter_min: int = 30,
        patience: int = 50,
        # core plugin arguments
        device: Any = DEVICE,
        workspace: Path = Path("workspace"),
        compress_dataset: bool = False,
        sampling_patience: int = 500,
        **kwargs: Any
    ) -> None:
        super().__init__(
            device=device,
            random_state=random_state,
            sampling_patience=sampling_patience,
            workspace=workspace,
            compress_dataset=compress_dataset,
            **kwargs
        )

        self.n_units_embedding = n_units_embedding
        self.task_type = task_type
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.random_state = random_state
        self.lr = lr
        self.weight_decay = weight_decay
        # Predictor
        self.predictor_n_layers_hidden = predictor_n_layers_hidden
        self.predictor_n_units_hidden = predictor_n_units_hidden
        self.predictor_nonlin = predictor_nonlin
        self.predictor_batch_norm = False
        self.predictor_dropout = predictor_dropout
        self.predictor_residual = False
        # Encoder
        self.encoder_n_layers_hidden = encoder_n_layers_hidden
        self.encoder_n_units_hidden = encoder_n_units_hidden
        self.encoder_nonlin = encoder_nonlin
        self.encoder_batch_norm = False
        self.encoder_dropout = encoder_dropout
        # Loss parameters
        self.loss_factor = loss_factor
        self.device = device
        self.clipping_value = clipping_value
        # early stopping
        self.n_iter_min = n_iter_min
        self.n_iter_print = n_iter_print
        self.patience = patience

    @staticmethod
    def name() -> str:
        return "fasd2"

    @staticmethod
    def type() -> str:
        return "privacy"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return [
            IntegerDistribution(name="n_units_embedding", low=50, high=250, step=50),
            CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
            CategoricalDistribution(name="loss_factor", choices=[0.1, 1, 2, 5, 10]),
            CategoricalDistribution(name="weight_decay", choices=[1e-3, 1e-4]),
            CategoricalDistribution(name="batch_size", choices=[64, 128, 256, 512]),
            IntegerDistribution(name="encoder_n_layers_hidden", low=1, high=3),
            IntegerDistribution(
                name="encoder_n_units_hidden", low=50, high=250, step=50
            ),
            CategoricalDistribution(
                name="encoder_nonlin",
                choices=["relu", "leaky_relu", "tanh", "elu"],
            ),
            FloatDistribution(name="encoder_dropout", low=0, high=0.2),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "FASD2Plugin":

        self.model = TabularFASD2(
            X=X.dataframe(),
            column_info={
                "discrete_columns": X.discrete_features,
                "target_column": X.target_column,
            },
            n_units_embedding=self.n_units_embedding,
            task_type=self.task_type,
            batch_size=self.batch_size,
            n_iter=self.n_iter,
            random_state=self.random_state,
            lr=self.lr,
            weight_decay=self.weight_decay,
            # Predictor
            predictor_n_layers_hidden=self.predictor_n_layers_hidden,
            predictor_n_units_hidden=self.predictor_n_units_hidden,
            predictor_nonlin=self.predictor_nonlin,
            predictor_batch_norm=self.predictor_batch_norm,
            predictor_dropout=self.predictor_dropout,
            predictor_residual=self.predictor_residual,
            # Encoder
            encoder_n_layers_hidden=self.encoder_n_layers_hidden,
            encoder_n_units_hidden=self.encoder_n_units_hidden,
            encoder_nonlin=self.encoder_nonlin,
            encoder_batch_norm=self.encoder_batch_norm,
            encoder_dropout=self.encoder_dropout,
            # Loss parameters
            loss_factor=self.loss_factor,
            device=self.device,
            clipping_value=self.clipping_value,
            # early stopping
            n_iter_min=self.n_iter_min,  # self.n_iter_min,
            n_iter_print=self.n_iter_print,
            patience=self.patience,  # self.patience,
        )
        self.model.fit(X.dataframe(), **kwargs)

        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.generate, count, syn_schema)


plugin = FASD2Plugin
