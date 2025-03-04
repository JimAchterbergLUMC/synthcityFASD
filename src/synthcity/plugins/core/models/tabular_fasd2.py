# stdlib
from typing import Any, Callable, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from sklearn.preprocessing import OneHotEncoder
from torch import nn

# synthcity absolute
from synthcity.utils.constants import DEVICE
from synthcity.utils.samplers import BaseSampler, ConditionalDatasetSampler
from synthcity.plugins.core.dataloader import DataLoader

# synthcity relative
from .tabular_encoder import TabularEncoder
from .vae import VAE
from .fasd2 import FASD


class TabularFASD2(nn.Module):
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.tabular_vae.TabularVAE
        :parts: 1


    VAE for tabular data.

    This class combines VAE and tabular encoder to form a generative model for tabular data.

    Args:
        X: pd.DataFrame
            Reference dataset, used for training the tabular encoder
        cond: Optional
            Optional conditional
        decoder_n_layers_hidden: int
            Number of hidden layers in the decoder
        decoder_n_units_hidden: int
            Number of hidden units in each layer of the decoder
        decoder_nonlin: string, default 'elu'
            Nonlinearity to use in the decoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        decoder_n_iter: int
            Maximum number of iterations in the decoder.
        decoder_batch_norm: bool
            Enable/disable batch norm for the decoder
        decoder_dropout: float
            Dropout value. If 0, the dropout is not used.
        decoder_residual: bool
            Use residuals for the decoder
        encoder_n_layers_hidden: int
            Number of hidden layers in the encoder
        encoder_n_units_hidden: int
            Number of hidden units in each layer of the encoder
        encoder_nonlin: string, default 'relu'
            Nonlinearity to use in the encoder. Can be 'elu', 'relu', 'selu' or 'leaky_relu'.
        encoder_n_iter: int
            Maximum number of iterations in the encoder.
        encoder_batch_norm: bool
            Enable/disable batch norm for the encoder
        encoder_dropout: float
            Dropout value for the encoder. If 0, the dropout is not used.
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
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        X: pd.DataFrame,
        column_info: dict,
        n_units_embedding: int,
        task_type: str = "classification",
        batch_size: int = 100,
        n_iter: int = 500,
        random_state: int = 0,
        lr: float = 2e-4,
        weight_decay: float = 1e-3,
        # Decoder
        predictor_n_layers_hidden: int = 2,
        predictor_n_units_hidden: int = 250,
        predictor_nonlin: str = "leaky_relu",
        predictor_batch_norm: bool = False,
        predictor_dropout: float = 0,
        predictor_residual: bool = False,
        # Encoder
        encoder_n_layers_hidden: int = 3,
        encoder_n_units_hidden: int = 300,
        encoder_nonlin: str = "leaky_relu",
        encoder_batch_norm: bool = False,
        encoder_dropout: float = 0.1,
        # Loss parameters
        loss_factor: int = 2,
        device: Any = DEVICE,
        clipping_value: int = 1,
        # early stopping
        n_iter_min: int = 100,
        n_iter_print: int = 10,
        patience: int = 20,
    ) -> None:
        super(TabularFASD2, self).__init__()
        self.target_column = column_info["target_column"]
        self.discrete_columns = column_info["discrete_columns"]

        # separately encode X and y
        self.data_encoder = TabularEncoder(
            continuous_encoder="minmax", cont_encoder_params={"feature_range": (-1, 1)}
        )
        self.data_encoder.fit(
            X.drop(self.target_column, axis=1),
            discrete_columns=self.discrete_columns,
        )

        self.target_encoder = TabularEncoder(
            continuous_encoder="minmax",
            cont_encoder_params={"feature_range": (-1, 1)},
            categorical_limit=10,
        )
        self.target_encoder.fit((X[self.target_column]).to_frame())

        self.model = FASD(
            n_features=self.data_encoder.n_features(),
            n_units_embedding=n_units_embedding,
            task_type=task_type,
            batch_size=batch_size,
            n_iter=n_iter,
            random_state=random_state,
            lr=lr,
            weight_decay=weight_decay,
            # Predictor
            predictor_n_layers_hidden=predictor_n_layers_hidden,
            predictor_n_units_hidden=predictor_n_units_hidden,
            predictor_nonlin=predictor_nonlin,
            predictor_nonlin_out=self.target_encoder.activation_layout(
                discrete_activation="softmax", continuous_activation="tanh"
            ),
            predictor_batch_norm=predictor_batch_norm,
            predictor_dropout=predictor_dropout,
            predictor_residual=predictor_residual,
            # Decoder
            decoder_nonlin_out=self.data_encoder.activation_layout(
                discrete_activation="softmax", continuous_activation="tanh"
            ),
            # Encoder
            encoder_n_layers_hidden=encoder_n_layers_hidden,
            encoder_n_units_hidden=encoder_n_units_hidden,
            encoder_nonlin=encoder_nonlin,
            encoder_batch_norm=encoder_batch_norm,
            encoder_dropout=encoder_dropout,
            # Loss parameters
            loss_factor=loss_factor,
            device=device,
            clipping_value=clipping_value,
            # early stopping
            n_iter_min=n_iter_min,
            n_iter_print=n_iter_print,
            patience=patience,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(X)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.inverse_transform(X)

    def get_encoder(self) -> TabularEncoder:
        return self.encoder

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:

        X_enc = self.data_encoder.transform(X.drop(self.target_column, axis=1))
        y_enc = self.target_encoder.transform((X[self.target_column]).to_frame())
        self.model.fit(X_enc, y_enc)
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        # draw new samples -> hits generate method -> returns dataframe of (encoded) syndata with attached (soft) labels
        samples, labels = self(count)

        # remove preprocessing of input data
        samples = self.data_encoder.inverse_transform(samples)

        # attach targets (in original format)
        samples[self.target_column] = self.target_encoder.inverse_transform(labels)

        return samples

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        count: int,
    ):
        return self.model.generate(count)
