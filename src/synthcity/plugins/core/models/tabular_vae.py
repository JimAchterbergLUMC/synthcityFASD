# stdlib
from typing import Any, Optional, Union

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
from synthcity.utils.dataframe import discrete_columns as find_cat_cols

# synthcity relative
from .tabular_encoder import TabularEncoder
from .vae import VAE
from .fasd import FASD_NN, FASD_Decoder


class TabularVAE(nn.Module):
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.tabular_vae.TabularVAE
        :parts: 1


    VAE for tabular data.

    This class combines VAE and tabular encoder to form a generative model for tabular data.

    Args:
        X: pd.DataFrame
            Reference dataset, used for training the tabular encoder
        fasd: bool
            Whether to use fidelity agnostic synthetic data generation
        fasd_args: dict
            When using fasd, dictionary of parameters relating to fasd.
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
        fasd: bool,
        n_units_embedding: int,
        fasd_args: dict = {},
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        lr: float = 2e-4,
        n_iter: int = 500,
        weight_decay: float = 1e-3,
        batch_size: int = 64,
        random_state: int = 0,
        loss_strategy: str = "standard",
        encoder_max_clusters: int = 20,
        decoder_n_layers_hidden: int = 2,
        decoder_n_units_hidden: int = 250,
        decoder_nonlin: str = "leaky_relu",
        decoder_nonlin_out_discrete: str = "softmax",
        decoder_nonlin_out_continuous: str = "tanh",
        decoder_batch_norm: bool = False,
        decoder_dropout: float = 0,
        decoder_residual: bool = True,
        encoder_n_layers_hidden: int = 3,
        encoder_n_units_hidden: int = 300,
        encoder_nonlin: str = "leaky_relu",
        encoder_batch_norm: bool = False,
        encoder_dropout: float = 0.1,
        encoder_whitelist: list = [],
        device: Any = DEVICE,
        robust_divergence_beta: int = 2,  # used for loss_strategy = robust_divergence
        loss_factor: int = 1,  # used for standar losss
        dataloader_sampler: Optional[BaseSampler] = None,
        clipping_value: int = 1,
        # early stopping
        n_iter_min: int = 100,
        n_iter_print: int = 10,
        patience: int = 20,
    ) -> None:
        super(TabularVAE, self).__init__()
        categorical_limit = 20
        self.fasd = fasd
        n_units_conditional = 0
        self.cond_encoder: Optional[OneHotEncoder] = None

        if self.fasd:
            self.tab_encoder = TabularEncoder(
                continuous_encoder="minmax",
                cont_encoder_params={"feature_range": (0, 1)},
                whitelist=encoder_whitelist,
                categorical_limit=categorical_limit,
            )
        else:
            self.tab_encoder = TabularEncoder(
                max_clusters=encoder_max_clusters,
                whitelist=encoder_whitelist,
                categorical_limit=categorical_limit,
            )

        # as in original implementation, discrete columns are found through find_cat_cols function by encoder itself
        self.tab_encoder = self.tab_encoder.fit(X)
        X_enc = self.tab_encoder.transform(X)

        if self.fasd:
            # find initial discrete columns, required in decoder later
            discrete_cols = find_cat_cols(dataframe=X, max_classes=categorical_limit)
            self.discrete_cols = [x for x in discrete_cols if x != "target"]

            # train fasd encoder and get representations
            X_enc = self.fasd_generate_(
                X_enc=X_enc,
                fasd_args=fasd_args,
                random_state=random_state,
            )
            # potentially scale representations to prevent numerical instability (exploding gradients) during training
            self.encoder = TabularEncoder(
                continuous_encoder="standard",
                cont_encoder_params={},
                categorical_encoder="passthrough",
                cat_encoder_params={},
            ).fit(X_enc)
            X_enc = self.encoder.transform(X_enc)
            # set raw data as the generated representations, needed for conditionals
            X = X_enc.copy()
            # store a copy of the representations, needed to override the input data in the fit method
            self.X_rep = X_enc.copy()
        else:
            self.encoder = self.tab_encoder

        def _cond_loss(
            real_samples: torch.tensor,
            fake_samples: torch.Tensor,
            cond: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if cond is None or self.predefined_conditional:
                return 0

            losses = []

            idx = 0
            cond_idx = 0

            for item in self.encoder.layout():
                length = item.output_dimensions

                if item.feature_type != "discrete":
                    idx += length
                    continue

                # create activate feature mask
                mask = cond[:, cond_idx : cond_idx + length].sum(axis=1).bool()

                if mask.sum() == 0:
                    idx += length
                    continue

                if not (fake_samples[mask, idx : idx + length] >= 0).all():
                    raise RuntimeError(
                        f"Values should be positive after softmax = {fake_samples[mask, idx : idx + length]}"
                    )
                # fake_samples are after the Softmax activation
                # we filter active features in the mask
                item_loss = torch.nn.NLLLoss()(
                    torch.log(fake_samples[mask, idx : idx + length] + 1e-8),
                    torch.argmax(real_samples[mask, idx : idx + length], dim=1),
                )
                losses.append(item_loss)

                cond_idx += length
                idx += length

            if idx != real_samples.shape[1]:
                raise RuntimeError(f"Invalid offset {idx} {real_samples.shape}")

            if len(losses) == 0:
                return 0

            loss = torch.stack(losses, dim=-1)
            return loss.sum() / len(real_samples)

        if cond is not None:
            cond = np.asarray(cond)
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)

            self.cond_encoder = OneHotEncoder(handle_unknown="ignore").fit(cond)
            cond = self.cond_encoder.transform(cond).toarray()

            n_units_conditional = cond.shape[-1]

        self.predefined_conditional = cond is not None

        if (
            dataloader_sampler is None and not self.predefined_conditional
        ):  # don't mix conditionals
            dataloader_sampler = ConditionalDatasetSampler(
                self.encoder.transform(X),
                self.encoder.layout(),
            )
            n_units_conditional = dataloader_sampler.conditional_dimension()

        self.dataloader_sampler = dataloader_sampler

        self.model = VAE(
            self.encoder.n_features(),
            n_units_embedding=n_units_embedding,
            n_units_conditional=n_units_conditional,
            batch_size=batch_size,
            n_iter=n_iter,
            lr=lr,
            weight_decay=weight_decay,
            random_state=random_state,
            loss_strategy=loss_strategy,
            decoder_n_layers_hidden=decoder_n_layers_hidden,
            decoder_n_units_hidden=decoder_n_units_hidden,
            decoder_nonlin=decoder_nonlin,
            decoder_nonlin_out=self.encoder.activation_layout(
                discrete_activation=decoder_nonlin_out_discrete,
                continuous_activation=decoder_nonlin_out_continuous,
            ),
            decoder_batch_norm=decoder_batch_norm,
            decoder_dropout=decoder_dropout,
            decoder_residual=decoder_residual,
            encoder_n_units_hidden=encoder_n_units_hidden,
            encoder_n_layers_hidden=encoder_n_layers_hidden,
            encoder_nonlin=encoder_nonlin,
            encoder_batch_norm=encoder_batch_norm,
            encoder_dropout=encoder_dropout,
            dataloader_sampler=dataloader_sampler,
            device=device,
            extra_loss_cbks=[_cond_loss],
            robust_divergence_beta=robust_divergence_beta,
            loss_factor=loss_factor,
            clipping_value=clipping_value,
            n_iter_print=n_iter_print,
            n_iter_min=n_iter_min,
            patience=patience,
        )

    def fasd_generate_(
        self,
        X_enc: pd.DataFrame,
        fasd_args: dict,
        random_state: int,
    ):
        X_enc = X_enc.copy()
        # retrieve y from X (target name should always be 'target')
        self.target_columns = [
            col for col in X_enc.columns if col.startswith("target_")
        ]
        if len(self.target_columns) == 0:
            raise Exception("Please ensure the target column is named target")

        y = X_enc[self.target_columns]
        X_enc = X_enc.drop(self.target_columns, axis=1)
        X_ori = X_enc.copy()

        # instantiate and train model
        fasd_nn = FASD_NN(
            input_dim=X_enc.shape[1],
            hidden_dim=fasd_args["hidden_dim"],
            num_classes=y.shape[1],
            random_state=random_state,
            checkpoint_dir="workspace",
            val_split=0.2,
            latent_activation=nn.Identity(),
        )

        # train neural network to predict targets from encoded input
        fasd_nn.train_model(
            X=X_enc,
            y=y,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(fasd_nn.parameters(), lr=0.001),
            batch_size=fasd_args["batch_size"],
            num_epochs=fasd_args["num_epochs"],
        )

        # pass encoded input data through encoder to create continuous representations (now considered the raw data, X)
        X_enc = fasd_nn.encoder.encode(X_enc)

        self.fasd_nn = fasd_nn

        self.fasd_decode_(
            X_rep=X_enc,
            X_raw=X_ori,
            fasd_args=fasd_args,
            random_state=random_state,
        )

        return X_enc

    def fasd_decode_(
        self,
        X_rep: pd.DataFrame,
        X_raw: pd.DataFrame,
        fasd_args: dict,
        random_state: int,
    ):
        X_rep, X_raw = X_rep.copy(), X_raw.copy()

        # find indices of discrete and continuous features in encoded space
        def find_discrete_ohe_features(ori_discrete_columns, df_cols):
            discrete = []
            for col_dis in ori_discrete_columns:
                cur_discrete = []
                for num, col_df in enumerate(df_cols):
                    if col_df.startswith(col_dis):
                        cur_discrete.append(num)
                if len(cur_discrete) > 0:
                    discrete.append(cur_discrete)
            return discrete

        if bool(set(X_raw.columns) & (set(self.target_columns))):
            raise Exception("please pass raw data without target features")

        discr_idx = find_discrete_ohe_features(
            ori_discrete_columns=self.discrete_cols,
            df_cols=X_raw.columns,
        )
        cont_idx = [
            x
            for x in list(range(len(X_raw.columns)))
            if x not in [item for list in discr_idx for item in list]
        ]

        # train decoder to decode representations from representation space to raw feature space
        fasd_decoder = FASD_Decoder(
            input_dim=X_rep.shape[1],
            cont_idx=cont_idx,
            cat_idx=discr_idx,
            checkpoint_dir="workspace",
            val_split=0.2,
            random_state=random_state,
        )
        # fasd_decoder_args = {
        #     "criterion_cont": nn.MSELoss(),
        #     "criterion_cat": nn.CrossEntropyLoss(),
        #     "optimizer": torch.optim.Adam(fasd_decoder.parameters(), lr=0.001),
        #     "num_epochs": fasd_args["num_epochs"],
        #     "batch_size": fasd_args["batch_size"],
        # }
        fasd_decoder_args = {
            "criterion": nn.MSELoss(),
            "optimizer": torch.optim.Adam(fasd_decoder.parameters(), lr=0.001),
            "num_epochs": fasd_args["num_epochs"],
            "batch_size": fasd_args["batch_size"],
        }
        fasd_decoder.train_model(
            X=X_rep,
            y=X_raw,
            **fasd_decoder_args,
        )

        self.fasd_decoder = fasd_decoder

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
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Any:

        if self.fasd:
            X_enc = self.X_rep.copy()
        else:
            X_enc = self.encode(X)

        if cond is not None and self.cond_encoder is not None:
            cond = np.asarray(cond)
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)

            cond = self.cond_encoder.transform(cond).toarray()

        if not self.predefined_conditional and self.dataloader_sampler is not None:
            cond = self.dataloader_sampler.get_dataset_conditionals()

        if cond is not None:
            if len(cond) != len(X_enc):
                raise ValueError(
                    f"Invalid conditional shape. {cond.shape} expected {len(X_enc)}"
                )

        self.model.fit(X_enc, cond, **kwargs)
        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(
        self,
        count: int,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        # draw new samples
        samples = pd.DataFrame(self(count, cond))

        if self.fasd:
            # predict targets from synthetic representations (after first decoding back from (0,1) tabular encoding)
            y = self.fasd_nn.predictor.predict(self.encoder.inverse_transform(samples))

            # decode synthetic representations to original data space
            samples = self.fasd_decoder.decode(samples)

            # attach predicted y to synthetic data
            samples[self.target_columns] = y

        # decode tabular encoding of the original data
        samples = self.tab_encoder.inverse_transform(samples)

        return samples

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        count: int,
        cond: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    ) -> torch.Tensor:
        if cond is not None and self.cond_encoder is not None:
            cond = np.asarray(cond)
            if len(cond.shape) == 1:
                cond = cond.reshape(-1, 1)

            cond = self.cond_encoder.transform(cond).toarray()

        if not self.predefined_conditional and self.dataloader_sampler is not None:
            cond = self.dataloader_sampler.sample_conditional(count)

        return self.model.generate(count, cond=cond)
