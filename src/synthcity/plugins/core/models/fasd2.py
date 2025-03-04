# stdlib
from typing import Any, Callable, List, Optional, Tuple

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, sampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd

# synthcity absolute
import synthcity.logger as log
from synthcity.utils.constants import DEVICE

# synthcity relative
from .mlp import MLP


class FASD_Encoder(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_in: int,
        n_units_embedding: int,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        random_state: int = 0,
        dropout: float = 0.1,
        batch_norm: bool = True,
        residual: bool = False,
        device: Any = DEVICE,
    ) -> None:
        super(FASD_Encoder, self).__init__()
        self.device = device
        self.shared = MLP(
            task_type="regression",
            n_units_in=n_units_in,
            n_units_out=n_units_hidden,
            n_units_hidden=n_units_hidden,
            n_layers_hidden=n_layers_hidden - 1,
            nonlin=nonlin,
            random_state=random_state,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
            device=device,
        ).to(self.device)

        self.mu_fc = nn.Linear(n_units_hidden, n_units_embedding).to(self.device)
        self.logvar_fc = nn.Linear(n_units_hidden, n_units_embedding).to(self.device)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        X: Tensor,
    ):
        shared = self.shared(X)
        mu = self.mu_fc(shared)
        logvar = self.logvar_fc(shared)
        # clamp logvar to avoid numerical instability after exponentiation. [-30,20] inspired from HuggingFace/Diffusers/VAE implementation
        logvar = torch.clamp(logvar, min=-30, max=20)
        return mu, logvar


class FASD_Predictor(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_embedding: int,
        n_units_out: int,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 100,
        nonlin: str = "relu",
        nonlin_out: Optional[List[Tuple[str, int]]] = None,
        random_state: int = 0,
        dropout: float = 0.1,
        batch_norm: bool = True,
        residual: bool = False,
        device: Any = DEVICE,
    ) -> None:
        super(FASD_Predictor, self).__init__()
        self.device = device
        self.model = MLP(
            task_type="regression",
            n_units_in=n_units_embedding,
            n_units_out=n_units_out,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            nonlin=nonlin,
            nonlin_out=nonlin_out,
            random_state=random_state,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
            device=device,
        ).to(self.device)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        X: Tensor,
    ) -> Tensor:
        return self.model(X)


class FASD(nn.Module):
    """
    .. inheritance-diagram:: synthcity.plugins.core.models.vae.VAE
        :parts: 1


    Basic VAE implementation.

    Args:
        n_features: int
            Number of features in the dataset
        n_units_embedding: int
            Number of units in the latent space
        batch_size: int
            Training batch size
        n_iter: int
            Number of training iterations
        random_state: int
            Random random_state
        lr: float
            Learning rate
        weight_decay: float:
            Optimizer weight decay
        decoder_n_layers_hidden: int
            Number of hidden layers in the decoder
        decoder_n_units_hidden: int
            Number of units in the hidden layer in the decoder
        decoder_nonlin_out: List
            List of activations layout, as generated by the tabular encoder
        decoder_batch_norm: bool
            Use batchnorm in the decoder
        decoder_dropout: float
            Use dropout in the decoder
        decoder_residual: bool
            Use residual connections in the decoder
        encoder_n_layers_hidden: int
            Number of hidden layers in the encoder
        encoder_n_units_hidden: int
            Number of units in the hidden layer in the encoder
        encoder_batch_norm: bool
            Use batchnorm in the encoder
        encoder_dropout: float
            Use dropout in the encoder
        encoder_residual: bool
            Use residual connections in the encoder
        loss_strategy: str
            - standard: classic VAE loss
            - robust_divergence: Algorithm 1 in "Robust Variational Autoencoder for Tabular Data with β Divergence"
        loss_factor: int
            Parameter for the standard loss
        robust_divergence_beta: int
            Parameter for the robust_divergence loss
        dataloader_sampler:
            Custom sampler used by the dataloader, useful for conditional sampling.
        device:
            CPU/CUDA
        extra_loss_cbks:
            Custom loss callbacks. For example, for conditional loss.
        clipping_value:
            Gradients clipping value. Zero disables the feature
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
        n_features: int,
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
        predictor_nonlin_out: Optional[List[Tuple[str, int]]] = None,
        decoder_nonlin_out: Optional[List[Tuple[str, int]]] = None,
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
        dataloader_sampler: Optional[sampler.Sampler] = None,
        device: Any = DEVICE,
        extra_loss_cbks: List[Callable] = [],
        clipping_value: int = 1,
        # early stopping
        n_iter_min: int = 100,
        n_iter_print: int = 10,
        patience: int = 20,
    ) -> None:
        super(FASD, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.loss_factor = loss_factor
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_units_embedding = n_units_embedding
        self.dataloader_sampler = dataloader_sampler
        self.extra_loss_cbks = extra_loss_cbks
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        self.clipping_value = clipping_value
        self.n_iter_print = n_iter_print
        self.n_iter_min = n_iter_min
        self.patience = patience
        self.task_type = task_type
        self.encoder_n_layers_hidden = encoder_n_layers_hidden
        self.encoder_n_units_hidden = encoder_n_units_hidden
        self.n_features = n_features
        self.encoder_nonlin = encoder_nonlin
        self.encoder_batch_norm = encoder_batch_norm
        self.encoder_dropout = encoder_dropout
        self.decoder_nonlin_out = decoder_nonlin_out

        self.encoder = FASD_Encoder(
            n_features,
            n_units_embedding,
            n_layers_hidden=encoder_n_layers_hidden,
            n_units_hidden=encoder_n_units_hidden,
            nonlin=encoder_nonlin,
            batch_norm=encoder_batch_norm,
            dropout=encoder_dropout,
            device=device,
        )

        for _, nonlin_len in predictor_nonlin_out:
            n_units_out = nonlin_len

        self.predictor = FASD_Predictor(
            n_units_embedding,
            n_units_out,
            n_layers_hidden=predictor_n_layers_hidden,
            n_units_hidden=predictor_n_units_hidden,
            nonlin=predictor_nonlin,
            nonlin_out=predictor_nonlin_out,
            batch_norm=predictor_batch_norm,
            dropout=predictor_dropout,
            residual=predictor_residual,
            device=device,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> Any:
        # train encoder
        self.target_col = y.columns
        Xt = self._check_tensor(X)
        yt = self._check_tensor(y)
        self._train(Xt, yt)

        # train decoder
        self.decoder = FASD_Decoder(
            n_units_in=self.n_units_embedding,
            n_units_hidden=self.encoder_n_units_hidden,
            n_units_out=self.n_features,
            n_layers_hidden=self.encoder_n_layers_hidden,
            nonlin=self.encoder_nonlin,
            nonlin_out=self.decoder_nonlin_out,
            device=self.device,
            batch_size=self.batch_size,
            n_iter=self.n_iter,
            patience=self.patience,
            lr=self.lr,
            weight_decay=self.weight_decay,
            n_iter_min=self.n_iter_min,
            dropout=self.encoder_dropout,
            clipping_value=self.clipping_value,
        )
        X_rep = self.encode(X)
        self.decoder.fit(X=X_rep, y=X)

        return self

    def encode(self, X: pd.DataFrame):
        Xt = self._check_tensor(X)
        mu, logvar = self.encoder(Xt)
        noise = self._reparameterize(mu, logvar)
        noise = pd.DataFrame(
            noise.cpu().detach().numpy(),
            columns=["col_" + str(x) for x in list(range(noise.shape[1]))],
        )
        return noise

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def generate(self, count: int):
        self.predictor.eval()
        self.decoder.eval()

        steps = count // self.batch_size + 1
        labels = []
        noises = []
        for idx in range(steps):
            mean = torch.zeros(self.batch_size, self.n_units_embedding)
            std = torch.ones(self.batch_size, self.n_units_embedding)
            noise = torch.normal(mean=mean, std=std).to(self.device)

            fake_labels = self.predictor(noise)
            labels.append(fake_labels.detach().cpu().numpy())
            noises.append(noise.detach().cpu().numpy())

        # get dataframe of (soft) labels
        labels = np.concatenate(labels, axis=0)
        labels = labels[:count]
        labels = pd.DataFrame(
            labels, columns=["target_" + str(x) for x in list(range(labels.shape[1]))]
        )
        # get dataframe of synthetic data from (noise) representations
        noises = np.concatenate(noises, axis=0)
        noises = noises[:count]
        noises = pd.DataFrame(
            noises, columns=["col_" + str(x) for x in list(range(noise.shape[1]))]
        )
        fake = self.decoder.decode(noises)
        return fake, labels

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _train(
        self,
        X: Tensor,
        y: Tensor,
    ) -> Any:

        stratify = None
        if self.task_type == "classification":
            stratify = y.detach().cpu().numpy()

        X, X_val, y, y_val = train_test_split(
            X.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            stratify=stratify,
            train_size=0.8,
            random_state=self.random_state,
        )
        Xt = self._check_tensor(X)
        Xt_val = self._check_tensor(X_val)
        yt = self._check_tensor(y)
        yt_val = self._check_tensor(y_val)

        loader = DataLoader(
            dataset=TensorDataset(Xt, yt), batch_size=self.batch_size, pin_memory=False
        )
        val_loader = DataLoader(
            dataset=TensorDataset(Xt_val, yt_val),
            batch_size=self.batch_size,
            pin_memory=False,
        )

        optimizer = Adam(
            self.parameters(),
            weight_decay=self.weight_decay,
            lr=self.lr,
        )

        best_loss = np.inf
        best_state_dict = None
        patience = 0
        for epoch in tqdm(range(self.n_iter)):
            self.train()
            for X, y in loader:

                mu, logvar = self.encoder(X)
                embedding = self._reparameterize(mu, logvar)

                pred = self.predictor(embedding)
                loss = self._loss_function(
                    pred,
                    y.float(),
                    mu,
                    logvar,
                )
                optimizer.zero_grad()
                if self.clipping_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.clipping_value
                    )
                loss.backward()
                optimizer.step()

            if epoch % self.n_iter_print == 0:
                self.eval()
                for X_val, y_val in val_loader:
                    mu, logvar = self.encoder(X_val)
                    embedding = self._reparameterize(mu, logvar)
                    pred = self.predictor(embedding)

                    val_loss = (
                        self._loss_function(
                            pred,
                            y_val,
                            mu,
                            logvar,
                        )
                        .detach()
                        .item()
                    )

                log.debug(f"[{epoch}/{self.n_iter}] Loss: {val_loss}")
                if val_loss >= best_loss:
                    patience += 1
                else:
                    best_loss = val_loss
                    best_state_dict = self.state_dict()
                    patience = 0

                if patience >= self.patience and epoch >= self.n_iter_min:
                    log.debug(f"[{epoch}/{self.n_iter}] Early stopping")
                    break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        return self

    def _check_tensor(self, X: Tensor) -> Tensor:
        if isinstance(X, Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _loss_function(
        self,
        pred: Tensor,
        y: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tensor:

        for activation, _ in self.decoder_nonlin_out:
            # reconstructed is after the activation
            if activation == "softmax":
                criterion = nn.CrossEntropyLoss()

            else:
                criterion = nn.MSELoss()
        pred_loss = criterion(pred, y)
        KLD_loss = (-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())) / y.shape[0]

        if torch.isnan(pred_loss):
            raise RuntimeError("NaNs detected in the reconstruction_loss")
        if torch.isnan(KLD_loss):
            raise RuntimeError("NaNs detected in the KLD_loss")

        return pred_loss * self.loss_factor + KLD_loss


class FASD_Decoder(nn.Module):

    def __init__(
        self,
        n_units_in: int,
        n_units_hidden: int,
        n_units_out: int,
        n_layers_hidden: int,
        nonlin: str,
        nonlin_out: List[Tuple[str, int]],
        device: Any = DEVICE,
        random_state: int = 0,
        batch_size: int = 200,
        n_iter: int = 300,
        patience: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        n_iter_min: int = 30,
        dropout: float = 0.1,
        clipping_value: int = 1,
    ) -> None:
        super(FASD_Decoder, self).__init__()
        self.random_state = random_state
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.device = device
        torch.manual_seed(self.random_state)
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_iter_min = n_iter_min
        self.nonlin_out = nonlin_out

        self.decoder = MLP(
            n_units_in=n_units_in,
            n_units_out=n_units_out,
            task_type="regression",
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            nonlin=nonlin,
            nonlin_out=nonlin_out,
            random_state=self.random_state,
            dropout=dropout,
            clipping_value=clipping_value,
            device=self.device,
        ).to(device=self.device)

    def forward(self, X: Tensor):
        """Forward pass through the Decoder"""
        Xt = self._check_tensor(X)
        Xt = self.decoder(Xt)
        return Xt

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train the network to reconstruct input features from representations"""

        self.target_cols = y.columns

        # validation split (no stratification for original input data as target)
        X, X_val, y, y_val = train_test_split(
            X, y, train_size=0.8, random_state=self.random_state
        )

        # create tensor datasets and dataloaders
        Xt = self._check_tensor(X)
        Xt_val = self._check_tensor(X_val)
        yt = self._check_tensor(y)
        yt_val = self._check_tensor(y_val)

        loader = DataLoader(
            dataset=TensorDataset(Xt, yt), batch_size=self.batch_size, pin_memory=False
        )
        val_loader = DataLoader(
            dataset=TensorDataset(Xt_val, yt_val),
            batch_size=self.batch_size,
            pin_memory=False,
        )

        # perform training loop
        self._train(loader=loader, val_loader=val_loader)
        return self

    def _train(self, loader: DataLoader, val_loader: DataLoader):
        """Perform the training loop"""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_state_dict = None
        best_loss = float("inf")
        patience = 0
        for epoch in tqdm(range(self.n_iter)):
            self.train()
            train_loss = 0
            for inputs, targets in loader:
                outputs = self.forward(inputs)
                loss = self._loss_function_standard(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_loss = self.validate(val_loader)

            if val_loss >= best_loss:
                patience += 1
            else:
                best_loss = val_loss
                best_state_dict = self.state_dict()
                patience = 0

            if patience >= self.patience and epoch >= self.n_iter_min:
                break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        return self

    def validate(self, val_loader):
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.forward(inputs)
                loss = self._loss_function_standard(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def decode(self, X: pd.DataFrame):
        """Pass representations through the Decoder to get reconstructed input features"""

        # turn into tensor
        Xt = self._check_tensor(X)

        # forward pass through decoder
        Xt = self.decoder(Xt)

        # turn into dataframe
        X_enc = pd.DataFrame(
            Xt.cpu().detach().numpy(),
            columns=self.target_cols,
        )

        return X_enc

    def _loss_function_standard(
        self,
        reconstructed: Tensor,
        real: Tensor,
    ) -> Tensor:
        step = 0
        loss = []
        for activation, length in self.nonlin_out:
            step_end = step + length
            # reconstructed is after the activation
            if activation == "softmax":
                discr_loss = nn.NLLLoss(reduction="sum")(
                    torch.log(reconstructed[:, step:step_end] + 1e-8),
                    torch.argmax(real[:, step:step_end], dim=-1),
                )
                loss.append(discr_loss)
            else:
                diff = reconstructed[:, step:step_end] - real[:, step:step_end]
                cont_loss = (50 * diff**2).sum()

                loss.append(cont_loss)
            step = step_end

        reconstruction_loss = torch.sum(torch.stack(loss)) / real.shape[0]

        if torch.isnan(reconstruction_loss):
            raise RuntimeError("NaNs detected in the reconstruction_loss")

        return reconstruction_loss

    def _check_tensor(self, X: Tensor) -> Tensor:
        if isinstance(X, Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)
