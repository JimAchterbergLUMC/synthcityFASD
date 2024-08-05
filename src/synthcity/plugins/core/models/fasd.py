# this file contains the code for the FASD model.
# this should be placed together with other plugins in the SynthCity package.
# other plugins import the model from this file to incorporate the FASD option.

# Note: currently only supports binary classification
import os
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split


class FASD_Encoder(nn.Module):
    """
    Single hidden layer Encoder.
    TBD:
    - add potential for more hidden layers.
    - add self-specifiable activation functions
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_activation) -> None:
        super(FASD_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False), latent_activation
        )

    def forward(self, x: Tensor):
        return self.encoder(x)

    def encode(self, X: pd.DataFrame):
        """
        Encode an input dataframe as representations using the trained encoder.
        """
        X = X.copy()
        x = torch.tensor(X.values, dtype=torch.float32)
        x = self.forward(x)

        X = pd.DataFrame(
            x.detach().numpy(),
            columns=["repr_" + str(x) for x in list(range(x.size(1)))],
        )

        return X


class FASD_Predictor(nn.Module):
    """
    Predictor layer, classifying output from latent space.
    TBD:
    - extend beyond only classification.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super(FASD_Predictor, self).__init__()
        self.num_classes = num_classes
        self.predictor = nn.Sequential(nn.Linear(input_dim, num_classes), nn.Softmax())

    def forward(self, x: Tensor):
        return self.predictor(x)

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        x = torch.tensor(X.values, dtype=torch.float32)
        x = self.forward(x)

        x = one_hot_from_probs(x)

        X = pd.DataFrame(
            x.detach().numpy(),
            columns=["target_" + str(x) for x in list(range(self.num_classes))],
        )

        return X


class FASD_NN(nn.Module):
    """
    Full FASD network, passing data through Encoder and Predictor.
    Encoder and Predictor are retrievable through this model object.
    TBD:
    - Add progress bar, including training info
    - generalize beyond only classification
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        random_state: int,
        checkpoint_dir: str,
        val_split: float,
        latent_activation,
    ) -> None:
        super(FASD_NN, self).__init__()
        self.encoder = FASD_Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_activation=latent_activation,
        )
        self.predictor = FASD_Predictor(input_dim=hidden_dim, num_classes=num_classes)
        self.random_state = random_state
        self.checkpoint_dir = checkpoint_dir
        self.val_split = val_split

    def forward(self, x: Tensor):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        criterion,
        optimizer,
        num_epochs=10,
        batch_size=64,
        early_stop_patience=20,
        early_stop_delta=1e-4,
    ):
        # path to fasd model for checkpointing
        model_file = "best_fasd_model.pth"
        model_path = os.path.join(self.checkpoint_dir, model_file)

        # clear existing best trained fasd model if it already exists
        if os.path.exists(model_path):
            os.remove(model_path)

        # split into stratified validation set for monitoring and best model selection
        X, y = X.copy(), y.copy()

        X, X_val, y, y_val = train_test_split(
            X,
            y,
            test_size=self.val_split,
            stratify=y,
            random_state=self.random_state,
            shuffle=True,
        )

        # create dataloaders for training
        dataloader = DataLoader(
            TensorDataset(
                torch.tensor(X.values, dtype=torch.float32),
                torch.tensor(y.values, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val.values, dtype=torch.float32),
                torch.tensor(y_val.values, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        best_acc = -999
        early_stop_counter = 0
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            for inputs, targets in dataloader:
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_total += targets.size(0)
                _, preds = torch.max(outputs.data, 1)
                _, tar = torch.max(targets, 1)
                train_correct += (preds == tar).sum().item()
            train_acc = 100 * train_correct / train_total
            val_loss, val_acc = self.validate(val_loader, criterion)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(dataloader):.4f}, Acc: {train_acc:.2f}, Val_Loss: {val_loss:.4f}, Val_Acc: {val_acc:.2f}"
            )

            # save best model if best model at current epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.state_dict(), model_path)

            # early stopping if validation metric not improving for many epochs
            if (val_acc != best_acc) & (val_acc < (best_acc + early_stop_delta)):
                early_stop_counter += 1
            else:
                early_stop_counter = 0
            if early_stop_counter == early_stop_patience:
                break

        # load best performing model on validation set
        self.load_state_dict(torch.load(model_path))

    def validate(self, val_loader, criterion):
        self.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Calculate accuracy
                val_total += targets.size(0)
                _, preds = torch.max(outputs.data, 1)
                _, tar = torch.max(targets, 1)
                val_correct += (preds == tar).sum().item()

        val_acc = 100 * val_correct / val_total
        return val_loss / len(val_loader), val_acc


# class FASD_Decoder(nn.Module):
#     """
#     Single hidden layer Decoder.
#     Decodes tabular encoded representations to tabular encoded original data space.

#     TBD:
#     - separate cont and cat output for easier loss calculation
#     - implement separate loss calculation
#     - add a 'restructure output' method to restructure output according to the original structure

#     TBD:
#     - add potential for more hidden layers.
#     - add self-specifiable activation functions
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         cont_idx: list,
#         cat_idx: list,
#         checkpoint_dir: str,
#         val_split: float,
#         random_state: int,
#     ) -> None:
#         """
#         cont_idx: list of indices where output is continuous
#         cat_idx: list of lists where each list contains the indices where output is categorical
#         """
#         super(FASD_Decoder, self).__init__()

#         self.cont_idx = cont_idx
#         self.cat_idx = cat_idx
#         self.checkpoint_dir = checkpoint_dir
#         self.val_split = val_split
#         self.random_state = random_state

#         self.dec_cont = nn.Linear(input_dim, len(self.cont_idx))
#         self.dec_cats = nn.ModuleList(
#             [nn.Linear(input_dim, len(idx_list)) for idx_list in self.cat_idx]
#         )

#     def forward(self, x: Tensor):
#         cont_out = torch.sigmoid(self.dec_cont(x))

#         cat_out = []
#         for dec_cat in self.dec_cats:
#             cat_out.append(torch.nn.functional.softmax(dec_cat(x), dim=1))

#         return cont_out, cat_out

#     def train_model(
#         self,
#         X: pd.DataFrame,
#         y: pd.DataFrame,
#         criterion_cont,
#         criterion_cat,
#         optimizer,
#         num_epochs=10,
#         batch_size=64,
#         early_stop_patience=20,
#         early_stop_delta=1e-4,
#     ):

#         self.output_cols = y.columns
#         # path to fasd model for checkpointing
#         model_file = "best_fasd_decoder.pth"
#         model_path = os.path.join(self.checkpoint_dir, model_file)

#         # clear existing best trained fasd model if it already exists
#         if os.path.exists(model_path):
#             os.remove(model_path)

#         # split into validation set for monitoring and best model selection
#         X, y = X.copy(), y.copy()
#         X, X_val, y, y_val = train_test_split(
#             X, y, test_size=self.val_split, random_state=self.random_state
#         )

#         # create dataloaders for training
#         dataloader = DataLoader(
#             TensorDataset(
#                 torch.tensor(X.values, dtype=torch.float32),
#                 torch.tensor(y.values, dtype=torch.float32),
#             ),
#             batch_size=batch_size,
#             shuffle=True,
#         )

#         val_loader = DataLoader(
#             TensorDataset(
#                 torch.tensor(X_val.values, dtype=torch.float32),
#                 torch.tensor(y_val.values, dtype=torch.float32),
#             ),
#             batch_size=batch_size,
#             shuffle=True,
#         )

#         # training loop
#         best_loss = float("inf")
#         val_loss = float('inf')
#         early_stop_counter = 0
#         for epoch in range(num_epochs):
#             self.train()
#             for inputs, targets in dataloader:
#                 targets_cont = targets[:, self.cont_idx]

#                 targets_cat = [targets[:, idx] for idx in self.cat_idx]

#                 outputs_cont, outputs_cat = self.forward(inputs)

#                 loss_cont = criterion_cont(outputs_cont, targets_cont)

#                 loss_cat = sum(
#                     criterion_cat(output, target)
#                     for output, target in zip(outputs_cat, targets_cat)
#                 )/len(outputs_cat)
#                 loss = loss_cont + loss_cat

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             val_loss = self.validate(val_loader, criterion_cont, criterion_cat)

#             print(
#                 f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val_Loss: {val_loss:.4f}"
#             )

#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 torch.save(self.state_dict(), model_path)

#            # early stopping if validation loss not improving for many epochs
#             if (val_loss != best_loss) & (val_loss > (best_loss - early_stop_delta)):
#                 early_stop_counter += 1
#             else:
#                 early_stop_counter = 0
#             if early_stop_counter == early_stop_patience:
#                 break

#         # load best performing model on validation set
#         self.load_state_dict(torch.load(model_path))

#     def validate(self, val_loader, criterion_cont, criterion_cat):
#         self.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for inputs, targets in val_loader:
#                 targets_cont = targets[:, self.cont_idx]
#                 targets_cat = [targets[:, idx] for idx in self.cat_idx]

#                 outputs_cont, outputs_cat = self.forward(inputs)

#                 loss_cont = criterion_cont(outputs_cont, targets_cont)
#                 loss_cat = sum(
#                     criterion_cat(output, target)
#                     for output, target in zip(outputs_cat, targets_cat)
#                 )/len(outputs_cat)
#                 loss = loss_cont + loss_cat

#                 val_loss += loss.item()
#         return val_loss / len(val_loader)

#     def decode(self, X: pd.DataFrame):
#         """
#         pass input data through the trained decoder and restructure it according to input data.
#         """
#         X = X.copy()
#         self.eval()
#         with torch.no_grad():
#             x = torch.tensor(X.values, dtype=torch.float32)
#             x_cont, x_cat = self.forward(x)

#         x_cat = [one_hot_from_probs(x) for x in x_cat]

#         # restructure output to original input structure
#         output = torch.zeros(
#             x_cont.size(0),
#             x_cont.size(1) + sum([x_cat_.size(1) for x_cat_ in x_cat]),
#             dtype=x_cont.dtype,
#             device=x_cont.device,
#         )
#         output[:, self.cont_idx] = x_cont
#         for idx, val in zip(self.cat_idx, x_cat):
#             output[:, idx] = val

#         # turn tensor into dataframe
#         X = pd.DataFrame(output.detach().numpy(), columns=self.output_cols)
#         return X


# # one hot encode the categoricals according to softmax probabilities
# def one_hot_from_probs(x: torch.Tensor):
#     idx = x.argmax(dim=1, keepdim=True)
#     x = torch.zeros_like(x).scatter_(1, idx, 1)
#     return x


class FASD_Decoder(nn.Module):
    """
    Single hidden layer Decoder.
    Decodes tabular encoded representations to tabular encoded original data space.

    TBD:
    - separate cont and cat output for easier loss calculation
    - implement separate loss calculation
    - add a 'restructure output' method to restructure output according to the original structure

    TBD:
    - add potential for more hidden layers.
    - add self-specifiable activation functions
    """

    def __init__(
        self,
        input_dim: int,
        cont_idx: list,
        cat_idx: list,
        checkpoint_dir: str,
        val_split: float,
        random_state: int,
    ) -> None:
        """
        cont_idx: list of indices where output is continuous
        cat_idx: list of lists where each list contains the indices where output is categorical
        """
        super(FASD_Decoder, self).__init__()

        self.cont_idx = cont_idx
        self.cat_idx = cat_idx
        self.checkpoint_dir = checkpoint_dir
        self.val_split = val_split
        self.random_state = random_state

        output_dim = len([x for xs in cat_idx for x in xs]) + len(cont_idx)

        self.out = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())

    def forward(self, x: Tensor):
        return self.out(x)

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        criterion,
        optimizer,
        num_epochs=10,
        batch_size=64,
        early_stop_patience=20,
        early_stop_delta=1e-4,
    ):

        self.output_cols = y.columns
        # path to fasd model for checkpointing
        model_file = "best_fasd_decoder.pth"
        model_path = os.path.join(self.checkpoint_dir, model_file)

        # clear existing best trained fasd model if it already exists
        if os.path.exists(model_path):
            os.remove(model_path)

        # split into validation set for monitoring and best model selection
        X, y = X.copy(), y.copy()
        X, X_val, y, y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=self.random_state, shuffle=True
        )

        # create dataloaders for training
        dataloader = DataLoader(
            TensorDataset(
                torch.tensor(X.values, dtype=torch.float32),
                torch.tensor(y.values, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val.values, dtype=torch.float32),
                torch.tensor(y_val.values, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        # training loop
        best_loss = float("inf")
        val_loss = float("inf")
        early_stop_counter = 0
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            for inputs, targets in dataloader:
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            val_loss = self.validate(val_loader, criterion)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(dataloader):.4f}, Val_Loss: {val_loss:.4f}"
            )

            # update best model state if best model at current epoch
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.state_dict(), model_path)

            # early stopping if validation loss not improving for many epochs
            if (val_loss != best_loss) & (val_loss > (best_loss - early_stop_delta)):
                early_stop_counter += 1
            else:
                early_stop_counter = 0
            if early_stop_counter == early_stop_patience:
                break

        # load best performing model on validation set
        self.load_state_dict(torch.load(model_path))

    def validate(self, val_loader, criterion):
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def decode(self, X: pd.DataFrame):
        """
        pass input data through the trained decoder and restructure it according to input data.
        """
        X = X.copy()
        self.eval()
        with torch.no_grad():
            x = torch.tensor(X.values, dtype=torch.float32)
            x = self.forward(x)

        for idx in self.cat_idx:
            x[:, idx] = one_hot_from_probs(x[:, idx])

        # turn tensor into dataframe
        X = pd.DataFrame(x.detach().numpy(), columns=self.output_cols)
        return X


# one hot encode categoricals
def one_hot_from_probs(x: torch.Tensor):
    idx = x.argmax(dim=1, keepdim=True)
    x = torch.zeros_like(x).scatter_(1, idx, 1)
    return x
