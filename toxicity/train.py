# toxicity/train.py
# Training Operations.

from argparse import Namespace
from typing import Dict, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve

from toxicity.config import logger


# Trainer
class Trainer:
    def __init__(
        self,
        model,
        device=torch.device("cpu"),
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        trial=None,
    ):

        # Set params
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trial = trial

    def train_step(self, dataloader):
        """Train step.

        Args:
            dataloader (torch.utils.data.DataLoader): Torch dataloader to load batches from.

        """

        # Set model to train mode
        self.model.train()
        loss = 0.0

        ## Iterate over dataloader batches
        for i, batch in enumerate(dataloader):

            ## Step
            # show_gpu(f'{i}: GPU memory usage after loading training objects:')
            batch = [item.to(self.device) for item in batch]  # set batch to device
            inputs, targets = batch[:-1], batch[-1]
            self.optimizer.zero_grad()  # Reset gradients
            z = self.model(inputs)  # Forward Pass
            J = self.loss_fn(z, targets.type_as(z))  # Define loss
            J.backward()  # Backward Pass
            self.optimizer.step()  # update weights
            # show_gpu(f'{i}: GPU memory usage after training model:')

            # Calculating cumulative loss at intermediate steps
            loss += (J.detach().item() - loss) / (i + 1)

        return loss

    def eval_step(self, dataloader):
        """Evaluation (val / test) step.

        Args:
            dataloader (torch.utils.data.DataLoader): Torch dataloader to load batches from.

        """

        # Set model to eval mode
        self.model.eval()
        loss = 0.0
        y_trues, y_probs = [], []

        ##Iterate over val batches
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                ## Step
                batch = [item.to(self.device) for item in batch]  # set batch to device
                inputs, y_true = batch[:-1], batch[-1]
                z = self.model(inputs)  # Forward Pass
                J = self.loss_fn(z, y_true.type_as(z)).item()  # Get loss

                # Calculating cumulative loss at intermediate steps
                loss += (J - loss) / (i + 1)

                ##Store outputs
                y_prob = torch.sigmoid(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(y_true.cpu().numpy())

        return loss, np.vstack(y_trues), np.vstack(y_probs)

    def predict_step(self, dataloader: torch.utils.data.DataLoader):
        """Prediction (inference) step.

        Note:
            Loss is not calculated for this loop.

        Args:
            dataloader (torch.utils.data.DataLoader): Torch dataloader to load batches from.

        """

        # Set model to eval mode
        self.model.eval()
        y_trues, y_probs = [], []

        ##Iterate over val batches
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                # Step with inputs
                batch = [item.to(self.device) for item in batch]
                inputs, targets = batch[:-1], batch[-1]
                z = self.model(inputs)

                # Store outputs
                y_prob = torch.sigmoid(z).cpu().numpy()
                y_probs.extend(y_prob)
                y_trues.extend(targets.cpu().numpy())

        return np.vstack(y_trues), np.vstack(y_probs)

    def train(
        self,
        num_epochs: int,
        patience: int,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
    ) -> Tuple:
        """Training loop.

        Args:
            num_epochs (int): Maximum number of epochs to train for (can stop earlier based on performance).
            patience (int): Number of acceptable epochs for continuous degrading performance.
            train_dataloader (torch.utils.data.DataLoader): Dataloader object with training data split.
            val_dataloader (torch.utils.data.DataLoader): Dataloader object with validation data split.

        Raises:
            optuna.TrialPruned: Early stopping of the optimization trial if poor performance.

        Returns:
            The best validation loss and the trained model from that point.

        """

        best_val_loss = np.inf
        best_model = None
        _patience = patience
        for epoch in range(num_epochs):

            # Steps
            train_loss = self.train_step(dataloader=train_dataloader)
            val_loss, _, _ = self.eval_step(dataloader=val_dataloader)
            self.scheduler.step(val_loss)

            # Pruning based on the intermediate value
            if self.trial:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    logger.info("Unpromising trial pruned!")
                    raise optuna.TrialPruned()

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience  # reset patience
            else:
                _patience -= 1

            if not _patience:
                logger.info("Stopping Early")
                break

            # # Tracking
            # mlflow.log_metrics(
            #     {"train_loss":train_loss, "val_loss":val_loss}, step=epoch
            # )

            # Logging
            logger.info(
                f"Epoch : {epoch+1} |"
                f" Train Loss : {train_loss : .7f},"
                f" Val Loss : {val_loss : .7f},"
                f" lr : {self.optimizer.param_groups[0]['lr'] : .2E},"
                f" _patience : {_patience}"
            )

        return best_model, best_val_loss


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    """Determine the best threshold for maximum f1 score.

    Usage:

    ```python
    # Find best threshold
    _, y_true, y_prob = trainer.eval_step(dataloader=train_dataloader)
    params.threshold = find_best_threshold(y_true=y_true, y_prob=y_prob)
    ```

    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Probability distribution for predicted labels.

    Returns:
        Best threshold for maximum f1 score.

    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = (2 * precisions * recalls) / (precisions + recalls)
    return thresholds[np.argmax(f1s)]


## Modifying to include information about trials
def train(
    params: Namespace,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    device: torch.device,
    class_weights: Dict,
    trial: optuna.trial._trial.Trial = None,
) -> Tuple:
    """Train a model.

    Args:
        params (Namespace): Parameters for data processing and training.
        train_dataloader (torch.utils.data.DataLoader): train data loader.
        val_dataloader (torch.utils.data.DataLoader): val data loader.
        model (nn.Module): Initialize model to train.
        device (torch.device): Device to run model on.
        class_weights (Dict): Dictionary of class weights.
        trial (optuna.trial._trial.Trail, optional): Optuna optimization trial. Defaults to None.

    Returns:
        The best trained model, loss and performance metrics.

    """
    # Define loss
    class_weight_tensor = torch.Tensor(np.array(list(class_weights.values())))
    loss_fn = nn.BCEWithLogitsLoss(weight=class_weight_tensor)

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    # Initialize trainer module
    trainer = Trainer(
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        trial=trial,
    )

    # Train
    best_model, best_val_loss = trainer.train(
        params.num_epochs, params.patience, train_dataloader, val_dataloader
    )

    ## Best threshold for f1
    _, y_true, y_prob = trainer.eval_step(dataloader=train_dataloader)
    params.threshold = find_best_threshold(y_true=y_true, y_prob=y_prob)

    return params, best_model, best_val_loss