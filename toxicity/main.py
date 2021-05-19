# toxicity/main.py
# Training, Optimization, etc.

import itertools
import json
from argparse import Namespace
from pathlib import Path
from typing import Dict
from nltk.corpus.reader import util
from torch._C import dtype

import mlflow
import numpy as np
import pandas as pd
import optuna
import torch
from numpyencoder import NumpyEncoder

from toxicity import config, data, eval, models, train, utils
from toxicity.config import logger


def run(params: Namespace, trial: optuna.trial._trial.Trial = None) -> Dict:
    """Operations for training

    Args:
        params (Namespace): Input parameters for operations.
        trial (optuna.trial._trial.Trial, optional): Optuna optimization trial. Defaults to None.

    Returns:
        Dict: Artifacts to save and load for later.

    """
    # 1. Set seed
    utils.set_seed(seed=params.seed)

    # 2. Set device
    device = utils.set_device(cuda=params.cuda)

    # 3. Load data
    df = pd.read_csv(Path(config.DATA_DIR, r"train/train.csv"))
    if params.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    # df = df[: params.subset]

    # 4. Preprocess data
    df.comment_text = df.comment_text.apply(
        data.preprocessing, lower=params.lower, stem=params.stem
    )

    # 5. Encode labels
    labels = df.iloc[:, 3:]
    label_encoder = data.MultiLabelLabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.encode(labels)

    # Class weights
    all_labels = list(labels.columns)
    counts = np.bincount([label_encoder.class_to_index[class_] for class_ in all_labels])
    class_weights = {i: 1.0 / count for i, count in enumerate(counts)}

    # 7. Split data
    utils.set_seed(seed=params.seed)  # needed for skmultilearn
    X = df.comment_text.to_numpy()
    X_train, X_, y_train, y_ = data.iterative_train_test_split(
        X=X, y=y, train_size=params.train_size
    )
    X_val, X_test, y_val, y_test = data.iterative_train_test_split(X=X_, y=y_, train_size=0.5)
    test_df = pd.DataFrame(
        {
            "text": X_test,
            "toxic": y_test[:, 0],
            "severe_toxic": y_test[:, 1],
            "obscene": y_test[:, 2],
            "threat": y_test[:, 3],
            "insult": y_test[:, 4],
            "identity_hate": y_test[:, 5],
        }
    )

    # 8. Tokenize inputs
    tokenizer = data.Tokenizer(char_level=params.char_level)
    tokenizer.fit_on_texts(texts=X_train)
    X_train = np.array(tokenizer.texts_to_sequences(X_train), dtype=object)
    X_val = np.array(tokenizer.texts_to_sequences(X_val), dtype=object)
    X_test = np.array(tokenizer.texts_to_sequences(X_test), dtype=object)

    # 9. Create Dataloaders
    train_dataset = data.RNNTextDataset(X=X_train, y=y_train, max_seq_len=params.max_seq_len)
    # test_dataset = data.RNNTextDataset(X=X_test, y=y_test, max_seq_len=params.max_seq_len)
    val_dataset = data.RNNTextDataset(X=X_val, y=y_val, max_seq_len=params.max_seq_len)

    train_dataloader = train_dataset.create_dataloader(batch_size=params.batch_size)
    # test_dataloader = test_dataset.create_dataloader(batch_size=params.batch_size)
    val_dataloader = val_dataset.create_dataloader(batch_size=params.batch_size)

    # 10. Initialize model
    model = models.initialize_model(
        params=params,
        vocab_size=len(tokenizer),
        num_classes=len(label_encoder),
        device=device,
    )

    # 11. Train model
    logger.info(f"Parameters: {json.dumps(params.__dict__, indent=2, cls=NumpyEncoder)}")
    params, model, loss = train.train(
        params=params,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        device=device,
        class_weights=class_weights,
        trial=trial,
    )

    # 12. Evaluate model
    artifacts = {
        "params": params,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
        "model": model,
        "loss": loss,
    }
    device = torch.device("cpu")
    y_true, y_pred, performance = eval.evaluate(df=test_df, artifacts=artifacts)
    artifacts["performance"] = performance

    return artifacts


def objective(params: Namespace, trial: optuna.trial._trial.Trial) -> float:
    """Objective function for optimization trials.

    Args:
        params (Namespace): Input parameters for each trial (see `config/params.json`).
        trial (optuna.trial._trial.Trial): Optuna optimization trial.

    Returns:
        F1 score from evaluating the trained model on the test data split.

    """
    # Parameters (to tune)
    params.embedding_dim = trial.suggest_int("embedding_dim", 64, 128)
    params.num_filters = trial.suggest_int("num_filters", 128, 256)
    params.hidden_dim = trial.suggest_int("hidden_dim", 128, 512)
    params.rnn_hidden_dim = trial.suggest_int("rnn_hidden_dim", 128, 512)
    params.dropout_p = trial.suggest_uniform("dropout_p", 0.3, 0.8)
    params.lr = trial.suggest_loguniform("lr", 5e-5, 5e-4)

    # Train (can move some of these outside for efficiency)
    logger.info(f"\nTrial {trial.number}:")
    logger.info(json.dumps(trial.params, indent=2))
    artifacts = run(params=params, trial=trial)

    # Set additional attributes
    params = artifacts["params"]
    performance = artifacts["performance"]
    logger.info(json.dumps(performance["overall"], indent=2))
    trial.set_user_attr("threshold", params.threshold)
    trial.set_user_attr("precision", performance["overall"]["precision"])
    trial.set_user_attr("recall", performance["overall"]["recall"])
    trial.set_user_attr("f1", performance["overall"]["f1"])

    return performance["overall"]["f1"]


def load_artifacts(run_id: str, device: torch.device = torch.device("cpu")) -> Dict:
    """Load artifacts for current model.

    Args:
        run_id (str): ID of the model run to load artifacts. Defaults to run ID in config.MODEL_DIR.
        device( (torch.device): Device to run model on. Defaults to CPU.

    Returns:
        Artifacts needed for inference.

    """
    # Load artifacts
    artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.splits("file://")[-1]
    params = Namespace(**utils.load_dict(filepath=Path(artifact_uri, "params.json")))
    label_encoder = data.LabelEncoder.load(fp=Path(artifact_uri, "label_encoder.json"))
    tokenizer = data.Tokenizer.load(fp=Path(artifact_uri, "tokenizer.json"))
    model_state = torch.load(Path(artifact_uri, "model.pt"), map_location=device)
    performance = utils.load_dict(filepath=Path(artifact_uri, "performance.json"))

    # Initialize model
    model = models.initialize_model(
        params=params, vocab_size=len(tokenizer), num_classes=len(label_encoder)
    )
    model.load_state_dict(model_state)

    return {
        "params": params,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
        "model": model,
        "performance": performance,
    }