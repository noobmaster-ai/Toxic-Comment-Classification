# toxicity/eval.py
# Evaluation components

import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

from toxicity import data, predict, train

# Function to calculate metrics
def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List) -> Dict:
    "Per Class performance metrics"

    # Performance
    performance = {"Overall": {}, "Class": {}}

    # Overall performance
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    performance["Overall"]["Precision"] = metrics[0]
    performance["Overall"]["Recall"] = metrics[1]
    performance["Overall"]["F-Score"] = metrics[2]
    performance["Overall"]["Num_Samples"] = np.float64(len(y_true))

    # Per class performance
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i in range(len(classes)):
        performance["Class"][classes[i]] = {
            "Precision": metrics[0][i],
            "Recall": metrics[1][i],
            "F-Score": metrics[2][i],
            "Num_Samples": np.float64(metrics[3][i]),
        }

    return performance


def evaluate(
    df: pd.DataFrame,
    artifacts: Dict,
    device: torch.device = torch.device("cpu"),
) -> Tuple:
    """Evaluate performance on data.

    Args:
        df (pd.DataFrame): DataFrame(used for slicing)
        artifacts (Dict): Artifacts needed for inference.
        device (torch.device, optional): Device to run model on. Defaults to torch.device("cpu").

    Returns:
        Tuple: Ground truth and predicted labels, performance.
    """
    # Artifacts
    params = artifacts["params"]
    model = artifacts["model"]
    tokenizer = artifacts["tokenizer"]
    model = model.to(device)
    classes = label_encoder.classes

    # Create dataloader
    X = np.array(tokenizer.texts_to_sequences(df.texts.to_numpy()), dtype="object")
    y = data.MultiLabelLabelEncoder.encode(df.iloc[1:])
    dataset = data.RNNTextDataset(X=X, y=y)
    dataloader = dataset.create_dataloader(batch_size=int(params.batch_size))

    # Determine predictions using threshold
    trainer = train.Trainer(model=model, device=device)
    y_true, y_prob = trainer.predict_step(dataloader=dataloader)
    y_pred = np.array([np.where(prob >= float(params.threshold), 1, 0) for prob in y_prob])

    # Evaluate performance
    performance = {}
    performance = get_metrics(y_true=y_true, y_pred=y_pred, classes=classes)

    return y_true, y_pred, performance