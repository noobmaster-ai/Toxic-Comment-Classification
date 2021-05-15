# toxicity/predict.py
# Prediction Operations.

from distutils.util import strtobool
from typing import Dict, List

import numpy as np
import torch

from toxicity import data, train


def predict(texts: List, artifacts: Dict, device: torch.device = torch.device("cpu")) -> Dict:
    """Predict toxicity for an input text using the
    best model from the 'best' experiment.

    Usage:

    ```Python
    texts = "::::Well, it sucks to have a university to be nicknameless. And it's the first time in NCAA history that it has happened. /"
    artifacts = load_artifacts(run_id="")
    predict(texts=texts, artifacts=artifacts)
    ```
    <pre>
    [
        {
            "input_text": "::::Well, it sucks to have a university to be nicknameless. And it's the first time in NCAA history that it has happened. /",
            "processed_text": "well sucks university nicknameless first time ncaa history happened"
            "predicted_toxicity": [
                'insult'
            ]
        }
    ]
    </pre>

    Note:
        The input parameter 'texts' can hold multiple input texts and so the resulting prediction dictionary will have 'len(texts)' items.

    Args:
        texts (List): List of input texts to predict toxicity type for.
        artifacts (Dict): Artifacts needed for inference.
        device (torch.device, optional): Device to run model on. Defaults to torch.device("cpu").

    Returns:
        Dict: Predicted toxicity types for each of the input texts.
    """

    # Retrieve artifacts
    params = artifacts["params"]
    label_encoder = data.label_encoder()
    tokenizer = artifacts["tokenizer"]
    model = artifacts["model"]

    # Prepare data
    preprocessed_texts = [
        data.preprocessing(
            text,
            lower=bool(strtobool(str(params.lower))),  # params.lower could be str/bool
            stem=bool(strtobool(str(params.stem))),
        )
        for text in texts
    ]
    X = np.array(tokenizer.texts_to_sequences(preprocessed_texts), dtype="object")
    y_filler = np.zeros((len(X), len(label_encoder)))
    dataset = RNNTextDataset(X=X, y=y_filler)
    dataloader = dataset.create_dataloader(batch_size=int(params.batch_size))

    # Get predictions
    trainer = Trainer(model=model, device=device)
    y_prob = trainer.predict_step(dataloader)
    y_pred = np.array([np.where(prob >= threshold, 1, 0) for prob in y_prob])
    toxic_level = label_encoder.decode(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "preprocessed_text": preprocessed_texts[i],
            "predicted_toxicity": toxic_level[i],
        }
        for i in range(len(tags))
    ]

    return predictions