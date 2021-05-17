# toxicity/models.py
# Model architectures

import math
from argparse import Namespace
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# import gensim
# from gensim.scripts.glove2word2vec import glove2word2vec
# from gensim.models import KeyedVectors
# from gensim.models import Word2Vec
# from io import BytesIO
# from urllib.request import urlopen
# from zipfile import ZipFile

# Function to get last relevant hidden layer output
def gather_last_relevant_hidden(hiddens: int, seq_lens: int) -> torch.Tensor:
    """Extract and collect the last relevant hidden state
    based on the sequence length

    Args:
        hiddens (int): Output of RNN's GRU block.
        seq_lens (int): Length of the input sequences.

    Returns:
        torch.Tensor: Output tensor.
    """
    seq_lens = seq_lens.long().detach().cpu().numpy() - 1
    out = []
    for batch_index, column_index in enumerate(seq_lens):
        out.append(hiddens[batch_index, column_index])
    return torch.stack(out)


# RNN model architecture
class RNN(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        rnn_hidden_dim: int,
        hidden_dim: int,
        dropout_p: int,
        num_classes: int,
        device: torch.device = torch.device("cpu"),
        pretrained_embeddings=None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0,
    ) -> None:

        """A Recurrent Neural Network architecture created for Natural
        Language processing tasks.

        Args:
            embedding_dim (int): Embedding dimension for tokens.
            vocab_size (int): Number of unique tokens in vocabulary.
            rnn_hidden_dim (int): Hidden dimension for Recurrent Layers.
            hidden_dim (int): Hidden dimension for fully-connected (FC) layers.
            dropout_p (float): Dropout proportion for FC layers.
            num_classes (int): Number of unique classes to classify into.
            device (torch.device): Device to run model on. Defaults to CPU.
            pretrained_embeddings (bool): Pretrained Embeddings used for representing data.
            freeze_embeddings (bool): Option to freeze pretrained embedding or train it on available data.
            padding_idx (int, optional): Index representing the `<PAD>` token. Defaults to 0.
        """

        super(RNN, self).__init__()

        ## Initialize embeddings
        if pretrained_embeddings is None:
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim, num_embeddings=vocab_size, padding_idx=padding_idx
            )
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float().to(device)
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim,
                num_embeddings=vocab_size,
                padding_idx=padding_idx,
                _weight=pretrained_embeddings,
            )

        ## Freeze embeddings or not
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

        ## RNN
        self.rnn = nn.GRU(embedding_dim, rnn_hidden_dim, batch_first=True, bidirectional=True)

        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(rnn_hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, channel_first=False):
        """Forward pass.

        Args:
            inputs (List): List of inputs (by feature).
            channel_first (bool, optional): Channel dimension is first in inputs. Defaults to False.

        Returns:
            Outputs from the model.

        """
        ## Inputs
        x_in, seq_lens = inputs

        ## Embed
        x_in = self.embeddings(x_in)

        ## RNN Outputs
        out, h_n = self.rnn(x_in)
        z = gather_last_relevant_hidden(hiddens=out, seq_lens=seq_lens)

        ## FC Layers
        z = self.fc1(z)
        z = self.dropout(z)
        z = self.fc2(z)

        return z


# Initialize model
def initialize_model(
    params: Namespace,
    vocab_size: int,
    num_classes: int,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Initialize a model using parameters (converted to appropriate data types).

    Args:
        params (Namespace): Parameters for data processing and training.
        vocab_size (int): Size of the vocabulary.
        num_classes (int): Number of unique classes.
        device (torch.device, optional): Device to run model on. Defaults to torch.device("cpu").

    Returns:
        nn.Module: Initialize torch model instance.
    """
    # Initialize model
    rnn_model = RNN(
        embedding_dim=int(params.embedding_dim),
        vocab_size=int(vocab_size),
        rnn_hidden_dim=int(params.rnn_hidden_dim),
        hidden_dim=int(params.hidden_dim),
        dropout_p=int(params.dropout_p),
        num_classes=int(num_classes),
        device=device,
    )
    rnn_model = rnn_model.to(device)
    return rnn_model