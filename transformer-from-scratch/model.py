import math
import torch
import torch.nn as nn
import torch.functional as F

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the Transformer model.

        Args:
            d_model (int): The embedding size.
            vocab_size (int): The size of the vocabulary.

        Returns:
            None
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the forward pass.
        """
        # in original paper, the embeddings are multiplied by sqrt of d_model
        return self.embedding(x) * math.sqrt(self.d_model)