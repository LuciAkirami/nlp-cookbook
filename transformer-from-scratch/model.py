import math
import torch
import torch.nn as nn
import torch.functional as F

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the InputEmbedding model.

        Args:
            d_model (int): The embedding size.
            vocab_size (int): The size of the vocabulary.

        Returns:
            None
        """
        super().__init__()
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

# positional encoding stores the positional information of the tokens in the sequence     
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """
         Initialize the PositionalEncoding model.

        Args:
            d_model (int): The embedding size.
            seq_len (int): The sequence length of the input sequence.
            dropout (float): The dropout rate.

        Returns:
            None
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # the below is the same as the original paper
        # pe = torch.zeros(seq_len, d_model)
        # # go through each position 
        # for pos in range(seq_len):
        #     # go through each vector component of the position
        #     for i in range(0, d_model, 2):
        #         # if even index
        #         pe[pos, i] = \
        #         math.sin(pos / (10000 ** ((2 * i)/d_model)))
        #         # if odd index
        #         pe[pos, i + 1] = \
        #         math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        # pe = pe.unsqueeze(0)
        # self.register_buffer('pe', pe)

        # here we are changing the div_term part of the code

        # create constant 'pe' matrix with values dependant on 
        # pos and i

        # Create a matrix 'pe' with zeros of size (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a tensor 'position' with values from 0 to seq_len of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Calculate the 'div_term' using exponential and log functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
                                (-math.log(10000.0) / d_model))
        # Apply sin function to even indices of 'pe'
        pe[:,0::2] = torch.sin(position/div_term)
        # Apply cos function to odd indices of 'pe'
        pe[:,1::2] = torch.cos(position/div_term)

        # Add an extra dimension to 'pe' at position 0 into include batch size
        # (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register 'pe' as a buffer that should not to be considered a model parameter
        self.register_buffer(pe)
    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the forward pass.
        """

        # Add positional encoding to the input tensor 'x'
        # 'requires_grad=False' indicates that gradients need not be computed for 'pe'
        seq_len = x.size(1)
        x = x + torch.tensor(self.pe[:, :seq_len], \
                             requires_grad=False)
        # Apply dropout
        return self.dropout(x)
    
"""
The LayerNormalization class is a type of normalization technique like Batch Normalization. 
However, unlike Batch Normalization, Layer Normalization performs normalization on the last 
dimension (features) instead of the batch dimension. 
This makes it batch size independent and can be used in a variety of contexts where 
Batch Normalization cannot be used.
"""
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10 ** -6):
        """
        Initialize the LayerNormalization model.

        Args:
            eps (float): A small number to prevent division by zero. Default is 10^-6.

        Returns:
            None
        """
        super().__init__()

        # If std is close to 0 then, (x-mean) / std becomes a very large number
        # and becomes difficult to calculate for the GPU. Hence, an epsilon 'eps' is used
        self.eps = eps

        # The nn.Parameter creates a learnable parameter
        # 'alpha' and 'bias' are two learnable parameters for the model
        # 'alpha' is a multiplicative factor and 'bias' is an additive factor
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplicative factor
        self.bias = nn.Parameter(torch.zeros(1))  # Additive factor

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the forward pass.
        """

        # Calculate the mean of 'x' along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        # Calculate the standard deviation of 'x' along the last dimension
        std = x.std(dim=-1, keepdim=True)

        # Normalize 'x' by subtracting the mean and dividing by the standard deviation
        # Multiply the result by 'alpha' and add 'bias'
        # The small number 'eps' is added to the denominator to prevent division by zero
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# the feedforward layer is the same as in the original paper
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Initialize the FeedForwardLayer model.

        Args:
            d_model (int): The embedding size.
            d_ff (int): The dimension of the feedforward network model.
            dropout (float): The dropout rate.

        Returns:
            None
        """
        super().__init__()

        # Define the first linear layer with input size 'd_model' and output size 'd_ff'
        self.linear_1 = nn.Linear(d_model, d_ff)
        # Define the ReLU activation function
        self.relu = nn.ReLU()
        # Define the dropout layer with dropout rate 'dropout'
        self.dropout = nn.Dropout(dropout)
        # Define the second linear layer with input size 'd_ff' and output size 'd_model'
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the forward pass.
        """

        # Pass the input 'x' through the first linear layer
        x = self.linear_1(x)
        # Apply the ReLU activation function
        x = self.relu(x)
        # Apply dropout
        x = self.dropout(x)
        # Pass the result through the second linear layer
        x = self.linear_2(x)

        return x