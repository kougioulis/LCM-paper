import torch
import torch.nn as nn

from src.models.embeddings.positional_embeddings import *
from src.models.embeddings.token_embedding import *


class InputEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, max_length=500, kernel_size=3):
        """
        Input Embedding Class
        This class consists of a 1D convolutional embedding of the input time-series, with positional embeddings added element-wise, and passed
        through a dropout layer.
        Number of input channels c_in is meant to correspond to the number of the input time-series.
        """
        super(InputEmbedding, self).__init__()
        self.value_embedding = ConvEmbedding(c_in, d_model, kernel_size=kernel_size)
        self.position_embedding = PositionalEmbedding(d_model, max_length=max_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a = self.value_embedding(x) # project input tokens
        b = self.position_embedding(x)
        x = a + b # element-wise addition  
        return self.dropout(x)
    

class RPEInputEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, kernel_size=3):
        """
        Relative Positional Embedding Class 
        Just the 1D convolutional embeddings, with the positional embeddings learnt inside the self-attention mechanism.
        Number of input channels c_in is meant to correspond to the number of the input time-series.
        
        [*] P. Shaw, J. Uszkoreit, A. Vaswani. Self-Attention with Relative Position Representations, NAACL 2018.
        """
        super(RPEInputEmbedding, self).__init__()
        self.value_embedding = ConvEmbedding(c_in, d_model, kernel_size=kernel_size)  # Keep token embedding only
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a = self.value_embedding(x)  # embed input tokens
        x = a  # no need for positional embedding as they are learnt in the self-attention mechanism
        return self.dropout(x) 