import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int=16, max_length: int=10000):
        """
        Sinusoidal positional embeddings as per Vaswani et al, 2017.
        """
        super(PositionalEmbedding, self).__init__()
        embedding = torch.zeros(max_length, d_model) # shape is [max_length, d_model]

        position = torch.arange(0, max_length).float().unsqueeze(1) # shape is [max_length, 1]
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        embedding[:, 0::2] = torch.sin(position * div_term) # even terms
        embedding[:, 1::2] = torch.cos(position * div_term) # odd terms

        embedding = embedding.unsqueeze(0) # [1, max_length, d_model]
        self.register_buffer("embedding", embedding)

    def forward(self, x):
        return self.embedding[:, : x.size(1)] # forward pass is of shape [1, x.size(1), d_model], just returns the positional encoding


class LearnableEmbedding(nn.Module):
    def __init__(self, d_model: int=16, max_length: int=500):
        """
        Learnable positional embeddings, initialized as random tensors of shape (L, d_model)
        [*] Wang, Yu-An, and Yun-Nung Chen. "What do position embeddings learn? an empirical study of pre-trained language model positional encoding." 
        arXiv preprint arXiv:2010.04903 (2020).
        """
        super(LearnableEmbedding, self).__init__()
        embedding = nn.Parameter(torch.rand(max_length, d_model))

        embedding = embedding.unsqueeze(0)
        self.register_buffer("embedding", embedding)

    def forward(self, x):
        return self.embedding[:, : x.size(1)] # forward pass is of shape [1, x.size(1), d_model], just returns the positional embedding


class RelativePositionEmbedding(nn.Module):
    def __init__(self, num_units: int, max_relative_position: int):
        """
        Relative positional embeddings according to Shaw et al., 2018
        [*] Shaw, Peter, Jakob Uszkoreit, and Ashish Vaswani. "Self-attention with Relative Position Representations." arXiv preprint arXiv:1803.02155 (2018).
        """
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings