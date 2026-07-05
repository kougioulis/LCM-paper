import math

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    # performs the QKV mapping and then runs the full attention operation
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention # here the attention layer is passed from FullAttention
        self.query_attention = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape # batch_size, seq_len, feature_dim
        _, S, _ = keys.shape # _, seq_len, _
        H = self.n_heads 

        queries = self.query_attention(queries).view(B, L, H, -1) # reshape output to 4D
        keys = self.key_projection(keys).view(B, S, H, -1) # -//-
        values = self.value_projection(values).view(B, S, H, -1) # -//-

        out, attention = self.inner_attention(queries, keys, values)

        out = out.view(B, L, -1) # reshape attention output to 3D

        return self.out_projection(out), attention


class FullAttention(nn.Module):
    # Vanilla full attention (self-attention) computation
    def __init__(
        self, scale=None, attention_dropout=0.1, output_attention=False, **kwargs
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape # batch_size, seq_len, num_heads, d_q
        _, S, _, D = values.shape  # _, seq_len, _ , d_v

        # Vaswani et al. scaling factor
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # dot product of queries and keys

        A = self.dropout(torch.softmax(scale * scores, dim=-1)) # attention weights, softmax to the scaled scores
        V = torch.einsum("bhls,bshd->blhd", A, values) # attention output - A multiplied with values

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class SelfAttentionDistil(nn.Module):
    def __init__(self, c_in):
        super(SelfAttentionDistil, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=c_in, out_channels=c_in, kernel_size=3, padding=2, padding_mode="circular"
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
         # permute the input for Conv1D to match nn.Conv1d format
        x = self.conv(x.permute(0, 2, 1)) # [batch_size, seq_len, features] -> Conv1D([batch_size, c_in=features, seq_len])
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = torch.transpose(x, 1, 2) # permute the output again to match [batch_size, seq_len, d_model=c_out]

        return x