import torch
import torch.nn as nn
from packaging import version


class ConvEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3):
        super(ConvEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
        )
        # perform Kaiming-He weight initialization for the conv weights
        nn.init.kaiming_normal_(
            self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        if self.token_conv.bias is not None:
            nn.init.constant_(self.token_conv.bias, 0)

    def forward(self, x):

        seq_len = x.shape[1]

        # permute the input for Conv1D to match nn.Conv1d format
        # [batch_size, seq_len, features] -> Conv1D([batch_size, c_in=features, seq_len])
        out = self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)
        if out.shape[1] != seq_len:
            out = out[:, :seq_len, :] # truncate sequence length 

        return out # output shape is [batch_size, seq_len, d_model]