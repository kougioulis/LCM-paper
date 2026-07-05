import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.embeddings.data_embedding import *
from src.models.embeddings.positional_embeddings import *
from src.models.embeddings.token_embedding import *
from src.models.full_informer.attention import *
from src.models.full_informer.encoder import *


#  https://github.com/martinwhl/Informer-PyTorch-Lightning
class Informer(nn.Module):
    def __init__(
        self,
        n_vars=12,
        max_lag=3,
        max_seq_len=500, # maximum length of the input time series
        d_model=32, # model dimension
        n_heads=4, # number of attention heads
        n_blocks=1, # number of encoder blocks
        d_ff=64, # feed-forward layer size
        dropout_coeff=0.05, # dropout rate
        activation="gelu", # GELU activation function as in the informer paper
        output_attention=False, # whether to return the attention weights
        attention_distilation=True, # distiled attention
        training_aids=False, # by default, don't concat crosscorrelation to the input
        **kwargs
    ):
        super(Informer, self).__init__()

        self.n_vars = n_vars
        self.max_lag = max_lag
        self.training_aids = training_aids

        self.norm = nn.BatchNorm1d(d_ff) # batch normalization
        self.activation = nn.ELU()

        self.enc_embedding = InputEmbedding( # using c_in = n_vars input channels 
            c_in=n_vars, d_model=d_model, dropout=dropout_coeff, max_length=max_seq_len, kernel_size=3
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            None,
                            attention_dropout=dropout_coeff,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout_coeff,
                    activation=activation,
                )
                for _ in range(n_blocks)
            ],
            (
                [SelfAttentionDistil(d_model) for _ in range(n_blocks - 1)] #c_\text{in} = d_\text{model}
                if attention_distilation
                else None
            ),
            nn.LayerNorm(d_model),
        )

        if self.training_aids:
           in_dim = d_model + (n_vars**2 * max_lag) # append the (flattened) crosscorrelation tensor to the input
        else:
           in_dim = d_model  # only use the last hidden state
       
        self.fc1 = torch.nn.Linear(in_dim, d_ff) # dim is [in_dim, d_ff]
        self.fc2 = torch.nn.Linear(d_ff, n_vars**2 * max_lag) # [d_ff, n_vars**2 * max_lags]

    # reshaping function 
    def reformat(self, x):
        return torch.reshape(x, (x.shape[0], self.n_vars, self.n_vars, self.max_lag)) # reshape to [n_vars, n_vars, max_lags] tensor

    # forward pass
    def forward(
        self,
        x_enc, # input time series data embedding (sum of token and positional embeddings)
        ):

        # Unpacking the input
        corr = None
        if isinstance(x_enc, (list, tuple)):
            if self.training_aids:
                x_enc, corr = x_enc[:2]
            else:
                x_enc = x_enc[0] if len(x_enc) > 0 else x_enc

        ## Embedding and encoding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attentions = self.encoder(enc_out)

        if self.training_aids:
            corr = corr.contiguous().view(corr.shape[0], -1) # flatten
        
            inp = torch.cat((enc_out[:, -1, :], corr), dim=1)
        else:
            inp = enc_out[:, -1, :]
        
        hidden1 = self.activation(self.norm(self.fc1(inp)))
        hidden2 = self.fc2(hidden1)

        return self.reformat(hidden2)