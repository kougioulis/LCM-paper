import pytorch_lightning as pl
import torch

from lcm_pytorch.models.full_informer.model import Informer as model

# ================================
# Informer Module from Stein et al.
# ================================

class InformerModule(pl.LightningModule):
    def __init__(
        self,
        n_vars: int = 12,
        max_lag: int = 3,
        max_seq_len: int = 500,
        d_model: int = 16,
        n_heads: int = 1,
        n_blocks: int = 2,
        d_ff: int = 32,
        dropout_coeff: float = 0.05,
        attention_distilation: bool = True,
        training_aids: bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model initialization
        self.model = model(
            n_vars=self.hparams.n_vars,
            max_lag=self.hparams.max_lag,
            max_seq_len=self.hparams.max_seq_len,
            d_model=self.hparams.d_model,
            n_heads=self.hparams.n_heads,
            n_blocks=self.hparams.n_blocks,
            d_ff=self.hparams.d_ff,
            dropout_coeff=self.hparams.dropout_coeff,
            attention_distilation=self.hparams.attention_distilation,
            training_aids=self.hparams.training_aids
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
