from typing import Tuple

import pytorch_lightning as pl

import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from src.models.full_informer.model import Informer as model
from src.utils.utils import binary_metrics, corr_regularization


class LCMModule(pl.LightningModule):
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
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.95, 0.98),
        scheduler_factor: float = 0.1,
        loss_balancing = None,
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

        # Loss and metrics initialization
        self._setup_losses()
        self._setup_metrics()

    def _setup_losses(self) -> None:
        # Classification loss
        self.classifier_loss = nn.BCEWithLogitsLoss()
        
        # Multi-task loss balancing
        if self.hparams.loss_balancing == "MultiNoiseLoss":
            raise NotImplementedError
        elif self.hparams.loss_balancing == "VanillaMultiLoss":
            raise NotImplementedError
        else:
            self.multi_loss = None
            self.loss_term_scaling = torch.tensor([1.0, 0.75], device=self.device)

    def _setup_metrics(self) -> None:
        """Initialize validation metrics."""
        self.val_metrics = nn.ModuleDict({
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError()
    })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        inputs, labels = batch
        y_pred = self(inputs)

        y_class, y_reg = y_pred, None
        lab_class, lab_reg = labels, None

        # Calculate losses
        losses = {
            "class": self.classifier_loss(y_class, lab_class),
            "corr": self._calculate_corr_loss(inputs, y_class) if self.hparams.training_aids else 0
        }

        # Combine losses
        total_loss = self._combine_losses(losses["class"], losses["corr"])
        
        # Log metrics
        self._log_metrics(y_class, lab_class, stage)
        self._log_losses(losses["class"], losses["corr"], total_loss, stage)
        
        return total_loss

    def _calculate_corr_loss(self, inputs: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        data_sample = inputs[0] if self.hparams.training_aids else inputs
        return corr_regularization(torch.sigmoid(predictions), data_sample)

    def _combine_losses(self, class_loss: torch.Tensor, corr_loss: torch.Tensor) -> torch.Tensor:
        """Combine losses based on the selected weighting strategy."""
        if self.multi_loss:
            return self.multi_loss([corr_loss, class_loss])
        return (corr_loss * self.loss_term_scaling[1] + 
                class_loss * self.loss_term_scaling[0])

    def _log_metrics(self, y_pred: torch.Tensor, labels: torch.Tensor, stage: str) -> None:
        """Log classification metrics."""
        tpr, fpr, tnr, fnr, auc = binary_metrics(torch.sigmoid(y_pred), labels)
        self.log_dict({
            f"{stage}_tp": tpr.float(),
            f"{stage}_fp": fpr.float(),
            f"{stage}_tn": tnr.float(),
            f"{stage}_fn": fnr.float(),
            f"{stage}_auc": auc.float()
        }, sync_dist=True)

    def _log_losses(self, class_loss: torch.Tensor, corr_loss: torch.Tensor, total_loss: torch.Tensor, stage: str) -> None:
        """Log various loss components."""
        self.log(f"{stage}_loss", total_loss, sync_dist=True, prog_bar=True)
        self.log(f"{stage}_class_loss", class_loss, sync_dist=True)
        
        if self.hparams.training_aids:
            self.log(f"{stage}_corr_loss", corr_loss, sync_dist=True, prog_bar=(stage=="train"))

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        y_pred = self(inputs)
    
        class_loss = self.classifier_loss(y_pred, labels)
        corr_loss = self._calculate_corr_loss(inputs, y_pred) if self.hparams.training_aids else 0
        val_loss = self._combine_losses(class_loss, corr_loss)
    
        # Log all metrics explicitly
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_class_loss", class_loss, sync_dist=True)
        if self.hparams.training_aids:
            self.log("val_corr_loss", corr_loss, sync_dist=True)
    
        # Calculate and log other metrics
        tpr, fpr, tnr, fnr, auc = binary_metrics(torch.sigmoid(y_pred), labels)
        self.log_dict({
            "val_tp": tpr.float(),
            "val_fp": fpr.float(),
            "val_tn": tnr.float(),
            "val_fn": fnr.float(),
            "val_auc": auc.float()
        }, sync_dist=True)
    
        return val_loss

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.hparams.optimizer == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                weight_decay=self.hparams.weight_decay,
                amsgrad=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
    
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams.scheduler_factor,
                patience=self.hparams.scheduler_patience,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1
        }
    }