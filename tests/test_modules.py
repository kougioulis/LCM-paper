"""
Smoke tests for the LightningModule inference wrappers.
No model artifact required, run anywhere (including CI).
"""

import pytest
import torch

from lcm_pytorch.modules.informer_module import InformerModule
from lcm_pytorch.modules.lcm_module import LCMModule

MODULE_CLASSES = [LCMModule, InformerModule]


@pytest.mark.parametrize("module_cls", MODULE_CLASSES)
def test_forward_output_shape(module_cls, tiny_config):
    """Forward pass returns adj. tensor of shape [B, n_vars, n_vars, max_lag]"""
    model = module_cls(**tiny_config).eval()
    batch = 2
    x = torch.randn(batch, tiny_config["max_seq_len"], tiny_config["n_vars"])

    with torch.no_grad():
        out = model(x)

    assert out.shape == (
        batch,
        tiny_config["n_vars"],
        tiny_config["n_vars"],
        tiny_config["max_lag"],
    )
    assert torch.isfinite(out).all()
