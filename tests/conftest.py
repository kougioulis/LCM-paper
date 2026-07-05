"""Shared pytest fixtures and path setup.

Adds the repository root to ``sys.path`` so ``import lcm_pytorch...`` works no matter
where pytest is invoked from, and exposes a tiny model config so the tests run
fast on CPU and never touch a checkpoint.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def tiny_config() -> dict:
    """A minimal LCM config; builds instantly."""
    return {
        "n_vars": 5,
        "max_lag": 3,
        "max_seq_len": 50,
        "d_model": 16,
        "n_heads": 1,
        "n_blocks": 2,
        "d_ff": 32,
        "dropout_coeff": 0.05,
        "attention_distilation": True,
        "training_aids": False,
    }
