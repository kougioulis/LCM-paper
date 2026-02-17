import torch
from pathlib import Path

def setup_pytorch_globals():
    """
    A helper that registers necessary classes to allow loading of older OmegaConf-based checkpoints. 
    Used for loading the CP pretrained model without any unpickling issues for newer versions of PyTorch (2.6+).
    """
    import typing
    import collections
    import omegaconf
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig

    torch.serialization.add_safe_globals([
    omegaconf.base.ContainerMetadata,
    omegaconf.base.Metadata,
    omegaconf.nodes.AnyNode,
    ListConfig, DictConfig,
    collections.defaultdict,
    typing.Any, int, list, dict
    ])


def load_model_safely(model_class, path):
    path = Path(path)
    if not path.exists():
        print(f"Skipping: {path.name} (File not found)")
        return None
    try:
        return model_class.load_from_checkpoint(path, weights_only=False)
    except Exception as e:
        print(f"Error loading {path.name}: {e}")
        return None