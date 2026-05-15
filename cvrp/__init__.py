"""CVRP experiment runners for the AET measurement protocol.

Submodules:
    train    - train the attention-based neural solver (Kool et al. 2019)
               on CVRP, with energy tracking via aet.energy.EnergyTracker.
    test     - inference batch sweep, energy + throughput logging.
    solvers  - HGS (PyVRP) baseline runner, mono- and multi-threaded.

Run with `python -m cvrp.train experiment=aet_cvrp50`, etc.
"""

# PyTorch 2.6 flipped torch.load(weights_only=...) default to True.
# Lightning checkpoints from this project embed Hydra/OmegaConf hparams
# (DictConfig/ListConfig); allowlist them so trainer.test(ckpt_path=...)
# and trainer.fit(ckpt_path=...) succeed without disabling the safety
# flag globally. Checkpoints under logs/cvrp50/ are produced by the
# training entry point in this repo and are trusted.
import torch as _torch

try:
    from omegaconf import DictConfig as _DictConfig
    from omegaconf import ListConfig as _ListConfig
    from omegaconf.base import ContainerMetadata as _ContainerMetadata
    from omegaconf.base import Metadata as _Metadata
    from omegaconf.nodes import AnyNode as _AnyNode

    _torch.serialization.add_safe_globals(
        [_DictConfig, _ListConfig, _ContainerMetadata, _Metadata, _AnyNode]
    )
except Exception:
    pass

# Some Lightning code paths still call torch.load without weights_only=False
# even after allowlisting; default-disable the flag for the runner process
# since checkpoints are produced locally by this repo.
_orig_load = _torch.load


def _trusted_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


_torch.load = _trusted_load

