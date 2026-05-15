"""CPU energy sampler.

- Linux: pyRAPL (Intel RAPL via MSR/perf_event).
- Other platforms: `tdp_estimate_wh()` returns TDP * duration as last-resort.
"""
from __future__ import annotations

import logging
import sys

from typing import Optional

log = logging.getLogger(__name__)


# Conservative TDP table for fallback (Watts, full-package average).
# Values approximate; declared in docs/aet/rebuttal.fr.md.
TDP_W: dict[str, float] = {
    "apple-m1": 20.0,
    "apple-m2": 22.0,
    "apple-m3": 25.0,
    "intel-xeon-8358": 250.0,
    "amd-epyc-7763": 280.0,
    "generic-cpu": 65.0,
    "generic-laptop-cpu": 28.0,
    # GPU defaults (used when running NN inference without NVML telemetry)
    "nvidia-a100": 300.0,
    "nvidia-h100": 700.0,
    "nvidia-v100": 250.0,
    "nvidia-rtx-3090": 350.0,
    "nvidia-rtx-4090": 450.0,
    "generic-gpu": 300.0,
}


def tdp_estimate_wh(hardware_id: Optional[str], duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    if hardware_id is None or hardware_id not in TDP_W:
        hardware_id = "generic-cpu"
    w = TDP_W[hardware_id]
    return w * duration_s / 3600.0


class RaplSampler:
    """RAPL-based CPU energy sampler (Linux only)."""

    def __init__(self):
        if not sys.platform.startswith("linux"):
            raise RuntimeError(
                f"RAPL only supported on Linux (got {sys.platform}); fall back to TDP"
            )
        try:
            import pyRAPL  # type: ignore
        except Exception as e:
            raise ImportError(f"pyRAPL not installed: {e}") from e
        self.pyRAPL = pyRAPL
        pyRAPL.setup()
        self._meter = pyRAPL.Measurement("aet-energy")
        self._started = False

    def start(self) -> None:
        self._meter.begin()
        self._started = True

    def stop(self) -> float:
        if not self._started:
            return 0.0
        self._meter.end()
        # pyRAPL reports energy per package in microjoules.
        result = self._meter.result
        if result is None or result.pkg is None:
            return 0.0
        total_uj = sum(result.pkg)
        # Add DRAM if available
        if getattr(result, "dram", None):
            total_uj += sum(result.dram)
        return total_uj / 1_000_000.0 / 3600.0  # uJ -> Wh
