"""NVML GPU energy sampler.

Primary path: `nvmlDeviceGetTotalEnergyConsumption` (mJ counter, monotonic).
Fallback: integrate `nvmlDeviceGetPowerUsage` over a polling thread.

Raises ImportError or RuntimeError if no usable GPU is available; the tracker
catches these and moves on to the next backend.
"""
from __future__ import annotations

import threading
import time

from typing import Optional


class NvmlSampler:
    """Energy sampler covering one or more GPUs via pynvml."""

    def __init__(self, gpu_indices: Optional[list[int]] = None, poll_hz: float = 10.0):
        try:
            import pynvml  # type: ignore
        except Exception as e:  # pragma: no cover - environment dependent
            raise ImportError(f"pynvml not installed: {e}") from e
        self.pynvml = pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            pynvml.nvmlShutdown()
            raise RuntimeError("no NVML devices detected")
        self.gpu_indices = (
            gpu_indices if gpu_indices is not None else list(range(device_count))
        )
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_indices]
        self.poll_dt = 1.0 / max(poll_hz, 1e-3)
        self._use_counter = self._counters_supported()
        self._start_mj: list[int] = []
        self._stop_mj: list[int] = []
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()
        self._poll_energy_wh = 0.0

    def _counters_supported(self) -> bool:
        for h in self.handles:
            try:
                self.pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            except Exception:
                return False
        return True

    def start(self) -> None:
        if self._use_counter:
            self._start_mj = [
                self.pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles
            ]
        else:
            self._poll_stop.clear()
            self._poll_energy_wh = 0.0
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()

    def stop(self) -> float:
        if self._use_counter:
            self._stop_mj = [
                self.pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles
            ]
            mj = sum(stop - start for start, stop in zip(self._start_mj, self._stop_mj))
            energy_wh = mj / 3_600_000.0
        else:
            self._poll_stop.set()
            if self._poll_thread is not None:
                self._poll_thread.join(timeout=2.0)
            energy_wh = self._poll_energy_wh
        try:
            self.pynvml.nvmlShutdown()
        except Exception:
            pass
        return energy_wh

    def _poll_loop(self) -> None:
        last = time.perf_counter()
        while not self._poll_stop.is_set():
            now = time.perf_counter()
            dt = now - last
            last = now
            total_w = 0.0
            for h in self.handles:
                try:
                    total_w += self.pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    continue
            self._poll_energy_wh += total_w * dt / 3600.0
            self._poll_stop.wait(self.poll_dt)
