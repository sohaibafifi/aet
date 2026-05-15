"""Unified energy tracker for training, inference, and baseline runs.

Backend chain (default order):
  1. codecarbon — gives both Wh and gCO2eq via regional grid intensity.
  2. pynvml + pyRAPL — direct hardware counters (precise, no CO2 conversion).
  3. tdp_fallback — TDP × wall time (last resort; clearly flagged in `backend_used`).

PUE (datacenter Power Usage Effectiveness) is applied to operational energy.
Embodied carbon is reported separately and added to gCO2eq when
`report_embodied=True`.
"""
from __future__ import annotations

import logging
import time

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

from aet.energy.embodied import amortize_embodied
from aet.energy.nvml import NvmlSampler
from aet.energy.rapl import RaplSampler, tdp_estimate_wh

log = logging.getLogger(__name__)

DEFAULT_PUE = 1.4
DEFAULT_GRID_INTENSITY_G_PER_KWH = 475.0  # global average; codecarbon overrides per region


@dataclass
class EnergyReading:
    label: str
    duration_s: float = 0.0
    energy_wh: Optional[float] = None
    co2_g: Optional[float] = None
    co2_g_embodied: Optional[float] = None
    n_items: int = 0
    backend_used: str = "none"
    pue: float = DEFAULT_PUE
    hardware_id: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def throughput(self) -> Optional[float]:
        if self.duration_s <= 0 or self.n_items <= 0:
            return None
        return self.n_items / self.duration_s

    @property
    def energy_wh_per_item(self) -> Optional[float]:
        if self.energy_wh is None or self.n_items <= 0:
            return None
        return self.energy_wh / self.n_items

    @property
    def co2_g_total(self) -> Optional[float]:
        if self.co2_g is None:
            return self.co2_g_embodied
        return self.co2_g + (self.co2_g_embodied or 0.0)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "duration_s": self.duration_s,
            "energy_wh": self.energy_wh,
            "co2_g_operational": self.co2_g,
            "co2_g_embodied": self.co2_g_embodied,
            "co2_g_total": self.co2_g_total,
            "energy_wh_per_item": self.energy_wh_per_item,
            "throughput_items_per_s": self.throughput,
            "n_items": self.n_items,
            "backend_used": self.backend_used,
            "pue": self.pue,
            "hardware_id": self.hardware_id,
            **self.extra,
        }


class EnergyTracker:
    """Context manager measuring energy / CO2 / throughput of a code block.

    Usage:
        with EnergyTracker("train", pue=1.4, backend="codecarbon",
                           hardware_id="nvidia-a100") as t:
            do_work()
            t.n_items = 200_000  # for throughput

        reading = t.reading  # EnergyReading
    """

    def __init__(
        self,
        label: str,
        backend: str = "codecarbon",
        pue: float = DEFAULT_PUE,
        hardware_id: Optional[str] = None,
        report_embodied: bool = True,
        lifetime_s: Optional[float] = None,
        grid_intensity_g_per_kwh: float = DEFAULT_GRID_INTENSITY_G_PER_KWH,
        country_iso_code: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.label = label
        self.backend = backend
        self.pue = pue
        self.hardware_id = hardware_id
        self.report_embodied = report_embodied
        self.lifetime_s = lifetime_s
        self.grid_intensity = grid_intensity_g_per_kwh
        self.country_iso_code = country_iso_code
        self.output_dir = output_dir
        self.n_items: int = 0
        self.reading: EnergyReading = EnergyReading(label=label, pue=pue,
                                                   hardware_id=hardware_id)
        self._t0: float = 0.0
        self._codecarbon = None
        self._nvml = None
        self._rapl = None

    def __enter__(self) -> "EnergyTracker":
        self._t0 = time.perf_counter()
        self._start_backend()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        duration = max(0.0, time.perf_counter() - self._t0)
        self._stop_backend(duration)
        self.reading.duration_s = duration
        self.reading.n_items = self.n_items
        if self.report_embodied and self.hardware_id is not None:
            kwargs = {"hardware_id": self.hardware_id, "t_used_s": duration}
            if self.lifetime_s is not None:
                kwargs["t_lifetime_s"] = self.lifetime_s
            self.reading.co2_g_embodied = amortize_embodied(**kwargs)

    # --- backend dispatch ---
    def _start_backend(self) -> None:
        order = self._resolve_chain(self.backend)
        for name in order:
            if name == "codecarbon":
                if self._try_start_codecarbon():
                    self.reading.backend_used = "codecarbon"
                    return
            elif name == "hwcounters":
                if self._try_start_hwcounters():
                    self.reading.backend_used = "hwcounters"
                    return
            elif name == "tdp":
                self.reading.backend_used = "tdp"
                return
        self.reading.backend_used = "none"

    def _stop_backend(self, duration: float) -> None:
        backend = self.reading.backend_used
        if backend == "codecarbon" and self._codecarbon is not None:
            try:
                emissions_kg = self._codecarbon.stop()
                if emissions_kg is not None:
                    self.reading.co2_g = float(emissions_kg) * 1000.0
                final = getattr(self._codecarbon, "final_emissions_data", None)
                if final is not None:
                    e_kwh = float(getattr(final, "energy_consumed", 0.0))
                    self.reading.energy_wh = e_kwh * 1000.0 * self.pue
            except Exception as e:
                log.warning(f"codecarbon stop failed: {e}; falling back to TDP estimate")
                self._fill_tdp(duration)
        elif backend == "hwcounters":
            gpu_wh = self._nvml.stop() if self._nvml is not None else 0.0
            cpu_wh = self._rapl.stop() if self._rapl is not None else 0.0
            total_wh = (gpu_wh + cpu_wh) * self.pue
            self.reading.energy_wh = total_wh
            self.reading.co2_g = total_wh / 1000.0 * self.grid_intensity
        elif backend == "tdp":
            self._fill_tdp(duration)
        # backend == "none": leave None values

    def _try_start_codecarbon(self) -> bool:
        try:
            from codecarbon import EmissionsTracker, OfflineEmissionsTracker
        except Exception as e:
            log.info(f"codecarbon unavailable ({e}); trying hardware counters")
            return False
        try:
            save_to_file = self.output_dir is not None
            output_dir = self.output_dir if self.output_dir is not None else "."
            if self.country_iso_code:
                self._codecarbon = OfflineEmissionsTracker(
                    project_name=self.label,
                    country_iso_code=self.country_iso_code,
                    measure_power_secs=1,
                    log_level="error",
                    save_to_file=save_to_file,
                    output_dir=output_dir,
                    allow_multiple_runs=True,
                )
            else:
                self._codecarbon = EmissionsTracker(
                    project_name=self.label,
                    measure_power_secs=1,
                    log_level="error",
                    save_to_file=save_to_file,
                    output_dir=output_dir,
                    allow_multiple_runs=True,
                )
            self._codecarbon.start()
            return True
        except Exception as e:
            log.warning(f"codecarbon start failed: {e}")
            self._codecarbon = None
            return False

    def _try_start_hwcounters(self) -> bool:
        gpu_ok = False
        cpu_ok = False
        try:
            self._nvml = NvmlSampler()
            self._nvml.start()
            gpu_ok = True
        except Exception as e:
            log.info(f"NVML unavailable ({e})")
            self._nvml = None
        try:
            self._rapl = RaplSampler()
            self._rapl.start()
            cpu_ok = True
        except Exception as e:
            log.info(f"RAPL unavailable ({e})")
            self._rapl = None
        return gpu_ok or cpu_ok

    def _fill_tdp(self, duration: float) -> None:
        wh = tdp_estimate_wh(self.hardware_id, duration)
        wh *= self.pue
        self.reading.energy_wh = wh
        self.reading.co2_g = wh / 1000.0 * self.grid_intensity
        self.reading.extra["tdp_fallback"] = True

    @staticmethod
    def _resolve_chain(preferred: str) -> list[str]:
        order = ["codecarbon", "hwcounters", "tdp"]
        if preferred in order:
            order.remove(preferred)
            return [preferred, *order]
        return order


@contextmanager
def measure(label: str, **kwargs) -> Iterator[EnergyTracker]:
    """Functional alias for `with EnergyTracker(...) as t:`."""
    tracker = EnergyTracker(label=label, **kwargs)
    with tracker:
        yield tracker
