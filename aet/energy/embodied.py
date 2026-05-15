"""Embodied carbon table and amortization helpers.

Sources:
- Gupta et al. 2021 "Chasing Carbon" (ASPLOS) — GPU/CPU fabrication carbon
- Luccioni et al. 2023 "Estimating the Carbon Footprint of BLOOM"
- Apple Product Environmental Reports (Apple Silicon)

All values in kgCO2eq for full hardware lifetime fabrication.
Lifetime default: 5 years * 8760 h = 43800 h = 157_680_000 s
(standard datacenter amortization assumption; user can override).
"""
from dataclasses import dataclass

DEFAULT_LIFETIME_S = 5 * 365 * 24 * 3600  # 5 years


@dataclass(frozen=True)
class EmbodiedCarbon:
    hardware_id: str
    kind: str  # "gpu" | "cpu" | "soc"
    kg_co2eq: float
    source: str


# Table — conservative midpoints. Cite the source in the blog post.
TABLE: dict[str, EmbodiedCarbon] = {
    # GPUs (datacenter).
    # Hopper / Blackwell values extrapolated from H100 (Gupta 2021 model)
    # by die area and HBM stack count. NVIDIA does not publish embodied
    # carbon for any datacenter SKU; values flagged as estimates.
    "nvidia-v100":  EmbodiedCarbon("nvidia-v100",  "gpu", 130.0, "Luccioni 2023"),
    "nvidia-a100":  EmbodiedCarbon("nvidia-a100",  "gpu", 150.0, "Luccioni 2023"),
    "nvidia-h100":  EmbodiedCarbon("nvidia-h100",  "gpu", 200.0, "Gupta 2021 extrapolation"),
    "nvidia-h200":  EmbodiedCarbon("nvidia-h200",  "gpu", 215.0, "H100 + HBM3e stacks (est.)"),
    "nvidia-b100":  EmbodiedCarbon("nvidia-b100",  "gpu", 380.0, "2x reticle die + HBM3e (est.)"),
    "nvidia-b200":  EmbodiedCarbon("nvidia-b200",  "gpu", 400.0, "2x reticle die + HBM3e (est.)"),
    "nvidia-gb200": EmbodiedCarbon("nvidia-gb200", "gpu", 900.0, "2x B200 + Grace CPU + NVLink (est.)"),
    # AMD Instinct (CDNA) datacenter accelerators.
    # No published fab carbon; values extrapolated from total silicon
    # area (chiplets + HBM stacks) using Gupta 2021 area model.
    "amd-mi250x":   EmbodiedCarbon("amd-mi250x",   "gpu", 140.0, "Dual MCM + HBM2e (est.)"),
    "amd-mi300x":   EmbodiedCarbon("amd-mi300x",   "gpu", 250.0, "8 XCD + 4 IOD + 8 HBM3 chiplets (est.)"),
    "amd-mi325x":   EmbodiedCarbon("amd-mi325x",   "gpu", 260.0, "MI300X + denser HBM3e (est.)"),
    "amd-mi350x":   EmbodiedCarbon("amd-mi350x",   "gpu", 300.0, "CDNA4, larger XCDs (est.)"),
    # GPUs (consumer)
    "nvidia-rtx-3090": EmbodiedCarbon(
        "nvidia-rtx-3090", "gpu", 100.0, "Gupta 2021 extrapolation"
    ),
    "nvidia-rtx-4090": EmbodiedCarbon(
        "nvidia-rtx-4090", "gpu", 120.0, "Gupta 2021 extrapolation"
    ),
    # CPUs (server).
    "intel-xeon-8358": EmbodiedCarbon("intel-xeon-8358", "cpu", 50.0, "Gupta 2021"),
    "intel-xeon-8480":  EmbodiedCarbon("intel-xeon-8480",  "cpu", 65.0, "Sapphire Rapids, area scaling (est.)"),
    "amd-epyc-7763":   EmbodiedCarbon("amd-epyc-7763",   "cpu", 60.0, "Gupta 2021"),
    "amd-epyc-9654":   EmbodiedCarbon("amd-epyc-9654",   "cpu", 70.0, "Genoa, 96c chiplets (est.)"),
    "amd-epyc-9754":   EmbodiedCarbon("amd-epyc-9754",   "cpu", 75.0, "Bergamo, 128c chiplets (est.)"),
    "amd-epyc-9965":   EmbodiedCarbon("amd-epyc-9965",   "cpu", 85.0, "Turin, 192c chiplets (est.)"),
    # SoC (Apple Silicon).
    # Base values from Apple Product Environmental Reports (production
    # stage, SoC portion attributed via Gupta 2021 die-area scaling).
    # Pro/Max/Ultra values scaled by die area: Pro ~1.7x, Max ~2.5x,
    # Ultra ~5x the base die (two Max dies on an interposer).
    "apple-m1":       EmbodiedCarbon("apple-m1",       "soc",  30.0, "Apple PER 2020"),
    "apple-m1-pro":   EmbodiedCarbon("apple-m1-pro",   "soc",  51.0, "Apple PER 2021 + die scaling"),
    "apple-m1-max":   EmbodiedCarbon("apple-m1-max",   "soc",  75.0, "Apple PER 2021 + die scaling"),
    "apple-m1-ultra": EmbodiedCarbon("apple-m1-ultra", "soc", 150.0, "Apple PER 2022 + die scaling"),
    "apple-m2":       EmbodiedCarbon("apple-m2",       "soc",  32.0, "Apple PER 2022"),
    "apple-m2-pro":   EmbodiedCarbon("apple-m2-pro",   "soc",  54.0, "Apple PER 2023 + die scaling"),
    "apple-m2-max":   EmbodiedCarbon("apple-m2-max",   "soc",  80.0, "Apple PER 2023 + die scaling"),
    "apple-m2-ultra": EmbodiedCarbon("apple-m2-ultra", "soc", 160.0, "Apple PER 2023 + die scaling"),
    "apple-m3":       EmbodiedCarbon("apple-m3",       "soc",  34.0, "Apple PER 2023"),
    "apple-m3-pro":   EmbodiedCarbon("apple-m3-pro",   "soc",  58.0, "Apple PER 2023 + die scaling"),
    "apple-m3-max":   EmbodiedCarbon("apple-m3-max",   "soc",  85.0, "Apple PER 2023 + die scaling"),
    "apple-m3-ultra": EmbodiedCarbon("apple-m3-ultra", "soc", 170.0, "Apple PER 2024 + die scaling"),
    "apple-m4":       EmbodiedCarbon("apple-m4",       "soc",  36.0, "Apple PER 2024"),
    "apple-m4-pro":   EmbodiedCarbon("apple-m4-pro",   "soc",  61.0, "Apple PER 2024 + die scaling"),
    "apple-m4-max":   EmbodiedCarbon("apple-m4-max",   "soc",  90.0, "Apple PER 2024 + die scaling"),
    "apple-m5":       EmbodiedCarbon("apple-m5",       "soc",  38.0, "Apple PER 2025 estimate"),
    "apple-m5-pro":   EmbodiedCarbon("apple-m5-pro",   "soc",  65.0, "Apple PER 2025 + die scaling"),
    "apple-m5-max":   EmbodiedCarbon("apple-m5-max",   "soc",  95.0, "Apple PER 2025 + die scaling"),
    # Generic fallback
    "generic-gpu": EmbodiedCarbon("generic-gpu", "gpu", 150.0, "midpoint"),
    "generic-cpu": EmbodiedCarbon("generic-cpu", "cpu", 55.0, "midpoint"),
}


def amortize_embodied(
    hardware_id: str,
    t_used_s: float,
    t_lifetime_s: float = DEFAULT_LIFETIME_S,
) -> float:
    """Return amortized embodied carbon in gCO2eq for a usage window.

    Linear-time amortization: fraction of lifetime consumed = t_used / t_lifetime.
    """
    if hardware_id not in TABLE:
        kind = "gpu" if "gpu" in hardware_id.lower() else "cpu"
        entry = TABLE[f"generic-{kind}"]
    else:
        entry = TABLE[hardware_id]
    if t_lifetime_s <= 0:
        return 0.0
    share = max(0.0, min(1.0, t_used_s / t_lifetime_s))
    return entry.kg_co2eq * 1000.0 * share
