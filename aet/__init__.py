"""AET — Amortized Efficiency Threshold framework.

Minimal package exporting the energy-accounting subsystem used by the
AET measurement protocol.
"""
from aet.energy import EnergyTracker, measure

__all__ = ["EnergyTracker", "measure"]
