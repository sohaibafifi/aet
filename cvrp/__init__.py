"""CVRP experiment runners for the AET measurement protocol.

Submodules:
    train    - train the attention-based neural solver (Kool et al. 2019)
               on CVRP, with energy tracking via aet.energy.EnergyTracker.
    test     - inference batch sweep, energy + throughput logging.
    solvers  - HGS (PyVRP) baseline runner, mono- and multi-threaded.

Run with `python -m cvrp.train experiment=aet_cvrp50`, etc.
"""
