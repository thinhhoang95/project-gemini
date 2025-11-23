"""GEM module exports for ATFM arrival propagation."""

from .arrival_moments import ArrivalMoments
from .atfm_network import ATFMNetworkModel, ATFMRunResult, VolumeTimeSeries
from .hourly_kernel_table import EdgeKernel, HourlyKernelTable
from .regulation_plan import RegulationPlan, VolumeRegulation

__all__ = [
    "ArrivalMoments",
    "ATFMNetworkModel",
    "ATFMRunResult",
    "EdgeKernel",
    "HourlyKernelTable",
    "RegulationPlan",
    "VolumeRegulation",
    "VolumeTimeSeries",
    "GemRunResult",
    "GemService",
]


def __getattr__(name):
    if name in {"GemService", "GemRunResult"}:
        from .gem_service import GemRunResult, GemService

        return {"GemService": GemService, "GemRunResult": GemRunResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
