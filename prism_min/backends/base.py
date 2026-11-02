"""Backend interface. A backend turns a Geometry into normalised flux data.

Every quantity is a fraction of the injected source power, per wavelength:

    through_total  : all power leaving through the through port
    through_fund   : power in the fundamental mode at the through port
    side[name]     : total power leaving through each side (crosstalk) port
    reflection     : power travelling backwards at the input port
    box_total      : net outward flux through a closed box enclosing the device
                     (ports included). Should equal 1 for a lossless, converged run.
                     This is the closure check - it is measured, not inferred.

Backends never read agent-editable files and never accept mesh/PML arguments:
everything numerical comes from the locked SimConfig.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..config import SimConfig
from ..geometry import Geometry


@dataclass
class SimResult:
    wavelengths: np.ndarray
    through_total: np.ndarray
    through_fund: np.ndarray
    side: Dict[str, np.ndarray]
    reflection: np.ndarray
    box_total: np.ndarray
    converged: bool = True
    backend: str = ""
    runtime_s: float = 0.0
    extra: Dict = field(default_factory=dict)  # file paths, decay level, notes

    def to_dict(self) -> dict:
        return {
            "wavelengths": self.wavelengths.tolist(),
            "through_total": self.through_total.tolist(),
            "through_fund": self.through_fund.tolist(),
            "side": {k: v.tolist() for k, v in self.side.items()},
            "reflection": self.reflection.tolist(),
            "box_total": self.box_total.tolist(),
            "converged": bool(self.converged),
            "backend": self.backend,
            "runtime_s": float(self.runtime_s),
            "extra": {k: (v if isinstance(v, (str, int, float, bool, list, dict)) else str(v)) for k, v in self.extra.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "SimResult":
        return SimResult(
            np.asarray(d["wavelengths"]),
            np.asarray(d["through_total"]),
            np.asarray(d["through_fund"]),
            {k: np.asarray(v) for k, v in d["side"].items()},
            np.asarray(d["reflection"]),
            np.asarray(d["box_total"]),
            d.get("converged", True),
            d.get("backend", ""),
            d.get("runtime_s", 0.0),
            d.get("extra", {}),
        )


class Backend:
    name = "base"

    def run(self, geom: Geometry, cfg: SimConfig, run_dir: Optional[Path] = None) -> SimResult:
        raise NotImplementedError

    def close(self) -> None:  # long-lived sessions override
        pass


_REGISTRY = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def get_backend(name: str, **kw) -> Backend:
    from . import mock  # noqa: F401  (registers)

    from . import mock_pn  # noqa: F401

    if name == "lumerical":
        from . import lumerical  # noqa: F401
    if name == "lumerical_charge_fde":
        from . import lumerical_charge_fde  # noqa: F401
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kw)
