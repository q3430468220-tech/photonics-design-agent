"""Device families: everything that depends on *what kind of device* is being designed.

A family bundles the pieces of the judge that are device-specific:
    features(geom, cfg)            descriptive measurements shown to the agent / used by model.py
    check(geom, cfg)               family rules on top of the generic raster DRC
    decompose(result, cfg)         backend result -> Breakdown (terms, closure)
    fom_and_constraints(bd, cfg)   scalar FOM (minimise) + constraint status
    summary_text(summary)          one line for the journal / digest
    improve_threshold(cfg, best)   how much better a result must be to count
    terms_help                     text for the prompt explaining the breakdown terms
    default_backends               (real, mock) backend names

The loop, the guard, the journal and the chart never look inside a family.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol


class Breakdown(Protocol):
    closure_ok: bool
    closure_err: float

    def summary(self) -> dict: ...

    def table(self) -> str: ...

    def to_dict(self) -> dict: ...


class Family:
    name: str = "base"
    terms_help: str = ""
    default_backends = ("", "mock")
    fom_label: str = "FOM"  # y-axis label of the evolution chart
    fom_unit: str = ""

    def features(self, geom, cfg) -> Dict[str, float]:
        raise NotImplementedError

    def check(self, geom, cfg) -> list:
        raise NotImplementedError

    def decompose(self, result, cfg):
        raise NotImplementedError

    def fom_and_constraints(self, bd, cfg) -> dict:
        raise NotImplementedError

    def summary_text(self, summary: dict) -> str:
        raise NotImplementedError

    def improve_threshold(self, cfg, best_fom: Optional[float]) -> float:
        raise NotImplementedError

    def table_from_dict(self, bd: dict) -> str:
        raise NotImplementedError


_FAMILIES: Dict[str, Family] = {}


def register(cls):
    _FAMILIES[cls.name] = cls()
    return cls


def get_family(name: str) -> Family:
    from . import passive_4port, pn_phase_shifter, pn_ring_modulator  # noqa: F401  (register)

    if name not in _FAMILIES:
        raise KeyError(f"unknown device family {name!r}; available: {sorted(_FAMILIES)}")
    return _FAMILIES[name]


def family_for(cfg) -> Family:
    return get_family(getattr(cfg, "device_family", "passive_4port"))
