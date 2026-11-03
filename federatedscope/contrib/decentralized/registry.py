"""Decentralized method registry.

This module imports no method package; the methods register themselves when
``bootstrap.py`` imports them.

Recognizing a name is separate from importing a method. Config validation runs
for every configuration, including ones whose ``federate.method`` is not a
decentralized method (the default is ``FedAvg``), and must not import the
torch-heavy method modules to decide that.
"""
from typing import Callable, Dict, Tuple

#: Names this package claims. Declared statically so recognition needs no import.
#: Config validation checks that a reserved name is registered once
#: ``bootstrap.load_methods()`` has run.
RESERVED_METHOD_NAMES: Tuple[str, ...] = (
    "dec_lora",
    "pulse_lora",
    "tad_lora",
    # A scalar test double (tests/core/toy_method.py), not a method of this
    # release. Reserved so end-to-end runner tests go through config validation.
    "toy_method",
)

_FACTORIES: Dict[str, Callable] = {}


def normalise_method_name(method) -> str:
    """Lowercase, trim, and fold hyphens to underscores."""
    if method is None:
        return ""
    return str(method).strip().lower().replace("-", "_")


def is_reserved_name(method) -> bool:
    """Cheap check: does this package claim the name? No plugin import."""
    return normalise_method_name(method) in RESERVED_METHOD_NAMES


def register_decentralized_method(name: str, factory: Callable,
                                  override: bool = False) -> None:
    """Register a factory **class** (or any callable) built as ``factory(cfg)``.

    Registering the class rather than an instance lets a method declare a
    cfg-dependent interaction form - Dec-LoRA's paper_fixed_q profile is a graph
    round while common_trace_pairwise is a pair session - and keeps
    ``validate_config`` reachable before any instance exists.
    """
    key = normalise_method_name(name)
    if not key:
        raise ValueError("decentralized method name must be non-empty")
    if key in _FACTORIES and not override and _FACTORIES[key] is not factory:
        raise ValueError(
            f"decentralized method {key!r} is already registered as "
            f"{_FACTORIES[key]!r}. Silently overwriting it would make which "
            f"implementation runs depend on import order; pass override=True if "
            f"the replacement is intended."
        )
    _FACTORIES[key] = factory


def is_decentralized_method(method) -> bool:
    """Is the name registered *right now*? Call ``load_methods()`` first."""
    return normalise_method_name(method) in _FACTORIES


def get_factory(method) -> Callable:
    key = normalise_method_name(method)
    if key not in _FACTORIES:
        raise KeyError(
            f"decentralized method {key!r} is not registered; "
            f"registered: {sorted(_FACTORIES)}"
        )
    return _FACTORIES[key]


def known_methods() -> Tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def clear_registry() -> None:
    """Test hook. Not used by the runtime."""
    _FACTORIES.clear()
