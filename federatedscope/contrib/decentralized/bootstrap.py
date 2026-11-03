"""Imports the method packages, each of which registers itself in ``registry.py``.

``core/`` does not import ``methods/`` and ``registry.py`` imports no method, so
this module is the one place that does. Config validation and
``simulation/scenario.py`` call :func:`load_methods` before they look a method up.

Adding a method means adding its package to ``_METHOD_MODULES`` and its name to
``registry.RESERVED_METHOD_NAMES``.
"""
import importlib
import logging

logger = logging.getLogger(__name__)

#: Method packages to import, in order. Each registers itself on import.
#: A reserved name with no module here fails config validation with a clear
#: message rather than an ImportError.
_METHOD_MODULES = (
    "federatedscope.contrib.decentralized.methods.dec_lora",
    "federatedscope.contrib.decentralized.methods.pulse_lora",
    "federatedscope.contrib.decentralized.methods.tad_lora",
)

_loaded = False


def load_methods(force: bool = False) -> None:
    """Import every method package. Idempotent.

    Called by config validation and by ``simulation.scenario.build_factory``;
    whichever runs first does the work.
    """
    global _loaded
    if _loaded and not force:
        return
    for module_name in _METHOD_MODULES:
        importlib.import_module(module_name)
    _loaded = True
    logger.debug("decentralized plugins loaded: %s", ", ".join(_METHOD_MODULES))


def reset_for_tests() -> None:
    global _loaded
    _loaded = False
