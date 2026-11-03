"""Registry isolation for the decentralized tests.

Some tests register a factory double for a real method so that they run on a
toy adapter on CPU. The method registry is global, so a double would otherwise
stay registered for every later test in the process. The autouse fixture
``isolate_decentralized_registry`` saves the registry before each test and
restores it afterwards, which keeps the suite independent of test order.

A double registered while pytest imports a test module (for example by building
a peer inside a ``@pytest.mark.parametrize`` argument) is already in the
registry when the fixture saves it, so the fixture cannot undo it.
``pytest_collection_finish`` checks after collection that every registered
factory comes from the methods package, and stops the run otherwise.
"""
import pytest

from federatedscope.contrib.decentralized import bootstrap, registry


def pytest_collection_finish(session):
    """Stop the run if a factory double was registered during collection."""
    bootstrap.load_methods()
    substituted = [
        name for name, factory in registry._FACTORIES.items()
        if not factory.__module__.startswith(
            "federatedscope.contrib.decentralized.methods")
    ]
    if substituted:
        raise pytest.UsageError(
            f"collection substituted the decentralized factories {substituted}. "
            f"A test double was registered while importing a module, for example "
            f"by constructing a peer inside a @parametrize argument, which pytest "
            f"evaluates before any fixture runs. The per-test isolation fixture "
            f"saves the registry after that, so it cannot undo it."
        )


@pytest.fixture(autouse=True)
def isolate_decentralized_registry():
    bootstrap.load_methods()          # real plugins present before any substitution
    saved = dict(registry._FACTORIES)
    yield
    registry._FACTORIES.clear()
    registry._FACTORIES.update(saved)
