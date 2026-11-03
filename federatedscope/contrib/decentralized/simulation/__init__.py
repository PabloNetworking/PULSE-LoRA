"""Scenario-driven simulation of a decentralized run in one process.

``scenario`` describes a run, ``credit_trace`` writes contact traces with byte credits,
``glue_data`` / ``partition`` / ``readout`` build its data and holdout, ``training`` fixes its
training seeds, ``wire_quote`` prices sessions in the serialized bytes of the ``socket_v1``
protocol (whose encoders are in ``socket_v1``), ``meter`` records those bytes per session, and
``driver`` executes a scenario with
:class:`~federatedscope.contrib.decentralized.core.runner.DecentralizedRunner`.

Importing this package registers the ``socket_v1`` wire-cost model.
"""
from federatedscope.contrib.decentralized.simulation import wire_quote  # noqa: F401
