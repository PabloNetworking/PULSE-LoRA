"""Decentralized method plugins.

Each plugin owns its algorithm state and policy and reuses the shared runtime
and LoRA substrate. Plugins are imported by ``..bootstrap``, never by ``core/``.
"""
