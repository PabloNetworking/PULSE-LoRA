"""Decentralized LoRA fine-tuning: runtime, LoRA layer, methods and simulation.

Three layers with a one-way dependency ``core -> lora -> methods``:

- ``core/``    method-agnostic runtime; knows nothing of LoRA
- ``lora/``    shared LoRA layer; knows no concrete method
- ``methods/`` method-specific state and policy (PULSE-LoRA, Dec-LoRA, TAD-LoRA)

``simulation/`` runs a scenario on top of them. ``registry.py`` maps method names
to factories and ``bootstrap.py`` imports the method packages.
"""
