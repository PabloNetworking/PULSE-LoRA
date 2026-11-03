"""PULSE-LoRA: heterogeneous supports with a relay plane, over the shared LoRA substrate.

Everything specific to PULSE-LoRA lives in this package: the flat canonical
component space, the two-plane peer state (``S_i``, ``M_i``) with lazy
materialisation, the local-support controllers (``fixed``, ``adas_c``), session
selection and transactional mixing. Nothing in ``core/``, ``lora/`` or the other
methods imports it.
"""
from federatedscope.contrib.decentralized.methods.pulse_lora.method import (  # noqa: F401
    PulseLoRAFactory,
    PulseLoRAPeer,
)
