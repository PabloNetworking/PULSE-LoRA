"""Dec-LoRA baseline.

Every peer holds the same rank and trains both factors locally; peers exchange
and mix the complete A/B state, in pair sessions or graph rounds depending on
the profile. No pruning, heterogeneous support, relay state, component selection
or factor alternation.
"""
from federatedscope.contrib.decentralized.methods.dec_lora.method import (  # noqa: F401
    DecLoRAFactory,
    DecLoRAPeer,
)
