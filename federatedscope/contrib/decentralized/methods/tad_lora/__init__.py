"""TAD-LoRA baseline: alternating factor updates over graph-round mixing.

Importing this package registers the method. It imports no other method
package: only its own policy, the shared runtime and the shared LoRA layer.
"""
from federatedscope.contrib.decentralized.methods.tad_lora.method import (  # noqa: F401
    TADLoRAFactory,
    TADLoRAPeer,
)

__all__ = ["TADLoRAFactory", "TADLoRAPeer"]
