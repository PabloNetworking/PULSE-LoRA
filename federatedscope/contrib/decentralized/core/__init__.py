"""Method-agnostic decentralized runtime.

This package must never import ``..lora`` or ``..methods``. It knows about
peers, events, offered communication, transport, bytes, and transactions - and
nothing about LoRA, components, relay planes, or any concrete algorithm.
"""
