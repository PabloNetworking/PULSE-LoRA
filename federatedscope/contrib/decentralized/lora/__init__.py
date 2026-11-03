"""LoRA layer shared by the decentralized methods.

Schema, adapter access, payload codec, common canonical initialization,
factor-group and optimizer-state helpers, and mixing kernels. May depend on
``..core``, never on ``..methods``; an invariant specific to one method belongs
under that method.
"""
