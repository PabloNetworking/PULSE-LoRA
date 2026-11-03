# Data

Datasets are not redistributed in this repository. Everything in `data/` except this file is ignored
by git.

## GLUE

The GLUE classification tasks (MNLI, QNLI, RTE, SST-2, MRPC, QQP, CoLA) are downloaded from the Hugging
Face Hub with `datasets` on first use and cached under `data.root` (`data/` in the example scenarios).
`datasets` also writes lock files there.

A scenario splits the training split between peers: FederatedScope's IID split, then the scenario's
`data_partition` if it states one (`label_stratified_round_robin_v1` in the example scenarios, or a
Dirichlet split with `{"splitter": "lda", "alpha": a, "seed": s}`). The readout holdout is drawn from the
training split and removed from every peer's training data. The GLUE validation and test splits are not
used for readouts.

## Contact traces

A scenario embeds its contact trace: for every frame, the pairs of peers that can hold a session, the
session duration and the goodput in each direction. `scripts/make_scenario.py` generates traces with
Erdős–Rényi contacts and per-direction byte credits; a trace written by other means can be embedded in a
scenario as long as it follows the same format (`federatedscope/contrib/decentralized/core/traces.py`).
