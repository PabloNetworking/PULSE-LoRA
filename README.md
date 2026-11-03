# PULSE-LoRA: Propagation-Preserving Decentralized Fine-Tuning under Heterogeneous Compute and Dynamic Peer Interactions

> **Work in progress.** This is a research implementation under active development. The method, its
> configuration and the code may still change, and no results have been published yet.

PULSE-LoRA (Peer-interaction Utility-aware LoRA with Support-preserving Exchange) fine-tunes a shared
LoRA adapter across peers that have different compute budgets and meet only through time-varying pairwise
sessions, without a server that owns the model. Each peer's adapter is a set of globally indexed rank-1
components, as in [AdaS-LoRA](https://github.com/PabloNetworking/AdaS-LoRA). A peer keeps two views of it:

- a **training plane** holding the components it can afford to train, its compute rank;
- a **relay plane** holding the latest value of every other component it has seen, so that components
  keep propagating through peers that do not train them.

When two peers meet, they turn the session's byte credit into a budget of complete components, reserve
part of it for the components that have waited longest, fill the rest by a utility score, and mix the
chosen components symmetrically in an atomic transaction. Components that neither peer has touched are
never sent: every peer derives the same initial value for them from a common seed.

This release simulates decentralized fine-tuning of DeBERTa on GLUE: one process runs every peer over a
contact trace with per-direction byte credits. Dec-LoRA and TAD-LoRA are included as baselines.

## Methods

| Method | `method` | Interaction | Example scenario |
| --- | --- | --- | --- |
| PULSE-LoRA | `pulse_lora` | pairwise sessions | [scenarios/pulse_lora_mnli.json](scenarios/pulse_lora_mnli.json) |
| Dec-LoRA | `dec_lora` | graph rounds (Metropolis weights over each frame's contacts) | [scenarios/dec_lora_mnli.json](scenarios/dec_lora_mnli.json) |
| TAD-LoRA | `tad_lora` | graph rounds, training the A and B factors alternately every T local episodes | [scenarios/tad_lora_mnli.json](scenarios/tad_lora_mnli.json) |

What the PULSE-LoRA implementation covers:

- **Canonical component space with common initialization.** All peers and methods address one space of
  200 rank-1 components. Their initial values are a deterministic function of the backbone files, the
  target modules and a model-initialization seed (`common_canonical_v1`), and the classifier and pooler are
  initialized identically on every peer.
- **Two-plane state.** Training support and relay storage, with relay values materialized lazily.
- **Local support.** Fixed supports (each peer keeps its initial rank), or the conservative rank-1 pruning
  rule of AdaS-LoRA (`controller: adas_c`): a weak component is pruned only if its score decreased during the
  episode. Pruned components stay in the relay plane.
- **Session budgets.** A session's credit is priced in the serialized bytes of a reference peer-to-peer
  protocol (`socket_v1`): control messages, payload and a reserve for the abort or completion messages.
  The number of components is the largest that fits both directions.
- **Selection.** A share of the budget (`exploration_fraction`) goes to the oldest eligible components;
  the rest goes to the highest utility under the selector: `local_update` (default, local-update
  magnitude), `sketch_disagreement`, or `exact_disagreement`, an oracle that reads the partner's tensors
  and only runs in simulation. Ablations: `age_only`, `random`, `shuffled_score`, `full_exchange`,
  `survivor_only`, and `relay_content: scrubbed`.
- **Mixing.** Symmetric pairwise averaging (γ = 1/2) of the complete A and B factors of the chosen
  components, applied atomically on both peers, with optimizer moments reset for mixed active components.

## Status

This is an early release of a method that is still being developed.

- Everything runs in one process. Sessions are simulated: their bytes are priced with the `socket_v1` model
  rather than sent over a network, and the real-device deployment is not included.
- The common initialization needs a local model directory, and copies the `classifier` and `pooler` of
  DeBERTa-style sequence classifiers. Only GLUE classification tasks have been tested.
- Readouts run inside the training loop. They require `training_rng: per_peer_episode`, which reseeds torch
  before every local episode, so that a run with readouts trains exactly as the same run without them.
- A run cannot be resumed: an interrupted run starts again from the first frame.
- The norm-product regularizer of the pruning rule and a convergence analysis are not part of this release.

## Project Structure

```
├── scenarios/                      # example scenarios (MNLI, DeBERTa-large, 10 peers, 120 frames)
├── scripts/
│   ├── make_scenario.py            # write a scenario: method settings and a credit trace
│   └── run_scenario.py             # run a scenario
├── data/                           # GLUE cache (not tracked, see data/README.md)
├── requirements/requirements.txt
└── federatedscope/
    ├── contrib/decentralized/
    │   ├── core/                   # runner, contact traces, pair sessions, graph rounds, byte accounting
    │   ├── lora/                   # LoRA schema, adapter access, common canonical initialization, mixing
    │   ├── methods/                # pulse_lora, dec_lora, tad_lora
    │   ├── simulation/             # scenarios, data and holdout, socket_v1 pricing, the driver
    │   └── tests/
    ├── contrib/trainer/            # the decentralized GLUE trainer
    ├── glue/                       # GLUE data loading and DeBERTa model building
    └── core/                       # FederatedScope configuration, data, model and trainer components
```

## Requirements

Developed with Python 3.10, PyTorch 2.1.0 and CUDA 12.1.

```bash
conda create -n pulse python=3.10 -y && conda activate pulse
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/requirements.txt
pip install -e .
```

## Data and Model Preparation

GLUE is downloaded from the Hugging Face Hub on first use and cached in `data/` (see
[data/README.md](data/README.md)). The model must be a local directory, because the common
initialization hashes the backbone's `config.json` and weight files:

```bash
huggingface-cli download microsoft/deberta-large --local-dir models/deberta-large
```

The canonical initial values depend on the exact set of weight files in that directory.

## Run

```bash
python scripts/run_scenario.py scenarios/pulse_lora_mnli.json --out exp/pulse_lora_mnli --device cuda:0
python scripts/run_scenario.py scenarios/dec_lora_mnli.json --out exp/dec_lora_mnli --device cuda:0
python scripts/run_scenario.py scenarios/tad_lora_mnli.json --out exp/tad_lora_mnli --device cuda:0
```

`--set KEY VALUE` overrides a configuration entry of the scenario, for example
`--set model.type /path/to/deberta-large@huggingface_llm --set data.root /path/to/data/`.

A run trains every peer for `local_steps` steps per frame, runs the frame's sessions or graph round, and
reads every peer out on the scenario's schedule: each peer's logits on a fixed holdout of 2,000 training
examples, removed from every peer's training data. The output directory holds:

| File | Content |
| --- | --- |
| `summary.json` | totals and the accuracy summary |
| `readouts.jsonl` | per readout: ensemble and per-peer accuracy, uploaded bytes, time |
| `sessions.jsonl` | per session or graph-round edge: status, selected components, bytes per direction |
| `peers/peer<p>.json` | per peer: training losses per episode, traffic, method metrics |
| `readout/` | holdout labels and every peer's logits at every readout |
| `runner/` | the runner's event log, manifest and summary |
| `final/peer<p>.pt` | with `--save-state`: every peer's final adapter and method state |

Accuracy is that of the uniform probability ensemble of the peers (the mean of their softmax outputs).
`summary.json` reports the final accuracy and the mean over the last five readouts.

Write other scenarios with `scripts/make_scenario.py`: the GLUE task, model, number of peers and frames,
per-peer ranks, controller, selector, seeds and the contact trace (Erdős–Rényi contacts with edge
probability `--edge-p`, and credits drawn uniformly between 0.6 and 1.8 times a reference credit):

```bash
python scripts/make_scenario.py pulse_lora --task qnli --peers 4 --frames 40 \
    --ranks 64,64,200,200 --controller adas_c --out scenarios/pulse_lora_qnli_small.json
```

Unit tests (CPU):

```bash
pytest federatedscope/contrib/decentralized/tests
```

The tests that build real GLUE data and DeBERTa models run only when told where to find them:

```bash
PULSE_GLUE_DATA_ROOT=data/ PULSE_MODEL_DIR=models/deberta-base \
    pytest federatedscope/contrib/decentralized/tests/simulation/test_readout_real.py
```

## Acknowledgement

This code is built on components of [FederatedScope](https://github.com/alibaba/FederatedScope) and
FederatedScope-LLM (Apache-2.0). See [NOTICE](NOTICE).
