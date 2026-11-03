"""Write a scenario document for PULSE-LoRA, Dec-LoRA or TAD-LoRA on a GLUE task.

The defaults are the reference setting: 10 peers, 120 one-second frames of Erdos-Renyi
contacts (p = 0.1) with byte credits, DeBERTa-large on MNLI, 10 local steps per frame, a
2,000-example readout holdout read out every 5 frames, and the common canonical construction.

    python scripts/make_scenario.py pulse_lora --out scenarios/pulse_lora_mnli.json
    python scripts/make_scenario.py dec_lora --out scenarios/dec_lora_mnli.json
    python scripts/make_scenario.py tad_lora --out scenarios/tad_lora_mnli.json
    python scripts/make_scenario.py pulse_lora --task qnli --ranks 64,64,200,200 --peers 4 \\
        --frames 40 --out scenarios/pulse_lora_qnli_small.json

The default reference credit (--b-ref-forward/--b-ref-reverse) is the ``socket_v1`` size of one
complete PULSE-LoRA session at rank 64 on DeBERTa-large/MNLI, per direction; a contact carries
0.6 to 1.8 times that.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federatedscope.contrib.decentralized.simulation.credit_trace import credit_trace  # noqa: E402
from federatedscope.contrib.decentralized.simulation.glue_data import GLUE_BUILDER  # noqa: E402
from federatedscope.contrib.decentralized.simulation.scenario import (  # noqa: E402
    SCENARIO_VERSION,
    save_scenario,
    scenario_from_dict,
)
from federatedscope.contrib.decentralized.simulation.partition import STRATIFIED  # noqa: E402
from federatedscope.contrib.decentralized.simulation.readout import STRATIFIED_HOLDOUT  # noqa: E402
from federatedscope.contrib.decentralized.simulation.training import PER_PEER_EPISODE  # noqa: E402

METHODS = ("pulse_lora", "dec_lora", "tad_lora")
#: Number of labels of each supported GLUE classification task.
TASK_LABELS = {"mnli": 3, "qnli": 2, "rte": 2, "sst2": 2, "mrpc": 2, "qqp": 2, "cola": 2}
TARGET_MODULES = ["attention.self.in_proj", "attention.output.dense", "intermediate.dense",
                  "output.dense"]
#: The reference PULSE-LoRA ranks: two peers at 64, two at 150, three at 120, three at 200.
DEFAULT_RANKS = "64,64,150,150,120,200,200,200,120,120"
DEFAULT_LR = {"pulse_lora": 5e-5, "dec_lora": 2e-4, "tad_lora": 5e-4}
#: Every session's messages must fit this envelope; rank-200 PULSE-LoRA messages exceed the
#: 64 MiB default.
ENVELOPE_BYTES = 512 * 1024 * 1024


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--out", required=True, help="scenario file to write")
    run = parser.add_argument_group("run")
    run.add_argument("--task", default="mnli", choices=sorted(TASK_LABELS))
    run.add_argument("--model", default="models/deberta-large",
                     help="local HuggingFace model directory (config.json and weight files)")
    run.add_argument("--data-root", default="data/")
    run.add_argument("--peers", type=int, default=10)
    run.add_argument("--frames", type=int, default=120)
    run.add_argument("--local-steps", type=int, default=10)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--max-length", type=int, default=128)
    run.add_argument("--lr", type=float, default=None, help="default: per method")
    run.add_argument("--seed", type=int, default=31, help="run seed (training RNG)")
    run.add_argument("--model-init-seed", type=int, default=31001)
    run.add_argument("--partition-seed", type=int, default=31)
    run.add_argument("--holdout", type=int, default=2000, help="readout holdout size")
    run.add_argument("--holdout-seed", type=int, default=39031)
    run.add_argument("--readout-every", type=int, default=5)
    trace = parser.add_argument_group("contact trace")
    trace.add_argument("--trace-seed", type=int, default=21)
    trace.add_argument("--edge-p", type=float, default=0.1)
    trace.add_argument("--b-ref-forward", type=int, default=101661258)
    trace.add_argument("--b-ref-reverse", type=int, default=135554034)
    trace.add_argument("--capacity", type=float, nargs=2, default=(0.6, 1.8), metavar=("LOW", "HIGH"))
    method = parser.add_argument_group("method")
    method.add_argument("--ranks", default=DEFAULT_RANKS,
                        help="PULSE-LoRA: comma-separated initial rank per peer")
    method.add_argument("--r-max", type=int, default=200, help="PULSE-LoRA: canonical rank")
    method.add_argument("--controller", default="fixed", choices=("fixed", "adas_c"),
                        help="PULSE-LoRA: local pruning controller")
    method.add_argument("--prune-gamma", type=float, default=None, help="PULSE-LoRA adas_c: eta")
    method.add_argument("--rank-min", type=int, default=2, help="PULSE-LoRA: rank floor")
    method.add_argument("--selector", default="local_update", help="PULSE-LoRA: utility selector")
    method.add_argument("--exploration", type=float, default=0.25,
                        help="PULSE-LoRA: exploration fraction of each session's slots")
    method.add_argument("--rank", type=int, default=64, help="Dec-LoRA / TAD-LoRA: common rank")
    method.add_argument("--alternation-interval", type=int, default=3, help="TAD-LoRA: T")
    return parser.parse_args(argv)


def lora_args(rank: int):
    return [{"adapter_method": "lora", "adapter_package": "peft", "lora_alpha": 2 * rank,
             "lora_dropout": 0.05, "r": rank, "target_modules": list(TARGET_MODULES)}]


def cfg_overrides(args):
    rank = args.r_max if args.method == "pulse_lora" else args.rank
    overrides = {
        "criterion.type": "CrossEntropyLoss",
        "data.root": args.data_root,
        "data.splits": [0.9, 0.1],
        "data.splitter": "iid",
        "data.type": f"{args.task}@glue",
        "dataloader.batch_size": args.batch_size,
        "dataloader.shuffle": True,
        "decentralized.auxiliary_state.policy": "local",
        "decentralized.comparison.gamma": 0.5,
        "decentralized.comparison.profile": "common_study",
        "decentralized.execution.local_steps": args.local_steps,
        "decentralized.transport.envelope_bytes": ENVELOPE_BYTES,
        "eval.count_flops": False,
        "glue.adapter.args": lora_args(rank),
        "glue.adapter.hetero_strategy": "homo",
        "glue.adapter.max_rank": rank,
        "glue.adapter.use": True,
        "glue.max_length": args.max_length,
        "glue.retry_on_nan_loss": True,
        "glue.task_name": args.task,
        "model.out_channels": TASK_LABELS[args.task],
        "model.task": "SequenceClassification",
        "model.type": f"{args.model.rstrip('/')}@huggingface_llm",
        "train.batch_or_epoch": "batch",
        "train.local_update_steps": args.local_steps,
        "train.optimizer.lr": args.lr if args.lr is not None else DEFAULT_LR[args.method],
        "train.optimizer.type": "AdamW",
        "train.scheduler.type": "LinearLR",
        "trainer.type": "gluedecentralizedtrainer",
        "use_gpu": False,
    }
    if args.method == "pulse_lora":
        overrides.update({
            "decentralized.execution.interaction": "pairwise_session",
            "decentralized.methods.pulse_lora.component_seed": 1234,
            "decentralized.methods.pulse_lora.controller": args.controller,
            "decentralized.methods.pulse_lora.exploration_fraction": args.exploration,
            "decentralized.methods.pulse_lora.initial_rank": args.r_max,
            "decentralized.methods.pulse_lora.r_max": args.r_max,
            "decentralized.methods.pulse_lora.rank_min": args.rank_min,
            "decentralized.methods.pulse_lora.selector": args.selector,
        })
        if args.prune_gamma is not None:
            overrides["decentralized.methods.pulse_lora.prune_gamma"] = args.prune_gamma
    else:
        overrides.update({
            "decentralized.execution.interaction": "graph_round",
            "decentralized.execution.graph_mixing": "dynamic_metropolis",
            "decentralized.execution.skipped_edge_policy": "self_weight",
            f"decentralized.methods.{args.method}.rank": args.rank,
        })
        if args.method == "dec_lora":
            overrides["decentralized.methods.dec_lora.profile"] = "dynamic_graph_adapted"
        else:
            overrides.update({
                "decentralized.methods.tad_lora.profile": "common_matching_trace",
                "decentralized.methods.tad_lora.alternation_interval": args.alternation_interval,
                "decentralized.methods.tad_lora.max_alternation_interval": 5 * args.alternation_interval,
                "decentralized.methods.tad_lora.topology_aware": False,
                "decentralized.methods.tad_lora.start_group": "B",
            })
    return overrides


def peer_overrides(args, peers):
    if args.method != "pulse_lora":
        return {}
    ranks = [int(r) for r in args.ranks.split(",")]
    if len(ranks) != len(peers):
        raise SystemExit(f"--ranks names {len(ranks)} ranks for {len(peers)} peers")
    if any(r < 1 or r > args.r_max for r in ranks):
        raise SystemExit(f"every rank must be in 1..{args.r_max}")
    return {str(p): {"decentralized.methods.pulse_lora.initial_rank": r}
            for p, r in zip(peers, ranks) if r != args.r_max}


def build(args):
    peers = list(range(1, args.peers + 1))
    trace = credit_trace({"forward": args.b_ref_forward, "reverse": args.b_ref_reverse},
                         peers=peers, frames=args.frames, p=args.edge_p,
                         trace_seed=args.trace_seed, capacity_seed=args.trace_seed,
                         low=args.capacity[0], high=args.capacity[1])
    document = {
        "scenario_version": SCENARIO_VERSION,
        "method": args.method,
        "peers": peers,
        "trace": trace,
        "seed": args.seed,
        "cfg_overrides": cfg_overrides(args),
        "peer_overrides": peer_overrides(args, peers),
        "data": GLUE_BUILDER,
        "training_rng": PER_PEER_EPISODE,
        "data_partition": {"splitter": STRATIFIED, "seed": args.partition_seed},
        "readout_schedule": {"every_frames": args.readout_every, "initial": True,
                             "holdout": {"source": "train", "count": args.holdout,
                                         "seed": args.holdout_seed, "rule": STRATIFIED_HOLDOUT}},
        "construction": {"mode": "common_canonical_v1", "model_init_seed": args.model_init_seed},
    }
    return scenario_from_dict(document)


def main(argv=None):
    args = parse_args(argv)
    scenario = build(args)
    directory = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(directory, exist_ok=True)
    save_scenario(scenario, args.out)
    print(f"wrote {args.out} (scenario {scenario.digest[:16]}, trace "
          f"{scenario.trace['generator']['trace_id'][:16]})")


if __name__ == "__main__":
    main()
