"""Run a scenario and write its results to a new output directory.

    python scripts/run_scenario.py scenarios/pulse_lora_mnli.json --out exp/pulse_lora_mnli --device cuda:0
    python scripts/run_scenario.py scenarios/dec_lora_mnli.json --out exp/dec_lora_mnli \\
        --set model.type /path/to/deberta-large@huggingface_llm --set data.root /path/to/data/

``--set KEY VALUE`` overrides one ``cfg_overrides`` entry of the scenario before it runs (VALUE is
read as JSON when it parses, as a string otherwise); the scenario that ran is written to the output
directory. See ``federatedscope/contrib/decentralized/simulation/driver.py`` for the outputs.
"""
import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federatedscope.contrib.decentralized.simulation.driver import run_scenario  # noqa: E402
from federatedscope.contrib.decentralized.simulation.scenario import load_scenario  # noqa: E402


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scenario", help="scenario JSON file")
    parser.add_argument("--out", required=True, help="output directory (must not exist)")
    parser.add_argument("--device", default="cpu", help='"cpu" or a CUDA device such as "cuda:0"')
    parser.add_argument("--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
                        help="override a cfg_overrides entry of the scenario")
    parser.add_argument("--save-state", action="store_true",
                        help="also write every peer's final state (without optimizer state)")
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    if args.set:
        overrides = dict(scenario.cfg_overrides)
        overrides.update({key: parse_value(value) for key, value in args.set})
        scenario = dataclasses.replace(scenario, cfg_overrides=overrides)
    summary = run_scenario(scenario, args.out, device=args.device, save_state=args.save_state)
    print(json.dumps({k: summary[k] for k in ("method", "frames", "exchanges", "committed",
                                              "upload_bytes_total", "accuracy", "wall_s")
                      if k in summary}, indent=1, default=str))


if __name__ == "__main__":
    main()
