from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from config import load_config, resolve_device
from evaluate import evaluate_checkpoint
from model import TrainableDiffSBDD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-pockets", type=Path, required=True)
    parser.add_argument("--ftdiff-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--qvina-binary")
    parser.add_argument("--obabel-binary")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = (args.output_dir or Path(cfg["paths"]["output_dir"]) / "eval").resolve()
    device = resolve_device(cfg.get("device"))
    test = pd.read_csv(args.test_pockets)
    summaries = {}
    if not args.skip_baseline:
        baseline = TrainableDiffSBDD(
            cfg["paths"]["checkpoint"], cfg["paths"]["diffsbdd_root"], device=device)
        baseline.model.eval()
        summaries["unguided"] = evaluate_checkpoint(
            baseline, test, cfg,
            qvina_binary=args.qvina_binary, obabel_binary=args.obabel_binary,
            output_dir=out, tag="unguided")
        del baseline
    tuned = TrainableDiffSBDD(
        args.ftdiff_checkpoint, cfg["paths"]["diffsbdd_root"], device=device)
    tuned.model.eval()
    summaries["ftdiff"] = evaluate_checkpoint(
        tuned, test, cfg,
        qvina_binary=args.qvina_binary, obabel_binary=args.obabel_binary,
        output_dir=out, tag="ftdiff")
    (out / "comparison.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
