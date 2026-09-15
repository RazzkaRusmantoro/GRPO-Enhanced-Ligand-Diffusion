from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from config import load_config, resolve_device
from model import TrainableDiffSBDD
from train_loop import train_ftdiff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ft-pockets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--qvina-binary")
    parser.add_argument("--obabel-binary")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = (args.output_dir or Path(cfg["paths"]["output_dir"])).resolve()
    device = resolve_device(cfg.get("device"))
    wrapper = TrainableDiffSBDD(
        cfg["paths"]["checkpoint"], cfg["paths"]["diffsbdd_root"], device=device)
    ft = pd.read_csv(args.ft_pockets)
    train_ftdiff(
        wrapper, ft, cfg,
        qvina_binary=args.qvina_binary,
        obabel_binary=args.obabel_binary,
        output_dir=out,
    )


if __name__ == "__main__":
    main()
