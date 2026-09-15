from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from config import load_config
from pockets import select_pockets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = (args.output_dir or Path(cfg["paths"]["output_dir"])).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pockets = cfg["pockets"]
    ft, test = select_pockets(
        cfg["paths"]["processed_dir"],
        source_split=pockets["source_split"],
        n_ft=int(pockets["n_ft_pockets"]),
        n_test=int(pockets["n_test_pockets"]),
        kmeans_k=int(pockets.get("kmeans_k") or pockets["n_ft_pockets"]),
        seed=int(cfg.get("seed", 17)),
        exclude_test=bool(pockets.get("exclude_test", True)),
    )
    ft.to_csv(out / "ft_pockets.csv", index=False)
    test.to_csv(out / "test_pockets.csv", index=False)
    print(f"FT pockets: {len(ft)} -> {out / 'ft_pockets.csv'}")
    print(f"Test pockets: {len(test)} -> {out / 'test_pockets.csv'}")


if __name__ == "__main__":
    main()
