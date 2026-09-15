from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value, digits=3):
    if value is None:
        return "missing"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads((args.eval_dir / "comparison.json").read_text(encoding="utf-8"))
    base = comparison.get("unguided") or {}
    tuned = comparison.get("ftdiff") or {}
    delta = None
    if base.get("mean_vina") is not None and tuned.get("mean_vina") is not None:
        delta = tuned["mean_vina"] - base["mean_vina"]
    improved = delta is not None and delta < 0
    lines = [
        "# FTDiff status",
        "",
        "Independent QuickVina / QED / SA on held-out test pockets.",
        "Do not treat training-reward improvement as success by itself.",
        "",
        "| Metric | Unguided DiffSBDD | FTDiff |",
        "|---|---:|---:|",
        f"| Mean Vina | {fmt(base.get('mean_vina'))} | {fmt(tuned.get('mean_vina'))} |",
        f"| Validity | {fmt(100 * float(base.get('validity') or 0), 1)}% | {fmt(100 * float(tuned.get('validity') or 0), 1)}% |",
        f"| QED | {fmt(base.get('mean_qed'))} | {fmt(tuned.get('mean_qed'))} |",
        f"| SA | {fmt(base.get('mean_sa'))} | {fmt(tuned.get('mean_sa'))} |",
        f"| Diversity | {fmt(base.get('diversity'))} | {fmt(tuned.get('diversity'))} |",
        f"| Mean reward | {fmt(base.get('mean_reward'))} | {fmt(tuned.get('mean_reward'))} |",
        "",
        f"Paired mean Vina delta (FTDiff - unguided): {fmt(delta)}. "
        "Negative is better.",
        "",
        ("Independent Vina improved." if improved
         else "Independent Vina did not improve. Do not claim FTDiff success."),
        "",
    ]
    (args.eval_dir / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print((args.eval_dir / "STATUS.md").read_text(encoding="utf-8"))
    raise SystemExit(0 if improved else 2)


if __name__ == "__main__":
    main()
