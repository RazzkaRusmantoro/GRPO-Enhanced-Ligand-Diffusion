from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
MAIN = PROJECT.parent


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    paths = cfg.setdefault("paths", {})
    for key, default in (
        ("checkpoint", MAIN / "model" / "checkpoints" / "crossdocked_fullatom_cond.ckpt"),
        ("diffsbdd_root", MAIN / "model" / "DiffSBDD"),
        ("processed_dir", MAIN / "data" / "processed_crossdock_noH_full_temp"),
        ("output_dir", PROJECT / "outputs" / "pilot"),
    ):
        raw = Path(paths.get(key, default))
        paths[key] = str(raw if raw.is_absolute() else (PROJECT / raw).resolve())
    return cfg


def resolve_device(name: str | None) -> str:
    if name and name != "auto":
        return name
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
