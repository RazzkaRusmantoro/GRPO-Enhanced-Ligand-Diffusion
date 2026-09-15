from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import torch


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrainableDiffSBDD:
    def __init__(
        self,
        checkpoint: str | Path,
        diffsbdd_root: str | Path,
        *,
        device: str | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        self.root = Path(diffsbdd_root).resolve()
        if not (self.root / "lightning_modules.py").is_file():
            raise FileNotFoundError(f"Not a DiffSBDD source directory: {self.root}")
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        module = importlib.import_module("lightning_modules")
        model_type = getattr(module, "LigandPocketDDPM")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_type.load_from_checkpoint(
            str(self.checkpoint), map_location=self.device
        ).to(self.device)
        self.model.train()
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.ddpm = self.model.ddpm
        self.provenance = {
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": checkpoint_sha256(self.checkpoint),
            "diffsbdd_root": str(self.root),
            "device": self.device,
        }

    @property
    def n_dims(self) -> int:
        return int(self.ddpm.n_dims)

    def trainable_parameters(self):
        return [p for p in self.ddpm.parameters() if p.requires_grad]
