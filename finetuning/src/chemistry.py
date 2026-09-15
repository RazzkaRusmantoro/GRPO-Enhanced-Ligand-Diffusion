from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ChemistryResult:
    valid: bool
    smiles: str | None = None
    atom_count: int = 0
    qed: float | None = None
    sa: float | None = None
    failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rdkit():
    try:
        from rdkit import Chem
        return Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for ligand processing") from exc


def write_sdf_molecule(mol, path: str | Path) -> None:
    Chem = _rdkit()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(target))
    try:
        writer.write(mol)
    finally:
        writer.close()


def _sa_score(mol, diffsbdd_root: str | Path | None) -> float | None:
    if diffsbdd_root:
        analysis = Path(diffsbdd_root) / "analysis"
        if str(analysis) not in sys.path:
            sys.path.insert(0, str(analysis))
    try:
        from SA_Score.sascorer import calculateScore
        return float(calculateScore(mol))
    except (ImportError, OSError, ValueError):
        return None


def analyze_molecule(mol, diffsbdd_root: str | Path | None = None) -> ChemistryResult:
    if mol is None:
        return ChemistryResult(False, failure="read_failed")
    Chem = _rdkit()
    try:
        candidate = Chem.Mol(mol)
        Chem.SanitizeMol(candidate)
        if candidate.GetNumConformers() == 0:
            raise ValueError("missing_3d_conformer")
        coords = np.asarray(candidate.GetConformer().GetPositions(), dtype=float)
        if coords.shape != (candidate.GetNumAtoms(), 3) or not np.isfinite(coords).all():
            raise ValueError("invalid_coordinates")
        fragments = Chem.GetMolFrags(candidate, asMols=True, sanitizeFrags=True)
        if fragments:
            candidate = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
        from rdkit.Chem import QED
        smiles = Chem.MolToSmiles(Chem.RemoveHs(candidate), canonical=True)
        return ChemistryResult(
            True,
            smiles=smiles,
            atom_count=int(candidate.GetNumHeavyAtoms()),
            qed=float(QED.qed(candidate)),
            sa=_sa_score(candidate, diffsbdd_root),
        )
    except Exception as exc:
        return ChemistryResult(False, failure=f"{type(exc).__name__}: {exc}")
