from __future__ import annotations

import sys
from pathlib import Path

import torch

from reward import RewardBreakdown, threshold_aware_reward


def _ensure_diffsbdd_path(diffsbdd_root: Path) -> None:
    if str(diffsbdd_root) not in sys.path:
        sys.path.insert(0, str(diffsbdd_root))


def latents_to_molecules(model, xh_lig, lig_mask, *, sanitize: bool, largest_frag: bool):
    from analysis.molecule_builder import build_molecule, process_molecule
    import utils

    x = xh_lig[:, :model.n_dims].detach().cpu()
    atom_type = xh_lig[:, model.n_dims:].argmax(1).detach().cpu()
    lig_mask = lig_mask.detach().cpu()
    molecules = []
    for coords, types in zip(utils.batch_to_list(x, lig_mask),
                             utils.batch_to_list(atom_type, lig_mask)):
        try:
            mol = build_molecule(
                coords, types, model.model.dataset_info, add_coords=True)
            mol = process_molecule(
                mol, add_hydrogens=False, sanitize=sanitize,
                relax_iter=0, largest_frag=largest_frag)
        except Exception:
            mol = None
        molecules.append(mol)
    return molecules


def _is_connected(mol) -> bool:
    if mol is None:
        return False
    from rdkit import Chem
    try:
        return len(Chem.GetMolFrags(mol, asMols=False)) == 1
    except Exception:
        return False


def score_molecules(
    molecules,
    *,
    pdb_path: str | Path,
    diffsbdd_root: str | Path,
    engine: str,
    binary: str | Path | None,
    obabel: str | Path,
    box_size: float,
    timeout: float,
    reward_cfg: dict,
    receptor_pdbqt: Path | None = None,
) -> list[RewardBreakdown]:
    _ensure_diffsbdd_path(Path(diffsbdd_root))
    from chemistry import analyze_molecule, write_sdf_molecule
    from docking import prepare_pdbqt, score_only
    from rdkit.Chem import rdMolTransforms

    pdb_path = Path(pdb_path)
    work = Path(pdb_path).parent / "_ftdiff_dock"
    work.mkdir(parents=True, exist_ok=True)
    rec = receptor_pdbqt
    if rec is None or not rec.is_file():
        rec = work / f"{pdb_path.stem}.pdbqt"
        if not rec.is_file():
            prepare_pdbqt(pdb_path, rec, receptor=True, obabel_binary=obabel)

    results: list[RewardBreakdown] = []
    for index, mol in enumerate(molecules):
        chem = analyze_molecule(mol, diffsbdd_root=diffsbdd_root)
        connected = _is_connected(mol) if mol is not None else False
        vina = None
        if chem.valid and mol is not None:
            ligand_sdf = work / f"lig_{index}.sdf"
            write_sdf_molecule(mol, ligand_sdf)
            conf = mol.GetConformer()
            center = tuple(float(x) for x in rdMolTransforms.ComputeCentroid(conf))
            docked = score_only(
                engine, rec, ligand_sdf, binary=binary, center=center,
                box_size=box_size, timeout=timeout, obabel_binary=obabel,
                work_dir=work,
            )
            vina = docked.score if docked.ok else None
        results.append(threshold_aware_reward(
            vina=vina, qed=chem.qed, sa=chem.sa, valid=chem.valid,
            connected=connected, **reward_cfg,
        ))
    return results
