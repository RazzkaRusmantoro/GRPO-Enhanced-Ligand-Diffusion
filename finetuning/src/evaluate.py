from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from decode import score_molecules
from sampler import sample_group
from train_loop import _reward_kwargs


def _fingerprints(molecules):
    from rdkit.Chem import AllChem, DataStructs
    fps = []
    for mol in molecules:
        if mol is None:
            continue
        try:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        except Exception:
            continue
    return fps


def pairwise_diversity(molecules) -> float:
    from rdkit.Chem import DataStructs
    fps = _fingerprints(molecules)
    if len(fps) < 2:
        return float("nan")
    dists = []
    for i, left in enumerate(fps):
        for right in fps[i + 1:]:
            dists.append(1.0 - float(DataStructs.TanimotoSimilarity(left, right)))
    return float(np.mean(dists)) if dists else float("nan")


def evaluate_checkpoint(
    wrapper,
    test_rows,
    cfg: dict,
    *,
    qvina_binary,
    obabel_binary,
    output_dir: Path,
    tag: str,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler_cfg = cfg["sampler"]
    dock_cfg = cfg["docking"]
    n_samples = int(cfg["eval"]["n_samples"])
    rows = []
    all_mols = []
    for pocket in test_rows.to_dict(orient="records"):
        rollout = sample_group(
            wrapper,
            pdb_path=pocket["pdb_path"],
            ref_ligand=pocket["ref_ligand_sdf"],
            group_size=n_samples,
            n_steps=int(cfg["eval"].get("timesteps", sampler_cfg["n_steps"])),
            kind=str(sampler_cfg["kind"]),
            interp_gamma=float(sampler_cfg["interp_gamma"]),
            interp_eta=float(sampler_cfg["interp_eta"]),
            interp_scale=float(sampler_cfg["interp_scale"]),
            sanitize=True,
            largest_frag=True,
        )
        breakdowns = score_molecules(
            rollout.molecules,
            pdb_path=pocket["pdb_path"],
            diffsbdd_root=cfg["paths"]["diffsbdd_root"],
            engine=dock_cfg["engine"],
            binary=qvina_binary,
            obabel=obabel_binary,
            box_size=float(dock_cfg["box_size"]),
            timeout=float(dock_cfg["timeout"]),
            reward_cfg=_reward_kwargs(cfg),
        )
        all_mols.extend(rollout.molecules)
        for mol_idx, item in enumerate(breakdowns):
            rows.append({
                "pocket_id": pocket["pocket_id"],
                "mol_idx": mol_idx,
                "valid": item.valid,
                "connected": item.connected,
                "reward": item.reward,
                "vina": item.vina,
                "s_vina": item.s_vina,
                "qed": item.s_qed,
                "sa": item.s_sa,
                "failure": item.failure,
            })

    valid_rows = [r for r in rows if r["valid"] and r["vina"] is not None]
    summary = {
        "tag": tag,
        "n_molecules": len(rows),
        "validity": sum(r["valid"] for r in rows) / max(len(rows), 1),
        "connected": sum(r["connected"] for r in rows) / max(len(rows), 1),
        "mean_reward": float(np.mean([r["reward"] for r in rows])) if rows else None,
        "mean_vina": float(np.mean([r["vina"] for r in valid_rows])) if valid_rows else None,
        "mean_qed": float(np.mean([r["qed"] for r in valid_rows])) if valid_rows else None,
        "mean_sa": float(np.mean([r["sa"] for r in valid_rows])) if valid_rows else None,
        "diversity": pairwise_diversity(all_mols),
    }
    (output_dir / f"{tag}_molecules.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / f"{tag}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary
