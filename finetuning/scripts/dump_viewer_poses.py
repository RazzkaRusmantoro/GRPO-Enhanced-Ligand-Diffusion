from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from chemistry import analyze_molecule, write_sdf_molecule
from config import load_config, resolve_device
from model import TrainableDiffSBDD
from reward import normalize_sa
from sampler import sample_group

FEATURED = ("1h36", "1ai4", "1phk", "14gs")
KEEP = 3
GROUP = 8


def _seed(tag: str, pocket_id: str, extra: int = 0) -> None:
    value = 17 + extra + (sum(ord(ch) for ch in f"{tag}:{pocket_id}") % 10_000)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _short(pocket_id: str) -> str:
    return pocket_id.split("-")[0]


def _pick_pockets(frame: pd.DataFrame) -> list[dict]:
    chosen = []
    for prefix in FEATURED:
        hits = frame[frame["pocket_id"].str.startswith(f"{prefix}-")]
        if hits.empty:
            hits = frame[frame["pocket_id"].str.startswith(prefix)]
        if hits.empty:
            raise SystemExit(f"Pocket {prefix} not in test set")
        chosen.append(hits.iloc[0].to_dict())
    return chosen


def _collect_valid(wrapper, pocket: dict, cfg: dict, tag: str, diffsbdd_root: str) -> list[dict]:
    sampler = cfg["sampler"]
    n_steps = int(cfg["eval"].get("timesteps", sampler["n_steps"]))
    kept: list[dict] = []
    for attempt in range(2):
        if len(kept) >= KEEP:
            break
        _seed(tag, pocket["pocket_id"], extra=attempt * 97)
        rollout = sample_group(
            wrapper,
            pdb_path=pocket["pdb_path"],
            ref_ligand=pocket["ref_ligand_sdf"],
            group_size=GROUP,
            n_steps=n_steps,
            kind=str(sampler["kind"]),
            interp_gamma=float(sampler["interp_gamma"]),
            interp_eta=float(sampler["interp_eta"]),
            interp_scale=float(sampler["interp_scale"]),
            sanitize=True,
            largest_frag=True,
        )
        for mol in rollout.molecules:
            chem = analyze_molecule(mol, diffsbdd_root=diffsbdd_root)
            if not chem.valid or mol is None:
                continue
            kept.append({
                "mol": mol,
                "qed": chem.qed,
                "sa": normalize_sa(chem.sa),
                "smiles": chem.smiles,
            })
            if len(kept) >= KEEP:
                break
    return kept[:KEEP]


def _sample_model(checkpoint: Path, tag: str, pockets: list[dict], cfg: dict, device: str) -> dict[str, list[dict]]:
    print(f"Loading {tag} <- {checkpoint}", flush=True)
    wrapper = TrainableDiffSBDD(checkpoint, cfg["paths"]["diffsbdd_root"], device=device)
    wrapper.model.eval()
    out: dict[str, list[dict]] = {}
    try:
        for pocket in pockets:
            name = _short(pocket["pocket_id"])
            print(f"  {tag} {name} ...", flush=True)
            out[name] = _collect_valid(wrapper, pocket, cfg, tag, cfg["paths"]["diffsbdd_root"])
            print(f"    kept {len(out[name])} valid", flush=True)
    finally:
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE.parent / "configs" / "sufficient_vina.yaml")
    parser.add_argument("--test-pockets", type=Path)
    parser.add_argument("--ftdiff-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "viewer")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device"))
    test_path = args.test_pockets or (Path(cfg["paths"]["output_dir"]) / "test_pockets.csv")
    ckpt = args.ftdiff_checkpoint or (Path(cfg["paths"]["output_dir"]) / "last.ckpt")
    out = args.output_dir.resolve()
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)

    pockets = _pick_pockets(pd.read_csv(test_path))
    unguided = _sample_model(Path(cfg["paths"]["checkpoint"]), "diffsbdd", pockets, cfg, device)
    tuned = _sample_model(Path(ckpt), "grpo", pockets, cfg, device)

    manifest = {"pockets": []}
    for pocket in pockets:
        name = _short(pocket["pocket_id"])
        pdb_name = f"{name}.pdb"
        shutil.copy2(pocket["pdb_path"], data / pdb_name)
        entry = {
            "id": name,
            "pocket_id": pocket["pocket_id"],
            "pdb": f"data/{pdb_name}",
            "models": {"diffsbdd": [], "grpo": []},
        }
        for tag, bag in (("diffsbdd", unguided), ("grpo", tuned)):
            for idx, item in enumerate(bag.get(name, [])):
                sdf_name = f"{name}_{tag}_{idx}.sdf"
                write_sdf_molecule(item["mol"], data / sdf_name)
                entry["models"][tag].append({
                    "sdf": f"data/{sdf_name}",
                    "qed": item["qed"],
                    "sa": item["sa"],
                    "smiles": item["smiles"],
                })
        manifest["pockets"].append(entry)

    (data / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    print(json.dumps({p["id"]: {k: len(v) for k, v in p["models"].items()} for p in manifest["pockets"]}, indent=2))


if __name__ == "__main__":
    main()
