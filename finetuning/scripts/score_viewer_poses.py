from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from chemistry import analyze_molecule
from docking import prepare_pdbqt, score_only
from reward import normalize_sa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", type=Path, default=HERE.parent / "viewer")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=HERE.parent / "outputs" / "sufficient_vina" / "eval",
    )
    parser.add_argument("--qvina", type=Path, required=True)
    parser.add_argument("--obabel", type=Path, required=True)
    args = parser.parse_args()

    viewer = args.viewer.resolve()
    manifest_path = viewer / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    work = HERE.parent / "cache" / "dock"
    work.mkdir(parents=True, exist_ok=True)

    for pocket in manifest["pockets"]:
        receptor = viewer / pocket["pdb"]
        receptor_pdbqt = work / f"{pocket['id']}.pdbqt"
        if not receptor_pdbqt.is_file():
            prepare_pdbqt(
                receptor,
                receptor_pdbqt,
                receptor=True,
                obabel_binary=args.obabel,
            )

        for model_name, ligands in pocket["models"].items():
            for index, item in enumerate(ligands):
                ligand_path = viewer / item["sdf"]
                supplier = Chem.SDMolSupplier(str(ligand_path), removeHs=False)
                mol = supplier[0] if supplier and len(supplier) else None
                chem = analyze_molecule(mol, diffsbdd_root=HERE.parents[1] / "model" / "DiffSBDD")
                item["valid"] = bool(chem.valid)
                item["atom_count"] = int(chem.atom_count)
                item["smiles"] = chem.smiles or item.get("smiles")
                item["qed"] = chem.qed
                item["sa"] = normalize_sa(chem.sa)
                item["vina"] = None
                if chem.valid and mol is not None:
                    center = tuple(float(value) for value in rdMolTransforms.ComputeCentroid(mol.GetConformer()))
                    result = score_only(
                        "qvina",
                        receptor_pdbqt,
                        ligand_path,
                        binary=args.qvina,
                        center=center,
                        box_size=20.0,
                        timeout=120.0,
                        obabel_binary=args.obabel,
                        work_dir=work,
                    )
                    item["vina"] = result.score if result.ok else None
                    item["score_failure"] = result.failure
                print(
                    f"{pocket['id']} {model_name} {index + 1}: "
                    f"valid={item['valid']} vina={item['vina']}",
                    flush=True,
                )

    eval_dir = args.eval_dir.resolve()
    for pocket in manifest["pockets"]:
        pocket["heldout"] = {}
        for model_name, filename in (
            ("diffsbdd", "unguided_molecules.json"),
            ("grpo", "ftdiff_molecules.json"),
        ):
            all_rows = json.loads((eval_dir / filename).read_text(encoding="utf-8"))
            rows = [row for row in all_rows if row["pocket_id"] == pocket["pocket_id"]]
            valid = [row for row in rows if row.get("valid")]

            def mean(key: str) -> float | None:
                values = [float(row[key]) for row in valid if row.get(key) is not None]
                return statistics.fmean(values) if values else None

            pocket["heldout"][model_name] = {
                "n": len(rows),
                "validity": sum(bool(row.get("valid")) for row in rows) / max(len(rows), 1),
                "vina": mean("vina"),
                "qed": mean("qed"),
                "sa": mean("sa"),
            }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
