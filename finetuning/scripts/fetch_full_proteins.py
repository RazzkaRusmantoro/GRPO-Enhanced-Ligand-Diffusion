from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RCSB = "https://files.rcsb.org/download/{code}.pdb"


def alpha_carbons(text: str, chain: str | None = None) -> dict[int, tuple[str, np.ndarray]]:
    found: dict[int, tuple[str, np.ndarray]] = {}
    for line in text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if chain and line[21] != chain:
            continue
        found[int(line[22:26])] = (
            line[17:20].strip(),
            np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
        )
    return found


def chains_in(text: str) -> list[str]:
    return sorted({line[21] for line in text.splitlines() if line.startswith("ATOM")})


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign])
    rotation = vt.T @ correction @ u.T
    return rotation, target_center - rotation @ mobile_center


def fit_chain(
    deposited: str, chain: str, pocket_ca: dict[int, tuple[str, np.ndarray]]
) -> tuple[float, int, np.ndarray, np.ndarray] | None:
    chain_ca = alpha_carbons(deposited, chain)
    shared = [
        i for i in sorted(set(chain_ca) & set(pocket_ca)) if chain_ca[i][0] == pocket_ca[i][0]
    ]
    if len(shared) < 4:
        return None
    mobile = np.array([chain_ca[i][1] for i in shared])
    target = np.array([pocket_ca[i][1] for i in shared])
    rotation, shift = kabsch(mobile, target)
    deviation = np.linalg.norm((rotation @ mobile.T).T + shift - target, axis=1)
    keep = deviation < max(2.0, float(np.median(deviation)) * 3.0)
    if keep.sum() >= 4 and not keep.all():
        rotation, shift = kabsch(mobile[keep], target[keep])
        deviation = np.linalg.norm((rotation @ mobile.T).T + shift - target, axis=1)
    core = float(np.sqrt((deviation[keep] ** 2).mean()))
    return core, int(keep.sum()), rotation, shift


def transform_atoms(text: str, rotation: np.ndarray, shift: np.ndarray) -> str:
    out = []
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        point = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        moved = rotation @ point + shift
        out.append(f"{line[:30]}{moved[0]:8.3f}{moved[1]:8.3f}{moved[2]:8.3f}{line[54:]}")
    out.append("END")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", type=Path, default=HERE.parent / "viewer")
    args = parser.parse_args()

    data = args.viewer.resolve() / "data"
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    downloads = HERE.parent / "cache" / "rcsb"
    downloads.mkdir(parents=True, exist_ok=True)

    for pocket in manifest["pockets"]:
        code = pocket["pocket_id"].split("-")[0]
        cache = downloads / f"{code}.pdb"
        if not cache.is_file():
            print(f"downloading {code} ...", flush=True)
            with urllib.request.urlopen(RCSB.format(code=code), timeout=90) as handle:
                cache.write_bytes(handle.read())

        deposited = cache.read_text(encoding="utf-8", errors="ignore")
        pocket_text = (args.viewer / pocket["pdb"]).read_text(encoding="utf-8", errors="ignore")
        pocket_ca = alpha_carbons(pocket_text)

        candidates = []
        for chain in chains_in(deposited):
            fit = fit_chain(deposited, chain, pocket_ca)
            if fit:
                candidates.append((fit, chain))
        candidates.sort(key=lambda item: (-item[0][1], item[0][0]))
        best, best_chain = candidates[0] if candidates else (None, "")
        if best is None:
            print(f"{pocket['id']}: no chain matched the pocket, skipping", flush=True)
            continue

        rmsd, shared, rotation, shift = best
        aligned_name = f"{pocket['id']}_full.pdb"
        (data / aligned_name).write_text(
            transform_atoms(deposited, rotation, shift), encoding="utf-8"
        )
        pocket["pdb_full"] = f"data/{aligned_name}"
        pocket["full_pdb_id"] = code.upper()
        pocket["full_chain"] = best_chain
        pocket["full_fit_rmsd"] = round(rmsd, 3)
        pocket["full_residues"] = len(alpha_carbons(deposited, best_chain))
        print(
            f"{pocket['id']}: chain {best_chain}, {pocket['full_residues']} residues, "
            f"{shared} shared CA, RMSD {rmsd:.2f} A",
            flush=True,
        )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
