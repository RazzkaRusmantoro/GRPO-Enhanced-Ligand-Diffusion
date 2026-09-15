from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
SRC = DATA / "processed_crossdock_noH_full_temp"
DST = DATA / "processed_smoke_from_val"


def main() -> None:
    if not (SRC / "val.npz").is_file():
        raise FileNotFoundError(SRC / "val.npz")

    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / "val.npz", DST / "train.npz")
    shutil.copy2(SRC / "val.npz", DST / "val.npz")
    if (SRC / "test.npz").is_file():
        shutil.copy2(SRC / "test.npz", DST / "test.npz")

    with np.load(DST / "train.npz", allow_pickle=True) as data:
        lig_mask = data["lig_mask"]
        pocket_mask = data["pocket_mask"]
        n_lig = np.unique(lig_mask, return_counts=True)[1]
        n_poc = np.unique(pocket_mask, return_counts=True)[1]
        joint = np.zeros((int(n_lig.max()) + 1, int(n_poc.max()) + 1), dtype=np.float64)
        for a, b in zip(n_lig, n_poc):
            joint[a, b] += 1.0
    np.save(DST / "size_distribution.npy", joint)

    smiles_src = SRC / "train_smiles.npy"
    if smiles_src.is_file():
        smiles = np.load(smiles_src, allow_pickle=True)[:500]
        np.save(DST / "train_smiles.npy", smiles)
    else:
        np.save(DST / "train_smiles.npy", np.array([], dtype=object))

    print(f"Smoke processed data → {DST}")
    print("Train config: configs/pcmg_crossdock_fullatom_cond_smoke.yml")


if __name__ == "__main__":
    main()
