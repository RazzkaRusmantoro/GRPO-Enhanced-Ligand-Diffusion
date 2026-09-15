from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _pair_ref_sdf(pdb: Path) -> Path | None:
    matches = sorted(pdb.parent.glob(f"{pdb.stem}_*.sdf"))
    return matches[0] if matches else None


def _kmeans(features: np.ndarray, k: int, seed: int, iters: int = 25):
    rng = np.random.default_rng(seed)
    k = min(k, len(features))
    centers = features[rng.choice(len(features), size=k, replace=False)].copy()
    assign = np.zeros(len(features), dtype=np.int64)
    for _ in range(iters):
        dist = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = dist.argmin(axis=1)
        for i in range(k):
            members = features[assign == i]
            if len(members):
                centers[i] = members.mean(axis=0)
    return assign, centers


def _npz_embeddings(npz_path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        names = [str(item) for item in data["names"]]
        one_hot = data["pocket_one_hot"]
        mask = data["pocket_mask"]
    sections = np.where(np.diff(mask))[0] + 1
    chunks = np.split(one_hot, sections)
    rows = []
    keep = []
    for name, chunk in zip(names, chunks):
        if len(chunk) == 0:
            continue
        hist = chunk.mean(axis=0)
        size = np.array([np.log1p(len(chunk))], dtype=np.float64)
        rows.append(np.concatenate([hist.astype(np.float64), size]))
        keep.append(name)
    return keep, np.stack(rows, axis=0)


def pocket_stem_from_npz_name(name: str) -> str:
    text = str(name).replace("\\", "/")
    lower = text.lower()
    if "_pocket10" in lower:
        head = text[: lower.index("_pocket10") + len("_pocket10")]
        leaf = head.split("/")[-1]
        leaf = leaf[:-4] if leaf.lower().endswith(".pdb") else leaf
        return leaf.replace("_", "-")
    leaf = text.split("/")[-1]
    if ".pdb" in leaf.lower():
        leaf = leaf[: leaf.lower().index(".pdb")]
    elif "_" in leaf and leaf.lower().endswith(".sdf"):
        leaf = leaf.split("_")[0]
    return leaf.replace("_", "-")


def _pdb_for_name(processed_dir: Path, split: str, name: str) -> Path | None:
    stem = pocket_stem_from_npz_name(name)
    folder = processed_dir / split
    candidate = folder / f"{stem}.pdb"
    if candidate.is_file():
        return candidate
    matches = sorted(folder.glob(f"{stem}*.pdb"))
    if matches:
        return matches[0]
    code = stem.split("-")[0]
    loose = sorted(folder.glob(f"{code}-*-pocket10.pdb"))
    return loose[0] if len(loose) == 1 else None


def _protein_id(name: str) -> str:
    return pocket_stem_from_npz_name(name).split("-")[0].lower()


def select_pockets(
    processed_dir: str | Path,
    *,
    source_split: str = "val",
    n_ft: int = 8,
    n_test: int = 8,
    kmeans_k: int | None = None,
    seed: int = 17,
    exclude_test: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    test_names, _ = _npz_embeddings(processed_dir / "test.npz")
    test_proteins = {_protein_id(name) for name in test_names}
    source_names, features = _npz_embeddings(processed_dir / f"{source_split}.npz")

    allowed = []
    allowed_feat = []
    for name, feat in zip(source_names, features):
        if exclude_test and _protein_id(name) in test_proteins:
            continue
        pdb = _pdb_for_name(processed_dir, source_split, name)
        if pdb is None or _pair_ref_sdf(pdb) is None:
            continue
        allowed.append(name)
        allowed_feat.append(feat)
    if not allowed:
        raise FileNotFoundError(
            f"No usable {source_split} pockets with PDB+SDF under {processed_dir}"
        )
    feat = np.stack(allowed_feat, axis=0)
    k = min(kmeans_k or n_ft, n_ft, len(allowed))
    assign, centers = _kmeans(feat, k, seed)
    chosen: list[int] = []
    for cluster in range(k):
        members = np.flatnonzero(assign == cluster)
        if len(members) == 0:
            continue
        dist = ((feat[members] - centers[cluster]) ** 2).sum(-1)
        chosen.append(int(members[int(dist.argmin())]))
    leftover = [i for i in range(len(allowed)) if i not in chosen]
    leftover.sort(key=lambda i: float(((feat[i] - centers[assign[i]]) ** 2).sum()))
    for index in leftover:
        if len(chosen) >= n_ft:
            break
        chosen.append(index)

    ft_rows = []
    for rank, index in enumerate(chosen[:n_ft]):
        name = allowed[index]
        pdb = _pdb_for_name(processed_dir, source_split, name)
        ft_rows.append({
            "split": "ft",
            "pocket_idx": rank,
            "pocket_id": pdb.stem,
            "npz_name": name,
            "pdb_path": str(pdb),
            "ref_ligand_sdf": str(_pair_ref_sdf(pdb)),
            "source_split": source_split,
        })

    test_dir = processed_dir / "test"
    test_pdbs = sorted(test_dir.glob("*-pocket10.pdb")) or sorted(test_dir.glob("*.pdb"))
    test_rows = []
    for rank, pdb in enumerate(test_pdbs[:n_test]):
        ref = _pair_ref_sdf(pdb)
        if ref is None:
            continue
        test_rows.append({
            "split": "test",
            "pocket_idx": rank,
            "pocket_id": pdb.stem,
            "npz_name": pdb.stem,
            "pdb_path": str(pdb),
            "ref_ligand_sdf": str(ref),
            "source_split": "test",
        })
    return pd.DataFrame(ft_rows), pd.DataFrame(test_rows)
