from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DIFF = Path(__file__).resolve().parent / "DiffSBDD"
sys.path.insert(0, str(DIFF))

from constants import dataset_params
from process_crossdock import (
    compute_smiles,
    get_bond_length_arrays,
    get_lennard_jones_rm,
    get_n_nodes,
    get_type_histograms,
)


PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed_crossdock_noH_full_temp"


def main() -> None:
    processed_dir = PROCESSED
    if not (processed_dir / "train.npz").is_file():
        raise FileNotFoundError(f"Missing {processed_dir / 'train.npz'}")

    dataset_info = dataset_params["crossdock_full"]
    atom_dict = dataset_info["atom_encoder"]
    amino_acid_dict = dataset_info["aa_encoder"]

    import process_crossdock as pc

    pc.dataset_info = dataset_info

    print(f"Loading {processed_dir / 'train.npz'} ...")
    with np.load(processed_dir / "train.npz", allow_pickle=True) as data:
        lig_mask = data["lig_mask"]
        pocket_mask = data["pocket_mask"]
        lig_coords = data["lig_coords"]
        lig_one_hot = data["lig_one_hot"]
        pocket_one_hot = data["pocket_one_hot"]
        n_train = int(len(np.unique(lig_mask)))

    n_val = n_test = "?"
    if (processed_dir / "val.npz").is_file():
        with np.load(processed_dir / "val.npz", allow_pickle=True) as data:
            n_val = int(len(np.unique(data["lig_mask"])))
    if (processed_dir / "test.npz").is_file():
        with np.load(processed_dir / "test.npz", allow_pickle=True) as data:
            n_test = int(len(np.unique(data["lig_mask"])))

    print(f"samples train/val/test (after): {n_train} / {n_val} / {n_test}")
    print("Computing SMILES (slow) ...")
    train_smiles = compute_smiles(lig_coords, lig_one_hot, lig_mask)
    np.save(processed_dir / "train_smiles.npy", train_smiles)
    print(f"saved train_smiles.npy ({len(train_smiles)} smiles)")

    n_nodes = get_n_nodes(lig_mask, pocket_mask, smooth_sigma=1.0)
    np.save(processed_dir / "size_distribution.npy", n_nodes)
    print("saved size_distribution.npy")

    bonds1, bonds2, bonds3 = get_bond_length_arrays(atom_dict)
    rm_LJ = get_lennard_jones_rm(atom_dict)
    atom_hist, aa_hist = get_type_histograms(
        lig_one_hot, pocket_one_hot, atom_dict, amino_acid_dict
    )

    summary_string = "# SUMMARY\n\n"
    summary_string += "# After processing (finish script)\n"
    summary_string += f"num_samples train: {n_train}\n"
    summary_string += f"num_samples val: {n_val}\n"
    summary_string += f"num_samples test: {n_test}\n\n"
    summary_string += "# Info\n"
    summary_string += f"'atom_encoder': {atom_dict}\n"
    summary_string += f"'atom_decoder': {list(atom_dict.keys())}\n"
    summary_string += f"'aa_encoder': {amino_acid_dict}\n"
    summary_string += f"'aa_decoder': {list(amino_acid_dict.keys())}\n"
    summary_string += f"'bonds1': {bonds1.tolist()}\n"
    summary_string += f"'bonds2': {bonds2.tolist()}\n"
    summary_string += f"'bonds3': {bonds3.tolist()}\n"
    summary_string += f"'lennard_jones_rm': {rm_LJ.tolist()}\n"
    summary_string += f"'atom_hist': {atom_hist}\n"
    summary_string += f"'aa_hist': {aa_hist}\n"
    summary_string += f"'n_nodes': {n_nodes.tolist()}\n"

    with open(processed_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_string)
    print(summary_string)
    print(f"Checkpoint E extras done → {processed_dir}")


if __name__ == "__main__":
    main()
