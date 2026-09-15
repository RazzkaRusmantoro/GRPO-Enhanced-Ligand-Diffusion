import numpy as np

from pockets import _kmeans, _protein_id, pocket_stem_from_npz_name, select_pockets


def test_protein_id_strips_chain_and_ligand():
    assert _protein_id("14gs-A-rec-20gs-cbd-lig-tt-min-0-pocket10") == "14gs"


def test_npz_name_maps_to_hyphen_pocket_stem():
    name = (
        "GSTP1_HUMAN_2_210_0/14gs_A_rec_20gs_cbd_lig_tt_min_0_pocket10.pdb_"
        "GSTP1_HUMAN_2_210_0/14gs_A_rec_20gs_cbd_lig_tt_min_0.sdf"
    )
    assert pocket_stem_from_npz_name(name) == (
        "14gs-A-rec-20gs-cbd-lig-tt-min-0-pocket10"
    )
    assert _protein_id(name) == "14gs"


def test_kmeans_separates_two_blobs():
    left = np.zeros((6, 2))
    right = np.ones((6, 2)) * 10
    feat = np.concatenate([left, right], axis=0)
    assign, centers = _kmeans(feat, k=2, seed=0)
    assert len(set(assign.tolist())) == 2
    assert np.linalg.norm(centers[0] - centers[1]) > 5


def test_select_pockets_holds_out_test_proteins(tmp_path):
    def write_npz(path, names, shift):
        n = len(names)
        mask = np.repeat(np.arange(n), 2)
        one_hot = np.zeros((n * 2, 4), dtype=np.float32)
        one_hot[np.arange(n * 2), (np.arange(n * 2) + shift) % 4] = 1
        np.savez(path, names=np.array(names), pocket_mask=mask, pocket_one_hot=one_hot)

    processed = tmp_path / "processed"
    for split in ("train", "val", "test"):
        (processed / split).mkdir(parents=True)
    write_npz(processed / "val.npz", ["aaaa-A-rec", "bbbb-A-rec", "cccc-A-rec"], 0)
    write_npz(processed / "test.npz", ["aaaa-A-rec-other"], 1)
    for split, stems in (("val", ["aaaa-A-rec", "bbbb-A-rec", "cccc-A-rec"]),
                         ("test", ["aaaa-A-rec-other"])):
        for stem in stems:
            pdb = processed / split / f"{stem}.pdb"
            sdf = processed / split / f"{stem}_{stem}.sdf"
            pdb.write_text("HEADER\nEND\n", encoding="utf-8")
            sdf.write_text("ligand\n", encoding="utf-8")
    ft, test = select_pockets(
        processed, source_split="val", n_ft=2, n_test=1, kmeans_k=2, seed=0)
    assert "aaaa" not in set(ft["pocket_id"].str[:4])
    assert len(test) == 1
