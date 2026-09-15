from __future__ import annotations

import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
URL = (
    "https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt"
    "?download=1"
)
OUT = CKPT_DIR / "crossdocked_fullatom_cond.ckpt"


def main() -> None:
    print("=== Download pretrained DiffSBDD checkpoint ===")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT.is_file() and OUT.stat().st_size > 1_000_000:
        print(f"already present: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"fetching {URL}")
    print(f"-> {OUT}")
    urllib.request.urlretrieve(URL, OUT)
    print(f"done: {OUT.stat().st_size / 1e6:.1f} MB")
    print()
    print("Next (conda activate diffsbdd):")
    print("  cd main\\model\\DiffSBDD")
    print(
        "  python generate_ligands.py ..\\checkpoints\\crossdocked_fullatom_cond.ckpt "
        "--pdbfile example\\3rfm.pdb --outfile ..\\outputs\\3rfm_smoke.sdf "
        "--ref_ligand A:330 --n_samples 4 --sanitize"
    )


if __name__ == "__main__":
    main()
