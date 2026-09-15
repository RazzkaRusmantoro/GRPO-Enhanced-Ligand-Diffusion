from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIFF = ROOT / "DiffSBDD"
SHARED_DATA = ROOT.parent / "data"


def main() -> None:
    print("=== DiffSBDD setup check ===")
    print(f"python: {sys.version.split()[0]}  ({sys.executable})")

    ok_clone = (DIFF / "train.py").is_file() and (DIFF / "generate_ligands.py").is_file()
    print(f"DiffSBDD clone: {'OK' if ok_clone else 'MISSING'}  ({DIFF})")

    conda = shutil.which("conda")
    print(f"conda on PATH: {'OK — ' + conda if conda else 'MISSING'}")
    if not conda:
        print("  install Miniconda, then:")
        print("  cd main\\model")
        print("  .\\install_env_windows.ps1")
        print("  conda activate diffsbdd")

    py_ok = sys.version_info[:2] == (3, 10)
    print(f"python 3.10 (DiffSBDD pin): {'OK' if py_ok else 'no — activate diffsbdd env'}")

    try:
        import torch

        print(f"torch: {torch.__version__}  cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"torch: FAIL ({e})")

    for name in ("rdkit", "Bio", "pytorch_lightning"):
        try:
            mod = __import__(name if name != "Bio" else "Bio")
            ver = getattr(mod, "__version__", "?")
            print(f"{name}: {ver}")
        except Exception as e:
            print(f"{name}: FAIL ({e})")

    pocket = SHARED_DATA / "crossdocked_pocket10"
    split = SHARED_DATA / "split_by_name.pt"
    print(f"shared pocket dir: {pocket.is_dir()}  ({pocket})")
    print(f"shared split file: {split.is_file()}  ({split})")

    print()
    if ok_clone and conda and py_ok:
        print("DiffSBDD + diffsbdd env OK")
    elif ok_clone and not conda:
        print("Incomplete — install Miniconda + create diffsbdd env")
    elif ok_clone and not py_ok:
        print("Incomplete — run this script inside `conda activate diffsbdd`")
    else:
        print("Incomplete — clone missing")


if __name__ == "__main__":
    main()
