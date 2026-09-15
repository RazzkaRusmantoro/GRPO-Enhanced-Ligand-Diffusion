from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

SUPPORTED_ENGINES = {"vina", "smina", "qvina"}


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class DockingResult:
    ok: bool
    score: float | None
    engine: str
    mode: str = "score_only"
    command: list[str] | None = None
    elapsed_sec: float = 0.0
    failure: str | None = None
    stdout: str = ""
    receptor_sha256: str | None = None
    ligand_sha256: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_engine(engine: str, binary: str | Path | None = None) -> Path:
    engine = engine.lower()
    if engine == "auto":
        raise ValueError("Training labels require an explicit docking engine, not 'auto'")
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(f"Unsupported docking engine {engine!r}")
    if binary:
        result = Path(binary).expanduser()
        if result.is_file():
            return result.resolve()
        found = shutil.which(str(binary))
    else:
        names = {
            "vina": ("vina", "vina.exe"),
            "smina": ("smina", "smina.static", "smina.exe"),
            "qvina": ("qvina2.1", "qvina2", "qvina", "qvina2.exe", "qvina2.1.exe"),
        }[engine]
        found = next((shutil.which(name) for name in names if shutil.which(name)), None)
    if not found:
        raise FileNotFoundError(f"No {engine} executable found; pass --binary")
    return Path(found).resolve()


def parse_score(output: str) -> float:
    patterns = (
        r"Affinity:\s*([+-]?\d+(?:\.\d+)?)",
        r"Estimated Free Energy of Binding\s*[:=]\s*([+-]?\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    table = re.search(
        r"-{3,}\+[-+]+\n\s*1\s+([+-]?\d+(?:\.\d+)?)", output, flags=re.MULTILINE
    )
    if table:
        return float(table.group(1))
    raise ValueError("Affinity not found in docking output")


def _run(command: Sequence[str], timeout: float) -> tuple[int, str, float]:
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    return completed.returncode, completed.stdout, time.monotonic() - started


def prepare_pdbqt(
    source: str | Path,
    target: str | Path,
    *,
    receptor: bool,
    obabel_binary: str | Path = "obabel",
    timeout: float = 120,
) -> None:
    found = shutil.which(str(obabel_binary)) or (
        str(obabel_binary) if Path(obabel_binary).is_file() else None
    )
    if not found:
        raise FileNotFoundError("Open Babel executable not found")
    command = [found, str(source), "-O", str(target)]
    command += ["-xr"] if receptor else ["-h"]
    code, output, _ = _run(command, timeout)
    if code or not Path(target).is_file():
        raise RuntimeError(f"Open Babel conversion failed ({code}): {output[-500:]}")
    keep = (
        ("ATOM", "HETATM", "TER", "END")
        if receptor
        else (
            "ATOM", "HETATM", "TER", "END", "ROOT", "ENDROOT",
            "BRANCH", "ENDBRANCH", "TORSDOF", "REMARK",
        )
    )
    lines = Path(target).read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines()
    cleaned = [line for line in lines if line.startswith(keep)]
    if not any(line.startswith(("ATOM", "HETATM")) for line in cleaned):
        raise RuntimeError("PDBQT conversion produced no atom records")
    Path(target).write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def score_only(
    engine: str,
    receptor: str | Path,
    ligand: str | Path,
    *,
    binary: str | Path | None = None,
    center: tuple[float, float, float] | None = None,
    box_size: float = 20.0,
    timeout: float = 300.0,
    obabel_binary: str | Path = "obabel",
    work_dir: str | Path | None = None,
) -> DockingResult:
    engine = engine.lower()
    receptor, ligand = Path(receptor).resolve(), Path(ligand).resolve()
    command: list[str] | None = None
    started = time.monotonic()
    result = DockingResult(
        False, None, engine, receptor_sha256=file_sha256(receptor),
        ligand_sha256=file_sha256(ligand),
    )
    try:
        executable = resolve_engine(engine, binary)
        if engine == "smina":
            command = [
                str(executable), "-r", str(receptor), "-l", str(ligand), "--score_only"
            ]
            code, output, elapsed = _run(command, timeout)
        else:
            if center is None:
                raise ValueError("vina/qvina score-only requires an explicit box center")
            base = Path(work_dir) if work_dir else None
            with tempfile.TemporaryDirectory(dir=base) as temporary:
                temporary = Path(temporary)
                rec_pdbqt = receptor if receptor.suffix.lower() == ".pdbqt" else temporary / "receptor.pdbqt"
                lig_pdbqt = ligand if ligand.suffix.lower() == ".pdbqt" else temporary / "ligand.pdbqt"
                if rec_pdbqt != receptor:
                    prepare_pdbqt(receptor, rec_pdbqt, receptor=True, obabel_binary=obabel_binary)
                if lig_pdbqt != ligand:
                    prepare_pdbqt(ligand, lig_pdbqt, receptor=False, obabel_binary=obabel_binary)
                x, y, z = center
                command = [
                    str(executable), "--receptor", str(rec_pdbqt), "--ligand", str(lig_pdbqt),
                    "--center_x", str(x), "--center_y", str(y), "--center_z", str(z),
                    "--size_x", str(box_size), "--size_y", str(box_size),
                    "--size_z", str(box_size), "--score_only",
                ]
                code, output, elapsed = _run(command, timeout)
        result.command, result.stdout, result.elapsed_sec = command, output[-4000:], elapsed
        if code:
            result.failure = f"engine_exit_{code}"
        else:
            result.score = parse_score(output)
            result.ok = True
    except subprocess.TimeoutExpired:
        result.command, result.failure = command, "timeout"
    except Exception as exc:
        result.command, result.failure = command, f"{type(exc).__name__}: {exc}"
    result.elapsed_sec = result.elapsed_sec or (time.monotonic() - started)
    return result
