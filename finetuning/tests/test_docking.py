import pytest

import docking


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Affinity: -7.25 (kcal/mol)", -7.25),
        ("Estimated Free Energy of Binding = -6.1", -6.1),
        ("-----+------------+----------+----------\n   1       -8.4      0.0      0.0", -8.4),
    ],
)
def test_parse_score(text, expected):
    assert docking.parse_score(text) == expected


def test_explicit_engine_rejects_auto():
    with pytest.raises(ValueError, match="explicit"):
        docking.resolve_engine("auto")


def test_smina_command_is_score_only(tmp_path, monkeypatch):
    receptor = tmp_path / "rec.pdb"
    ligand = tmp_path / "lig.sdf"
    binary = tmp_path / "smina.exe"
    for path in (receptor, ligand, binary):
        path.write_text("fixture", encoding="utf-8")

    observed = {}

    def fake_run(command, timeout):
        observed["command"] = command
        return 0, "Affinity: -5.5 (kcal/mol)", 0.1

    monkeypatch.setattr(docking, "_run", fake_run)
    result = docking.score_only("smina", receptor, ligand, binary=binary)
    assert result.ok and result.score == -5.5
    assert "--score_only" in observed["command"]
    assert "--exhaustiveness" not in observed["command"]
