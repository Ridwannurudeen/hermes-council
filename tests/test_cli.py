"""Tests for the hermes-council CLI."""

from pathlib import Path

import pytest

from hermes_council import cli


def test_get_skills_source_exists():
    source = cli._get_skills_source()
    assert source.exists()
    assert source.joinpath("DESCRIPTION.md").exists()


def test_install_skills_copies_bundle(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    cli.install_skills()

    target = tmp_path / ".hermes" / "skills" / "council"
    assert target.joinpath("DESCRIPTION.md").exists()
    assert target.joinpath("multi-perspective-analysis", "SKILL.md").exists()
    assert "Installed council skills" in capsys.readouterr().out


def test_install_skills_refuses_existing_without_force(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    target = tmp_path / ".hermes" / "skills" / "council"
    target.mkdir(parents=True)

    with pytest.raises(SystemExit):
        cli.install_skills()


def test_install_skills_force_overwrites(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    target = tmp_path / ".hermes" / "skills" / "council"
    target.mkdir(parents=True)
    target.joinpath("old.md").write_text("old", encoding="utf-8")

    cli.install_skills(force=True)

    assert not target.joinpath("old.md").exists()
    assert target.joinpath("DESCRIPTION.md").exists()


def test_main_help(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["hermes-council"])
    cli.main()
    assert "install-skills" in capsys.readouterr().out
