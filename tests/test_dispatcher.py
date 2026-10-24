"""Focused tests for the dependency-aware ``python -m sandbox`` dispatcher."""

from __future__ import annotations

import sys
from pathlib import Path

from sandbox import __main__ as dispatcher


def test_help_lists_llm(capsys):
    assert dispatcher.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "llm      smoke-test small, medium, or large (default: small)" in output
    assert "season.json" in output


def test_setup_help_prints_usage_instead_of_installing(capsys, monkeypatch):
    def unexpected_setup() -> str:
        raise AssertionError("setup called")

    monkeypatch.setattr(dispatcher, "setup", unexpected_setup)

    assert dispatcher.main(["setup", "--help"]) == 0
    assert "usage: python -m sandbox setup" in capsys.readouterr().out


def test_llm_dispatches_with_its_probe_and_forwards_arguments(monkeypatch):
    seen: list[tuple[list[str], str]] = []

    monkeypatch.setattr(dispatcher, "_run", lambda args, probe: seen.append((args, probe)) or 0)

    assert dispatcher.main(["llm", "--example-flag", "value"]) == 0
    assert seen == [(["-m", "sandbox.llm_example", "--example-flag", "value"], "import openai, dotenv")]


def test_browser_commands_dispatch_their_positional_modes(monkeypatch):
    seen: list[tuple[list[str], str]] = []
    monkeypatch.setattr(dispatcher, "_run", lambda args, probe: seen.append((args, probe)) or 0)

    assert dispatcher.main([]) == 0
    assert dispatcher.main(["play", "--seed", "7"]) == 0
    assert dispatcher.main(["watch", "--seed", "8"]) == 0

    assert seen == [
        (["-m", "sandbox.play", "human"], dispatcher._RUNTIME_PROBE),
        (["-m", "sandbox.play", "human", "--seed", "7"], dispatcher._RUNTIME_PROBE),
        (["-m", "sandbox.play", "agent", "--seed", "8"], dispatcher._RUNTIME_PROBE),
    ]


def test_retired_human_command_is_rejected(capsys):
    assert dispatcher.main(["human"]) == 2
    error = capsys.readouterr().err
    assert "unknown command 'human'" in error
    assert "play     control a player in your browser" in error


def test_current_interpreter_without_llm_dependencies_uses_setup(tmp_path: Path, monkeypatch):
    missing_venv = tmp_path / "missing-python"
    probes: list[tuple[str, str]] = []

    monkeypatch.setattr(dispatcher, "_venv_python", lambda: missing_venv)
    monkeypatch.setattr(
        dispatcher,
        "_has_runtime",
        lambda python, probe: probes.append((python, probe)) or False,
    )
    monkeypatch.setattr(dispatcher, "setup", lambda: "repaired-python")

    assert dispatcher._runtime_python(dispatcher._LLM_PROBE) == "repaired-python"
    assert probes == [(sys.executable, "import openai, dotenv")]


def test_stale_venv_without_llm_dependencies_is_repaired(tmp_path: Path, monkeypatch):
    stale_python = tmp_path / "python"
    stale_python.touch()
    probes: list[tuple[str, str]] = []

    monkeypatch.setattr(dispatcher, "_venv_python", lambda: stale_python)
    monkeypatch.setattr(
        dispatcher,
        "_has_runtime",
        lambda python, probe: probes.append((python, probe)) or False,
    )
    monkeypatch.setattr(dispatcher, "setup", lambda: "repaired-python")

    assert dispatcher._runtime_python(dispatcher._LLM_PROBE) == "repaired-python"
    assert probes == [(str(stale_python), "import openai, dotenv")]


def test_current_compatible_interpreter_does_not_bootstrap(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dispatcher, "_venv_python", lambda: tmp_path / "missing-python")
    monkeypatch.setattr(dispatcher, "_has_runtime", lambda python, probe: True)

    def unexpected_setup() -> str:
        raise AssertionError("setup called")

    monkeypatch.setattr(dispatcher, "setup", unexpected_setup)

    assert dispatcher._runtime_python(dispatcher._LLM_PROBE) == sys.executable
