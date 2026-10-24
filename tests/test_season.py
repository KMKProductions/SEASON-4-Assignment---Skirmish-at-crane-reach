"""Tests for optional local season settings and their command-line precedence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from sandbox import evaluate, play
from sandbox.harness.environment import EnvParameter, PlayerBounds, resolve_parameters
from sandbox.season import SeasonSettings, load_season_settings


def _write_settings(root: Path, document: object) -> None:
    (root / "season.json").write_text(json.dumps(document), encoding="utf-8")


def test_load_season_settings_keeps_current_behavior_when_the_file_is_absent(tmp_path: Path):
    assert load_season_settings(tmp_path, play.META) is None


def test_load_season_settings_reads_the_downloaded_values(tmp_path: Path):
    _write_settings(
        tmp_path,
        {
            "env_id": play.META.env_id,
            "season": "Spring practice",
            "parameters": {},
            "decision_limit_ms": 50,
            "game_limit_ms": 3_000,
        },
    )

    settings = load_season_settings(tmp_path, play.META)

    assert settings is not None
    assert settings.label == "Spring practice"
    assert settings.parameters == resolve_parameters(play.META)
    assert settings.decision_limit_ms == 50
    assert settings.game_limit_ms == 3_000


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("not JSON", "not valid JSON"),
        (
            {"env_id": play.META.env_id, "season": "Here", "game_limit_mss": 3_000},
            "unknown field 'game_limit_mss'",
        ),
        ({"env_id": "other", "season": "Elsewhere"}, "for environment"),
        ({"env_id": play.META.env_id, "season": ""}, "nonempty string"),
        ({"env_id": play.META.env_id, "season": "Here", "parameters": []}, "parameters"),
        (
            {"env_id": play.META.env_id, "season": "Here", "parameters": {"unknown": 1}},
            "unknown environment parameter",
        ),
        (
            {"env_id": play.META.env_id, "season": "Here", "parameters": {"pipe_gap": 1}},
            "pipe_gap",
        ),
        (
            {"env_id": play.META.env_id, "season": "Here", "decision_limit_ms": 0},
            "decision_limit_ms",
        ),
    ],
)
def test_load_season_settings_rejects_invalid_files(tmp_path: Path, document: object, message: str):
    path = tmp_path / "season.json"
    path.write_text(document if isinstance(document, str) else json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_season_settings(tmp_path, play.META)


def test_play_cli_overrides_season_parameters_and_limits_once(monkeypatch, capsys):
    meta = replace(
        play.META,
        parameters=(
            *play.META.parameters,
            EnvParameter(
                name="test_option",
                title="Test option",
                description="A test-only option.",
                type="int",
                default=100,
                min=50,
                max=200,
            ),
        ),
    )
    monkeypatch.setattr(play, "META", meta)
    settings = SeasonSettings(
        "Spring practice",
        resolve_parameters(meta, {"test_option": 120}),
        50,
        3_000,
    )
    monkeypatch.setattr(play, "load_season_settings", lambda root, meta: settings)
    monkeypatch.setattr(play, "possible_players", lambda parameters=None: ("player_0",))
    captured: dict[str, object] = {}
    monkeypatch.setattr(play, "run_headless", lambda **kwargs: captured.update(kwargs) or 1.0)

    assert (
        play.main(
            [
                "agent",
                "--headless",
                "--parameter",
                "test_option=140",
                "--decision-limit-ms",
                "70",
                "--game-limit-ms",
                "4",
            ]
        )
        == 0
    )

    assert captured["parameters"] == resolve_parameters(meta, {"test_option": 140})
    assert captured["decision_limit_ms"] == 70
    assert captured["game_limit_ms"] == 4
    assert capsys.readouterr().out.count("Using Spring practice settings from season.json.") == 1


def test_play_browser_passes_season_limits_to_the_local_runner(monkeypatch, capsys):
    settings = SeasonSettings("Spring practice", resolve_parameters(play.META), 50, 3_000)
    monkeypatch.setattr(play, "load_season_settings", lambda root, meta: settings)
    monkeypatch.setattr(play, "possible_players", lambda parameters: ("player_0",))
    captured: dict[str, object] = {}
    monkeypatch.setattr(play, "local_config", lambda **kwargs: captured.update(kwargs) or {})
    monkeypatch.setattr(play, "launch_browser", lambda config, **kwargs: 0)

    assert play.main(["agent", "--no-browser"]) == 0

    assert captured["decision_limit_ms"] == 50
    assert captured["game_limit_ms"] == 3_000
    assert capsys.readouterr().out.count("Using Spring practice settings from season.json.") == 1


def test_eval_loads_and_announces_settings_once_for_many_seeds(monkeypatch, capsys):
    settings = SeasonSettings("Spring practice", resolve_parameters(play.META), 50, 3_000)
    monkeypatch.setattr(evaluate, "load_season_settings", lambda root, meta: settings)
    monkeypatch.setattr(evaluate, "parse_rival", lambda parser, raw, seat, parameters: None)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(evaluate, "run_headless", lambda **kwargs: calls.append(kwargs) or 1.0)

    assert evaluate.main(["--episodes", "2"]) == 0

    assert [call["seed"] for call in calls] == [0, 1]
    assert all(call["decision_limit_ms"] == 50 for call in calls)
    assert capsys.readouterr().out.count("Using Spring practice settings from season.json.") == 1


def test_eval_rejects_a_seat_outside_the_resolved_season_layout(monkeypatch, capsys):
    meta = replace(play.META, layout=PlayerBounds(min=1, max=1), presets=())
    monkeypatch.setattr(play, "META", meta)
    settings = SeasonSettings("Solo season", resolve_parameters(meta), 50, 3_000)
    monkeypatch.setattr(evaluate, "load_season_settings", lambda root, meta: settings)

    with pytest.raises(SystemExit) as error:
        evaluate.main(["--seat", "1"])

    assert error.value.code == 2
    assert "--seat must name one of 0..0" in capsys.readouterr().err
