"""The generic browser launcher always gives the live runner complete player attribution."""

from __future__ import annotations

import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sandbox import evaluate, live_local, play
from sandbox.harness.environment import (
    EnvParameter,
    EnvPreset,
    PlayerBounds,
    SeatDeclaration,
    SeatPlan,
    SeatPlans,
    resolve_parameters,
)
from sandbox.harness.live import parse_config
from sandbox.season import SeasonSettings


def _rival_repo(tmp_path: Path) -> Path:
    rival = tmp_path / "rivals" / "v1"
    rival.mkdir(parents=True)
    (rival / "manifest.json").write_text("{}", encoding="utf-8")
    return rival


def _use_player_bounds(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    *,
    human_players: tuple[str, ...] | None = None,
) -> None:
    humans = tuple(f"player_{index}" for index in range(count)) if human_players is None else human_players
    monkeypatch.setattr(
        play,
        "META",
        replace(
            play.META,
            layout=PlayerBounds(min=count, max=count),
            human_players=humans,
            presets=(),
        ),
    )


def _use_partnership_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    plans = SeatPlans(
        (
            SeatPlan(
                "partnership",
                "Partnership",
                (SeatDeclaration((0, 2)), SeatDeclaration((1, 3))),
            ),
        )
    )
    monkeypatch.setattr(
        play,
        "META",
        replace(
            play.META,
            layout=plans,
            human_players=("player_0", "player_1", "player_2", "player_3"),
            presets=(),
        ),
    )


def _use_restricted_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    plans = SeatPlans(
        (
            SeatPlan(
                "roles",
                "Roles",
                (
                    SeatDeclaration((2,)),
                    SeatDeclaration((0, 1), restricted_builtin="naive"),
                ),
            ),
        )
    )
    monkeypatch.setattr(
        play,
        "META",
        replace(play.META, layout=plans, human_players=("player_0",), presets=()),
    )


def _use_three_branches_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror Days at Three Branches' real seat plans (environments/three_branches/__init__.py):
    a cast seat that scales with the plan, plus one visitor seat restricted to a builtin. Only
    the visitor (player_0) is declared human-playable. The restriction names ``naive`` rather than
    three_branches' real ``scripted_visitor`` builtin, since this fixture composes against
    whichever environment the template test run happens to target, and ``naive`` is the one
    builtin every environment declares; the seat shape under test does not depend on which name
    the restriction carries."""
    plans = SeatPlans(
        (
            SeatPlan(
                "cast_5",
                "Five villagers",
                (
                    SeatDeclaration((1, 2, 3, 4, 5)),
                    SeatDeclaration((0,), restricted_builtin="naive"),
                ),
            ),
            SeatPlan(
                "cast_10",
                "Ten villagers",
                (
                    SeatDeclaration(tuple(range(1, 11))),
                    SeatDeclaration((0,), restricted_builtin="naive"),
                ),
            ),
        )
    )
    monkeypatch.setattr(
        play,
        "META",
        replace(play.META, layout=plans, human_players=("player_0",), presets=()),
    )


def test_single_player_local_config_uses_metadata_timeout_when_omitted(monkeypatch, tmp_path: Path):
    _use_player_bounds(monkeypatch, 1)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")

    config = play.local_config(
        seed=7,
        mode="human",
        seat=0,
        recording_dir=tmp_path / "recordings",
        step_limit=None,
    )

    assert config["player_bindings"] == {"player_0": {"kind": "external"}}
    assert config["players"] == {"player_0": {"kind": "human", "label": "You"}}
    assert "human_timeout_ms" not in config
    assert "max_steps" not in config


def test_four_player_local_config_covers_every_player_and_preserves_null_timeout(monkeypatch, tmp_path: Path):
    player_ids = ("player_0", "player_1", "player_2", "player_3")
    _use_player_bounds(monkeypatch, 4)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")

    config = play.local_config(
        seed=8,
        mode="human",
        seat=2,
        recording_dir=tmp_path / "recordings",
        step_limit=52,
        human_timeout_ms=None,
    )

    assert set(config["player_bindings"]) == set(player_ids)
    assert set(config["players"]) == set(player_ids)
    assert config["player_bindings"]["player_2"] == {"kind": "external"}
    assert config["players"]["player_2"] == {"kind": "human", "label": "You"}
    assert config["human_timeout_ms"] is None
    assert config["max_steps"] == 52


def test_human_mode_rejects_a_seat_excluded_from_metadata(monkeypatch, capsys):
    _use_player_bounds(monkeypatch, 2, human_players=("player_0",))

    with pytest.raises(SystemExit) as error:
        play.main(["human", "--seat", "1"])

    assert error.value.code == 2
    assert "not human-playable" in capsys.readouterr().err


def test_headless_allows_an_unrestricted_seat_excluded_from_human_metadata(monkeypatch, capsys):
    _use_player_bounds(monkeypatch, 2, human_players=("player_0",))
    monkeypatch.setattr(play, "run_headless", lambda **kwargs: 3.5)

    assert play.main(["human", "--headless", "--seat", "1", "--seed", "7"]) == 0
    assert capsys.readouterr().out == "seed 7: score 3.50\n"


def test_mode_defaults_select_the_platform_human_seat_and_first_unrestricted_seat(monkeypatch):
    _use_restricted_layout(monkeypatch)
    layout = play.resolved_layout(resolve_parameters(play.META))

    assert play.default_human_seat_index(layout) == 1
    assert play.default_agent_seat_index(layout) == 0


def test_restricted_human_seat_derives_its_builtin_companion(monkeypatch, tmp_path: Path):
    _use_restricted_layout(monkeypatch)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(play, "BUILTIN_AGENT_ROOT", tmp_path / "builtins")
    (play.BUILTIN_AGENT_ROOT / "naive").mkdir(parents=True)

    config = play.local_config(
        seed=0,
        mode="human",
        seat=1,
        recording_dir=tmp_path / "recordings",
        step_limit=None,
    )

    assert config["player_bindings"]["player_0"] == {"kind": "external"}
    assert config["player_bindings"]["player_1"] == {
        "kind": "builtin-agent",
        "path": str(play.BUILTIN_AGENT_ROOT / "naive"),
        "name": "naive",
    }
    assert config["players"]["player_1"]["builtin_name"] == "naive"
    assert config["external_chat_player"] == "player_0"

    with pytest.raises(ValueError, match="restricted human seat"):
        play.local_config(
            seed=0,
            mode="human",
            seat=1,
            recording_dir=tmp_path / "recordings",
            step_limit=None,
            companion="self",
        )


@pytest.mark.parametrize("plan_key", ["cast_5", "cast_10"])
def test_three_branches_layouts_default_the_human_seat_to_the_visitor(
    monkeypatch: pytest.MonkeyPatch, plan_key: str
):
    _use_three_branches_layout(monkeypatch)
    layout = play.resolved_layout(resolve_parameters(play.META, {"seat_plan": plan_key}))

    assert layout.seats[1].players == ("player_0",)
    assert play.default_human_seat_index(layout) == 1


@pytest.mark.parametrize("plan_key", ["cast_5", "cast_10"])
def test_three_branches_local_config_gives_the_visitor_seat_to_the_human(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plan_key: str
):
    _use_three_branches_layout(monkeypatch)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    parameters = resolve_parameters(play.META, {"seat_plan": plan_key})
    seat = play.default_human_seat_index(play.resolved_layout(parameters))

    config = play.local_config(
        seed=0,
        mode="human",
        seat=seat,
        recording_dir=tmp_path / "recordings",
        step_limit=None,
        parameters=parameters,
    )

    assert config["external_chat_player"] == "player_0"
    assert config["player_bindings"]["player_0"] == {"kind": "external"}
    assert config["players"]["player_0"] == {"kind": "human", "label": "You"}


def test_launcher_rejects_a_mode_it_does_not_define():
    # `python -m sandbox watch` reaches this launcher as its `agent` mode, not by that name.
    with pytest.raises(SystemExit) as error:
        play.main(["watch"])

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("command", "arguments"),
    ((play.main, ["agent", "--player", "0"]), (evaluate.main, ["--player", "0"])),
)
def test_student_commands_reject_the_removed_player_option(command, arguments, capsys):
    with pytest.raises(SystemExit) as error:
        command(arguments)

    assert error.value.code == 2
    assert "unrecognized arguments: --player 0" in capsys.readouterr().err


def test_preset_replaces_season_parameters_even_when_its_values_are_empty(monkeypatch, capsys):
    monkeypatch.setattr(play, "META", replace(play.META, presets=(EnvPreset("season_1", "Season 1", {}),)))
    monkeypatch.setattr(
        play,
        "load_season_settings",
        lambda _root, _meta: SeasonSettings("Downloaded season", {"round_cap": 150}, 123, 456),
    )
    captured: dict[str, object] = {}

    def run_headless(**kwargs: object) -> float:
        captured.update(kwargs)
        return 2.0

    monkeypatch.setattr(play, "run_headless", run_headless)

    assert play.main(["agent", "--headless", "--preset", "season_1"]) == 0
    assert captured["parameters"] == resolve_parameters(play.META)
    assert captured["decision_limit_ms"] == 123
    assert captured["game_limit_ms"] == 456
    assert capsys.readouterr().out == (
        "Using the season_1 preset with the time limits from season.json.\nseed 0: score 2.00\n"
    )


def test_preset_parameters_yield_to_explicit_parameter_overrides(monkeypatch):
    # Declare a test-only parameter on the metadata so the preset and the override validate in
    # every environment, whatever its real parameters are.
    knob = EnvParameter(
        name="trial_knob",
        title="Trial knob",
        description="Test-only tuning parameter.",
        type="int",
        default=100,
        min=1,
        max=1000,
    )
    monkeypatch.setattr(
        play,
        "META",
        replace(
            play.META,
            parameters=(*play.META.parameters, knob),
            presets=(EnvPreset("short", "Short", {"trial_knob": 150}),),
        ),
    )
    captured: dict[str, object] = {}

    def run_headless(**kwargs: object) -> float:
        captured.update(kwargs)
        return 2.0

    monkeypatch.setattr(play, "run_headless", run_headless)

    assert play.main(["agent", "--headless", "--preset", "short", "--parameter", "trial_knob=200"]) == 0
    assert captured["parameters"] == resolve_parameters(play.META, {"trial_knob": 200})


def test_unknown_preset_reports_the_available_names(monkeypatch, capsys):
    monkeypatch.setattr(
        play,
        "META",
        replace(play.META, presets=(EnvPreset("season_1", "Season 1", {}),)),
    )

    with pytest.raises(SystemExit) as error:
        play.main(["agent", "--preset", "missing"])

    assert error.value.code == 2
    assert "unknown environment preset 'missing'; available: season_1" in capsys.readouterr().err


def test_evaluate_forwards_the_selected_seat(monkeypatch, capsys):
    calls: list[dict[str, object]] = []
    _use_player_bounds(monkeypatch, 2)
    parameters = resolve_parameters(evaluate.play.META)

    def run_headless(**kwargs: object) -> float:
        calls.append(kwargs)
        return 2.0

    monkeypatch.setattr(evaluate, "run_headless", run_headless)

    assert evaluate.main(["--episodes", "2", "--seat", "1"]) == 0
    assert calls == [
        dict(
            seed=0,
            max_steps=None,
            seat=1,
            vs=None,
            parameters=parameters,
            decision_limit_ms=None,
            game_limit_ms=None,
        ),
        dict(
            seed=1,
            max_steps=None,
            seat=1,
            vs=None,
            parameters=parameters,
            decision_limit_ms=None,
            game_limit_ms=None,
        ),
    ]
    assert "mean over 2 episode(s): 2.00" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("command", "arguments"),
    ((play, ["agent", "--headless"]), (evaluate, ["--episodes", "1"])),
)
def test_headless_commands_propagate_participant_value_errors(monkeypatch, command, arguments):
    _use_player_bounds(monkeypatch, 1)

    def crash(**kwargs: object) -> float:
        raise ValueError("student act crashed")

    monkeypatch.setattr(command, "run_headless", crash)

    with pytest.raises(ValueError, match="student act crashed"):
        command.main(arguments)


def test_local_runner_passes_stdin_to_the_harness_run_seam(monkeypatch, tmp_path: Path):
    """The relocated harness, not this shim, starts stdin after client-ready state emits."""
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(live_local, "_claim_stdout", lambda: io.StringIO())
    monkeypatch.setattr(live_local, "run", fake_run)
    seat = play.default_human_seat_index(play.resolved_layout())
    config = play.local_config(
        seed=0,
        mode="human",
        seat=seat,
        recording_dir=tmp_path / "recordings",
        step_limit=None,
    )

    assert live_local.main([json.dumps(config)]) == 0
    assert captured["command_lines"] is sys.stdin


def test_resolve_rival_accepts_a_folder_and_its_manifest_path(monkeypatch, tmp_path: Path):
    rival = _rival_repo(tmp_path)

    assert play.resolve_rival(str(rival)) == rival.resolve()
    assert play.resolve_rival(str(rival / "manifest.json")) == rival.resolve()

    monkeypatch.chdir(tmp_path)
    assert play.resolve_rival("rivals/v1") == rival.resolve()


def test_resolve_rival_rejects_missing_and_manifestless_paths(tmp_path: Path):
    with pytest.raises(ValueError, match="could not find"):
        play.resolve_rival(str(tmp_path / "missing"))

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no manifest.json"):
        play.resolve_rival(str(empty))


def test_vs_fills_every_opposing_player_in_a_one_player_per_seat_layout(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(play, "META", replace(play.META, layout=PlayerBounds(min=4, max=4), presets=()))
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    rival = _rival_repo(tmp_path)

    config = play.local_config(
        seed=1, mode="agent", seat=0, recording_dir=tmp_path / "recordings", step_limit=None, vs=rival
    )

    bindings = config["player_bindings"]
    assert bindings["player_0"] == {"kind": "builtin-agent", "path": str(tmp_path / "repo")}
    for other in ("player_1", "player_2", "player_3"):
        assert bindings[other] == {"kind": "builtin-agent", "path": str(rival)}
    assert config["players"]["player_0"] == {
        "kind": "agent",
        "submission_id": "local",
        "label": "Your agent",
    }
    assert config["players"]["player_1"] == {
        "kind": "agent",
        "submission_id": "local-rival",
        "label": "Rival (v1)",
    }
    parse_config([json.dumps(config)], entry=play._entry())


def test_vs_keeps_the_selected_seat_on_your_agent_in_a_partnership_layout(monkeypatch, tmp_path: Path):
    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    rival = _rival_repo(tmp_path)

    config = play.local_config(
        seed=1, mode="agent", seat=0, recording_dir=tmp_path / "recordings", step_limit=None, vs=rival
    )

    bindings = config["player_bindings"]
    assert bindings["player_0"]["path"] == str(tmp_path / "repo")
    assert bindings["player_2"]["path"] == str(tmp_path / "repo")
    assert bindings["player_1"]["path"] == str(rival)
    assert bindings["player_3"]["path"] == str(rival)


def test_human_mode_without_companion_controls_the_whole_wide_seat(monkeypatch, tmp_path: Path):
    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    rival = _rival_repo(tmp_path)

    config = play.local_config(
        seed=1, mode="human", seat=0, recording_dir=tmp_path / "recordings", step_limit=None, vs=rival
    )

    assert config["player_bindings"]["player_0"] == {"kind": "external"}
    assert config["player_bindings"]["player_2"] == {"kind": "external"}
    assert config["players"]["player_2"] == {"kind": "human", "label": "You"}
    assert config["player_bindings"]["player_1"]["path"] == str(rival)
    assert config["player_bindings"]["player_3"]["path"] == str(rival)


def test_saved_companion_keeps_the_first_human_capable_member_as_the_chat_sender(
    monkeypatch,
    tmp_path: Path,
):
    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "META", replace(play.META, human_players=("player_2",)))
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    companion = _rival_repo(tmp_path)

    config = play.local_config(
        seed=1,
        mode="human",
        seat=0,
        recording_dir=tmp_path / "recordings",
        step_limit=None,
        companion=str(companion),
    )

    assert config["external_chat_player"] == "player_2"
    assert config["player_bindings"]["player_2"] == {"kind": "external"}
    assert config["player_bindings"]["player_0"]["path"] == str(companion.resolve())
    assert config["players"]["player_0"]["submission_id"] == "local-companion"


def test_whole_seat_default_uses_the_first_member_as_the_chat_sender(monkeypatch, tmp_path: Path):
    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")

    config = play.local_config(
        seed=1, mode="human", seat=0, recording_dir=tmp_path / "recordings", step_limit=None
    )

    assert config["external_chat_player"] == "player_0"
    assert config["player_bindings"]["player_0"] == {"kind": "external"}
    assert config["player_bindings"]["player_2"] == {"kind": "external"}
    # Playing by hand leaves your own agent opposite you, so a session can test it.
    assert config["player_bindings"]["player_1"]["path"] == str(tmp_path / "repo")
    assert config["player_bindings"]["player_3"]["path"] == str(tmp_path / "repo")


def test_watching_without_vs_puts_the_naive_baseline_opposite_your_agent(monkeypatch, tmp_path: Path):
    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(play, "BUILTIN_AGENT_ROOT", tmp_path / "builtins")
    (play.BUILTIN_AGENT_ROOT / "naive").mkdir(parents=True)

    config = play.local_config(
        seed=1, mode="agent", seat=0, recording_dir=tmp_path / "recordings", step_limit=None
    )

    for player_id in ("player_0", "player_2"):
        assert config["player_bindings"][player_id] == {
            "kind": "builtin-agent",
            "path": str(tmp_path / "repo"),
        }
        assert config["players"][player_id] == {
            "kind": "agent",
            "submission_id": "local",
            "label": "Your agent",
        }
    for player_id in ("player_1", "player_3"):
        assert config["player_bindings"][player_id] == {
            "kind": "builtin-agent",
            "path": str(play.BUILTIN_AGENT_ROOT / "naive"),
            "name": "naive",
        }
        assert config["players"][player_id] == {
            "kind": "agent",
            "builtin_name": "naive",
            "label": play.META.builtin_agents[0].label,
        }


def test_vs_errors_in_a_single_player_game(monkeypatch, capsys, tmp_path: Path):
    _use_player_bounds(monkeypatch, 1)

    with pytest.raises(SystemExit) as error:
        play.main(["agent", "--vs", str(tmp_path)])
    assert error.value.code == 2
    assert "only one player" in capsys.readouterr().err

    with pytest.raises(SystemExit) as error:
        evaluate.main(["--vs", str(tmp_path)])
    assert error.value.code == 2
    assert "only one player" in capsys.readouterr().err


def test_vs_errors_when_one_seat_covers_every_player(monkeypatch, capsys, tmp_path: Path):
    plans = SeatPlans((SeatPlan("coop", "Cooperative", (SeatDeclaration((0, 1, 2)),)),))
    monkeypatch.setattr(play, "META", replace(play.META, layout=plans, presets=()))

    with pytest.raises(SystemExit) as error:
        play.main(["agent", "--vs", str(tmp_path)])
    assert error.value.code == 2
    assert "part of your team" in capsys.readouterr().err

    with pytest.raises(SystemExit) as error:
        evaluate.main(["--vs", str(tmp_path)])
    assert error.value.code == 2
    assert "part of your team" in capsys.readouterr().err


def test_self_companion_is_rejected_where_it_cannot_apply(monkeypatch, capsys, tmp_path: Path):
    _use_player_bounds(monkeypatch, 1)

    # A team of one has nobody else to play.
    with pytest.raises(SystemExit) as error:
        play.main(["human", "--companion", "self"])
    assert error.value.code == 2
    assert "more than one player" in capsys.readouterr().err

    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "META", replace(play.META, human_players=("player_0",)))

    # A teammate nobody may steer cannot be played by hand either.
    with pytest.raises(SystemExit) as error:
        play.main(["human", "--companion", "self"])
    assert error.value.code == 2
    assert "cannot be controlled by a human" in capsys.readouterr().err

    with pytest.raises(SystemExit) as error:
        play.main(["human"])
    assert error.value.code == 2
    assert "pass --companion" in capsys.readouterr().err

    # Watching your own agent is not playing it.
    with pytest.raises(SystemExit) as error:
        play.main(["agent", "--companion", "self"])
    assert error.value.code == 2
    assert "browser human mode" in capsys.readouterr().err

    with pytest.raises(SystemExit) as error:
        play.main(["human", "--companion", str(tmp_path)])
    assert error.value.code == 2


def test_named_builtin_companion_carries_matching_binding_and_attribution(monkeypatch, tmp_path: Path):
    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "BUILTIN_AGENT_ROOT", tmp_path / "builtins")
    (play.BUILTIN_AGENT_ROOT / "naive").mkdir(parents=True)

    config = play.local_config(
        seed=1,
        mode="human",
        seat=0,
        recording_dir=tmp_path / "recordings",
        step_limit=None,
        companion="naive",
    )

    assert config["player_bindings"]["player_0"] == {"kind": "external"}
    assert config["player_bindings"]["player_2"] == {
        "kind": "builtin-agent",
        "path": str(play.BUILTIN_AGENT_ROOT / "naive"),
        "name": "naive",
    }
    assert config["players"]["player_2"] == {
        "kind": "agent",
        "builtin_name": "naive",
        "label": play.META.builtin_agents[0].label,
    }


def test_run_headless_vs_loads_a_separate_agent_for_every_other_player(monkeypatch, tmp_path: Path):
    _use_partnership_layout(monkeypatch)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(play, "make_env", lambda parameters: SimpleNamespace(close=lambda: None))
    loads: list[Path] = []

    def fake_load(root: Path) -> object:
        loads.append(Path(root))
        return object()

    monkeypatch.setattr(play, "load_agent", fake_load)
    captured: dict[str, object] = {}

    def fake_episode(agent: object, env: object, **kwargs: object) -> float:
        captured.update(kwargs)
        captured["agent"] = agent
        return 1.5

    monkeypatch.setattr(play, "play_episode", fake_episode)
    rival = _rival_repo(tmp_path)

    assert play.run_headless(seed=3, max_steps=None, seat=0, vs=rival) == 1.5

    other_agents = captured["other_agents"]
    assert isinstance(other_agents, dict)
    assert set(other_agents) == {"player_1", "player_2", "player_3"}
    assert loads.count(rival) == 2
    assert loads.count(tmp_path / "repo") == 2
    assert captured["score_player_ids"] == ("player_0", "player_2")
    instances = [captured["agent"], *other_agents.values()]
    assert len({id(instance) for instance in instances}) == 4


def test_run_headless_without_vs_loads_naive_for_each_opposing_player(monkeypatch, tmp_path: Path):
    _use_player_bounds(monkeypatch, 2)
    monkeypatch.setattr(play, "REPO_ROOT", tmp_path / "repo")
    baseline = tmp_path / "builtins" / "naive"
    monkeypatch.setattr(play, "builtin_agent_path", lambda name: baseline)
    monkeypatch.setattr(play, "make_env", lambda parameters: SimpleNamespace(close=lambda: None))
    loads: list[Path] = []
    monkeypatch.setattr(play, "load_agent", lambda root: loads.append(Path(root)) or object())
    captured: dict[str, object] = {}

    def fake_episode(agent: object, env: object, **kwargs: object) -> float:
        captured.update(kwargs)
        return 0.0

    monkeypatch.setattr(play, "play_episode", fake_episode)

    assert play.run_headless(seed=0, max_steps=None, seat=0) == 0.0
    assert set(captured["other_agents"]) == {"player_1"}
    assert captured["score_player_ids"] == ("player_0",)
    assert loads == [baseline, tmp_path / "repo"]


def test_play_episode_binds_other_agents_and_defaults(monkeypatch):
    players = ("player_0", "player_1", "player_2")
    monkeypatch.setattr(play, "possible_players", lambda parameters=None: players)
    captured: dict[str, object] = {}

    def fake_run_episode(entry: object, players: object, **kwargs: object) -> SimpleNamespace:
        captured["players"] = players
        return SimpleNamespace(scores={"player_0": 4.0})

    monkeypatch.setattr(play, "run_episode", fake_run_episode)
    mine, rival_agent = object(), object()

    score = play.play_episode(
        mine, object(), seed=0, player_id="player_0", other_agents={"player_2": rival_agent}
    )

    assert score == 4.0
    players = captured["players"]
    assert isinstance(players, dict)
    assert isinstance(players["player_0"], play.AgentPlayer)
    assert players["player_0"].agent is mine
    assert isinstance(players["player_2"], play.AgentPlayer)
    assert players["player_2"].agent is rival_agent
    assert isinstance(players["player_1"], play.ExternalPlayer)


def test_play_episode_averages_the_selected_seat_scores(monkeypatch):
    players = ("player_0", "player_1", "player_2")
    monkeypatch.setattr(play, "possible_players", lambda parameters=None: players)
    monkeypatch.setattr(
        play,
        "run_episode",
        lambda *args, **kwargs: SimpleNamespace(scores={"player_0": 2.0, "player_1": 99.0, "player_2": 8.0}),
    )

    assert (
        play.play_episode(
            object(),
            object(),
            seed=0,
            player_id="player_0",
            score_player_ids=("player_0", "player_2"),
        )
        == 5.0
    )


def test_evaluate_forwards_vs_to_every_episode(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(play, "META", replace(play.META, layout=PlayerBounds(min=2, max=2), presets=()))
    parameters = resolve_parameters(evaluate.play.META)
    rival = _rival_repo(tmp_path)
    calls: list[dict[str, object]] = []

    def run_headless(**kwargs: object) -> float:
        calls.append(kwargs)
        return 1.0

    monkeypatch.setattr(evaluate, "run_headless", run_headless)

    assert evaluate.main(["--seeds", "5", "--vs", str(rival)]) == 0
    assert calls == [
        dict(
            seed=5,
            max_steps=None,
            seat=0,
            vs=rival.resolve(),
            parameters=parameters,
            decision_limit_ms=None,
            game_limit_ms=None,
        )
    ]
