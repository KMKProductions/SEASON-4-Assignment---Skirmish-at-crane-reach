# Skirmish at Crane Reach agent

Edit `agent.py` to build one unit's behavior for Skirmish at Crane Reach. Every unit on a side runs a separate instance of the same `Agent` class, so they do not share state or variables. `sandbox/` is provided code: do not edit it.

Start with the [Getting Started guide](https://vox-deorum.github.io/game-sandbox/students/getting-started/). Then run these commands from this folder as you work:

```console
python -m sandbox play   # command a side yourself in your browser
python -m sandbox watch  # watch your agent take on Naive
python -m sandbox test   # run the provided checks
python -m sandbox eval   # compare your agent with Naive
```

**Naive** is a simple built-in opponent. It holds the other side in `watch` and `eval`, and `eval` reports your side's average score over repeatable matches. The [`environment.md`](environment.md) guide explains rivals, presets, and the other command options.

## Files you will use

| Path | Purpose |
| --- | --- |
| `agent.py` | Your `Agent` implementation and the first TODO locations. |
| `environment.md` | Crane rules, starter walkthrough, helpers, observations, and settings. |
| `manifest.json` | Names the agent class for a submission. |
| `season.json` | Optional local season settings downloaded from My Submissions. |
| `tests/` | Checks your submission should pass. |
| `sandbox/` | Local game, commands, helper package, and observation types. Do not edit it. |
| `requirements.txt` | Exact Python package versions used by the server. |
| `requirements-dev.txt` | Test dependencies. |
| `.env.example` | Example local LLM settings. |

The starter returns Crane orders with `action.move()` and `action.stay()` from `sandbox.crane`. Its `act(observation)` receives the current observation and action mask. Before changing the strategy, read [`environment.md`](environment.md). It starts with a small archer improvement you can copy, then explains when an order is legal.

Leave `sandbox/`, `requirements.in`, and `requirements.txt` unchanged. The pinned packages match the server. Ask your instructor before adding a package.

When your agent is ready, follow the shared [submitting guide](https://vox-deorum.github.io/game-sandbox/students/submitting/). For the optional `learn` and `chat` hooks, see the shared [agent interface](https://vox-deorum.github.io/game-sandbox/students/agent-interface/). Crane messaging begins in Season 3.

## Optional LLM API

If your instructor enables model calls, follow [Using the LLM API](llm.md). Copy `.env.example` to `.env`, add the endpoint and key, and never commit either secret.

Test the connection with:

```console
python -m sandbox llm
```
