# Repository Guidelines

## Project Structure & Module Organization
Core Deep CFR logic stays in `src/core/`; training orchestration and entry points in `src/training/`; agent variants in `src/agents/` and `src/opponent_modeling/`; shared helpers in `src/utils/`. Web UI experiments live in `src/web/`, CLI wrappers in `scripts/`, pretrained weights in `flagship_models/`, notebooks in `notebooks/`, and media in `images/`. Packaging metadata is managed by `setup.py`, `MANIFEST.in`, and `readme.md`.

## Build, Test, and Development Commands
- `pip install -e .` — install dependencies and expose the CLI.
- `deepcfr-train --iterations 200 --traversals 500` — run a baseline training job (outputs to `models/` and `logs/deepcfr/`).
- `deepcfr-play --models-dir flagship_models --model-pattern '*.pt'` — launch a local game against saved agents.
- `deepcfr-tournament --checkpoints flagship_models/model_a.pt flagship_models/model_b.pt` — compare checkpoints and write plots.
- `pytest` — execute the test suite; pair with `-k` for focused runs.
- `tensorboard --logdir logs/deepcfr` — inspect training curves.

## Coding Style & Naming Conventions
Stick to PEP 8, four-space indentation, and `snake_case` modules, functions, and variables. Classes stay `CamelCase`, constants uppercase, and docstrings follow the concise narrative used in `src/core/deep_cfr.py`. Favour clear, pure functions, add type hints for new public APIs, and keep imports grouped standard/third-party/local.

## Testing Guidelines
Mirror package paths under `tests/` (e.g., `tests/core/`) and name files `test_<feature>.py`. Target key behaviors: illegal action handling, checkpoint reloads, opponent modeling branches, and ensure `pytest --maxfail=1` passes before submission.

## Commit & Pull Request Guidelines
Replace the current terse lowercase commits with imperative subjects such as `training: tighten tensorboard logging` and add explanatory body text when context matters. Reference issues with `Fixes #123`, list validation commands, and attach GUI or plot screenshots when you touch `scripts/` visualizers. PRs modifying `src/core/` or `src/training/` should call out regression risks and request peer review.

## Model Artifacts & Configuration
Only check in reproducible checkpoints, naming them clearly (e.g., `iter_05000.pt`) before parking them in `flagship_models/`. Document any required environment keys for `scripts/telegram_notifier.py` in a shared `.env.example`, not the real `.env`. When experimenting, copy notebooks from `notebooks/`, log seeds and hyperparameters, and reflect durable defaults back into the CLI flags or README.
