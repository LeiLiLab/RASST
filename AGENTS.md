# RASST Agent Rules

This repository is now in release-preparation mode. The default task is to make
the RASST codebase reproducible, understandable, and safe to open source. Do not
treat this repo as an active paper-exploration workspace unless the user
explicitly asks for a new experiment.

## Active Scope

- Prioritize code organization, release-facing wrappers, eval manifests,
  documentation, tests, and artifact validation.
- Keep changes small and reviewable, primarily under `code/rasst/`, `docs/`, and
  root-level release documentation.
- Do not edit the submitted paper PDF, figures, or paper text unless the user
  explicitly requests it.
- Avoid ablation work, metric chasing, or new exploratory variants unless the
  user explicitly brings them back into scope.
- Preserve frozen or legacy material as provenance. Do not refactor archived
  legacy code in place just to clean it up.

The old InfiniSST experiment-control rule is archived at
`code/rasst/common/rasst_original_frozen_agent_rule.md`. It is historical
provenance, not the active operating rule for this repo.

## Canonical Paths And Hosts

- Current host: `taurus`.
- Canonical repo root: `/mnt/taurus/data2/jiaxuanluo/RASST`.
- Canonical large-artifact/output base for RASST release work:
  `/mnt/taurus/data2/jiaxuanluo/RASST_release_runs` unless the user specifies a
  different root.
- On Taurus, `/mnt/data2` resolves to `/mnt/taurus/data2`.
- On Aries, `/mnt/data2` is Aries-local and effectively `/mnt/aries/data2`.
- On Taurus, `/home/<user>` is the local view of `/mnt/taurus/home/<user>`.
- On Aries, `/home/<user>` is the local view of `/mnt/aries/home/<user>`.
- For any cross-host script, manifest, launcher, log, report, command, env var,
  model path, cache path, or Python executable, use host-qualified absolute
  paths such as `/mnt/taurus/data2/...` and `/mnt/taurus/home/...`.
- Do not write cross-host references as `/mnt/data2/...` or `/home/...`; those
  paths change meaning when the same command is executed on Aries.
- A command intended to run unchanged on both Taurus and Aries should contain no
  bare `/mnt/data2`, `/mnt/home`, or `/home` paths. Expand them to
  `/mnt/taurus/...` before launching.
- Aries should be accessed through port `20042` when submitting or checking
  remote jobs, for example:

```bash
ssh -p 20042 aries 'hostname'
```

If port `20042` is unavailable in a local environment, report that explicitly
instead of silently switching path assumptions.

## Release Hygiene

- Do not add generated outputs, caches, checkpoints, logs, local W&B folders, or
  large artifacts to git.
- Keep runtime products under ignored runtime roots such as `outputs/`, `logs/`,
  `checkpoints/`, external data roots, or explicitly documented release-run
  directories.
- Prefer deterministic wrappers and manifests over ad hoc commands.
- Prefer fail-fast validation over fallback behavior. Missing data, models,
  glossaries, checkpoints, metrics, manifests, or paths must raise a clear error
  unless a fallback is explicitly documented and enabled by the user.
- Do not silently skip malformed samples or use dummy data. If filtering is
  intentional, log total, kept, dropped, and reasons.

## Eval And Experiment Safety

- Long-running data prep, training, eval, SimulEval, or analysis jobs must not
  run in the foreground of an interactive shell.
- Prefer Slurm. If Slurm is not appropriate, use detached `setsid` with
  persistent stdout/stderr logs and a pid file.
- Standalone eval truth is the verified artifact set:
  `eval_results.tsv`, `instances.log`, `instances.strip_term.log`, and the
  manifest/config evidence that names the exact inputs and settings.
- W&B is optional for standalone eval readouts and should not block a valid eval
  when TSV/log artifacts are complete and validated.
- When reporting metrics or status, verify from artifacts, logs, pids, Slurm, or
  manifests. Do not quote metrics from chat memory.

## Practical Defaults

- Start RASST-local work from `/mnt/taurus/data2/jiaxuanluo/RASST`.
- Use host-qualified absolute paths in docs, launchers, manifests, and detached
  commands when paths may be consumed from both Taurus and Aries.
- Prefer `/mnt/taurus/home/<user>/...` over `/home/<user>/...` for conda envs,
  Python binaries, Hugging Face caches, local tools, and other home-directory
  dependencies referenced by release scripts.
- Use `rg`/`rg --files` for repository search.
- Use `apply_patch` for manual file edits.
- Respect a dirty worktree: do not revert or overwrite user changes unless the
  user explicitly requests it.
