# InfiniSST Agent Experiment Control Rules

This document is the agent-facing operating rule for experiment management in
`/home/jiaxuanluo/InfiniSST`.  It replaces ad hoc notes-only tracking with a
lineage-first workflow: WandB stores run truth, event manifests store
provenance, and SQLite is a generated index for fast lookup.

## 1. Core Principle

Every experiment action must answer these questions without guessing:

- Which launcher and command created this run or artifact?
- Which data, glossary, checkpoint, and config files were consumed?
- Which checkpoint, eval output, report, or log files were produced?
- Which WandB run id, if required, and Slurm job id correspond to the action?
- Which prior runs or data-prep events does this event depend on?

For training and checkpoint-producing runs, the answer must be in WandB plus an
event manifest.  For standalone readout/evaluation runs, an event manifest plus
verified output artifacts is sufficient; WandB is optional and should not be the
only proof of success.  Do not reconstruct missing provenance from memory unless
explicitly labelled as a backfill based on evidence.

## 2. Source Of Truth

WandB is authoritative for training and checkpoint-producing runs:

- run config
- run metrics
- code snapshot and diff
- run status
- verdict
- final metric bundles produced during training/model selection

Event manifests are authoritative for:

- launcher path
- launch command
- Slurm job id
- WandB run id link
- notes path
- input artifacts
- output artifacts
- logs and reports
- parent or baseline run links
- downstream event links

For standalone `offline_eval`, `simuleval`, and `stream_eval` readouts, the
verified output artifacts linked by the manifest are authoritative for metrics,
for example `eval_results.tsv`, `scores.tsv`, `instances.log`, and
`instances.strip_term.log`.  If WandB exists for such an eval, it is a useful
index/cache, not the sole source of truth.

Git is authoritative for source code.

SQLite at `documents/code/.cache/experiments.sqlite` is only a rebuildable index
derived from WandB plus event manifests.  For W&B-tracked training or
checkpoint-producing runs, if SQLite disagrees with WandB, WandB wins for
metrics/config/status.  For standalone eval readouts, verified artifacts linked
from the manifest win over stale or failed W&B cache entries.  If SQLite
disagrees with a manifest, the manifest should be corrected and re-registered.

Do not maintain hand-written metric tables as truth.  Reports can summarize
results.  Training metrics must remain in WandB and can only be cached by
`db-sync`; standalone eval metrics may be cached from verified TSV/log artifacts
when the manifest records the exact paths and validation evidence.

## 3. Required Tools

Use the repo tools:

```bash
python documents/code/general/wandb_tool.py ...
python documents/code/general/experiment_event.py ...
```

Do not hand-roll `wandb.Api()` snippets for read-side queries unless
`wandb_tool.py` cannot express the query.  The CLI encodes project conventions,
metric presets, at-best-step reporting, and SQLite sync.

Useful event commands:

```bash
python documents/code/general/experiment_event.py register <manifest.json>
python documents/code/general/experiment_event.py launch <manifest.json> -- sbatch --parsable <launcher.sh>
python documents/code/general/experiment_event.py find --family <family>
python documents/code/general/experiment_event.py show <event_id-or-run_id>
python documents/code/general/experiment_event.py files <event_id-or-run_id>
```

Useful WandB commands:

```bash
python documents/code/general/wandb_tool.py --project <project> db-sync --runs <run_id>
python documents/code/general/wandb_tool.py --project <project> db-show <run_id> --config --notes
python documents/code/general/wandb_tool.py --project <project> compare <run_ids...> --preset retriever_eval retriever_train --at-best-step --anchor-metric both
python documents/code/general/wandb_tool.py --project <project> db-compare <run_ids...> --refresh --anchor-metric both
```

## 4. Event Manifest Is Mandatory

Before any data-prep, retriever training, retriever eval, speech LLM training,
offline eval, simuleval, stream eval, or analysis event, create and register a
manifest under:

```text
documents/code/<module>/manifests/YYYY/MM/<event_id>.json
```

Use the template:

```text
documents/code/_templates/experiment_event_manifest_template.json
```

Each manifest must include:

```text
schema_version
event_id
event_type
family
variant
status
project
wandb_run_id
slurm_job_id
launcher_path
notes_path
cwd
command
parent_run_ids
artifacts
metadata
```

Allowed event types:

```text
data_prepare
retriever_train
retriever_eval
speech_llm_train
offline_eval
simuleval
stream_eval
analysis
maintenance
```

Every artifact entry must include:

```text
role
type
direction
path
metadata
```

Directions:

```text
input
output
control
log
unknown
```

Common artifact roles:

```text
launcher
common_launcher
train_script
eval_script
notes
train_jsonl
dev_jsonl
acl_dev_jsonl
medicine_dev_jsonl
eval_wiki_glossary
acl_eval_wiki_glossary
medicine_eval_wiki_glossary
train_exclude_term_glossary
checkpoint
checkpoint_dir
index_dir
eval_output
report
slurm_logs
```

Notes files explain intent and verdict.  Manifests link exact files.  Do not
rely on notes alone for provenance.

## 5. File Layout For New Work

Do not create a new folder for every ablation.  Use a fixed module structure
and timestamped files.

For `documents/code/train/term_train/`:

```text
src/
common/
launchers/YYYY/MM/
notes/YYYY/MM/
manifests/YYYY/MM/
reports/
archive/
```

Use the same shape for:

```text
documents/code/simuleval/
documents/code/offline_evaluation/
documents/code/data_pre/<module>/
```

Existing root-level files may stay during migration.  New experiments should
use the fixed subfolders.  When substantially editing an old root-level launcher,
either create a new timestamped launcher under `launchers/YYYY/MM/` or add/update
a manifest under `manifests/YYYY/MM/`.

Do not put dependency folders such as `node_modules`, local WandB run folders,
logs, cache directories, or generated frontend build outputs under experiment
data/control roots unless there is a specific reason.  If they already exist,
ignore them for provenance unless explicitly referenced as artifacts.

## 6. Pre-Launch Workflow

Before launching any training or checkpoint-producing eval:

1. Query WandB for candidate baselines with `wandb_tool.py`.
2. If the user did not name a baseline, report top candidates and ask once.
3. Never quote baseline metrics from memory, markdown, chat history, or stale
   SQLite rows.
4. Verify the notes file has all required sections.
5. Apply the WandB tag hygiene rule before Slurm/GPU launch.
6. Create and register an event manifest.
7. Prefer launching through `experiment_event.py launch`.
8. Watch startup logs until WandB init succeeds or a run id appears.
9. If WandB init fails, cancel the Slurm job.  Do not let untracked training or
   checkpoint-producing eval continue.
10. Once the WandB run id appears, add it to the manifest, re-register the
    manifest, and sync WandB into SQLite.

Before launching a standalone `offline_eval`, `simuleval`, or `stream_eval`
readout:

1. Create and register an event manifest.
2. Record exact input data, glossary, checkpoint/model, launcher, output, and
   log paths.
3. Prefer W&B logging when it is cheap and stable, but do not require it for the
   eval to run.
4. If W&B logging fails after verified output artifacts already exist, do not
   discard the eval.  Mark W&B as missing/partial/failed in the manifest, then
   validate `eval_results.tsv`, logs, row counts, and scoring inputs directly.

WandB sync after init:

```bash
python documents/code/general/wandb_tool.py --project <project> db-sync --runs <run_id>
```

Launch wrapper:

```bash
python documents/code/general/experiment_event.py launch <manifest.json> -- \
  sbatch --parsable <launcher.sh>
```

## 7. WandB Tag Hygiene

Before launching or registering an experiment, validate every structured W&B tag
after prefixing, e.g. `family:*`, `task:*`, `data:*`, `variant:*`, and extra
tags. W&B tags must be `1..64` characters. Keep long descriptive names in
`WANDB_EXP_NAME`, notes, manifest metadata, or artifact paths, not in
`DATA_TAG`/`VARIANT_TAG`.

If a tag is too long, fail before Slurm/GPU launch; do not wait for
`wandb.init`.

## 8. Required WandB Schema

Every training script, checkpoint-producing script, and eval script that opts
into W&B should initialize WandB with:

```python
wandb.init(
    project="<qwen3_rag|sst_omni|simuleval_eval>",
    name=f"{family}__{variant}__{YYYYMMDD-HHMM}__{key_hp}",
    config=vars(args),
    tags=[
        f"family:{family}",
        f"task:{task}",
        f"data:{data_tag}",
        "status:running",
    ],
    notes=open(args.notes_file).read(),
    save_code=True,
)
wandb.config.update({"baseline_run_ids": args.baseline_run_ids})
```

At run end for W&B-tracked runs:

- set `run.summary["verdict"]`
- replace `status:running` with `status:success` or `status:failed`
- for W&B-tracked eval runs, set `run.config["trained_from_run"]`
- sync the run with best bundles

## 9. Required Notes Sections

Every notes file must include non-empty:

```text
## Hypothesis
## Background / Motivation
## What changed vs baseline
## Expected metrics
## Verdict
```

Notes are not a metrics database.  Do not hand-enter final metric values into
notes as the authoritative record.  Metrics live in WandB.

After a run finishes:

1. Fill `## Verdict`.
2. Push verdict to WandB notes/summary.
3. Flip the `status:*` tag to `success` or `failed`.
4. Refresh SQLite:

```bash
python documents/code/general/wandb_tool.py --project <project> db-sync --runs <run_id> --best-bundles
```

## 10. Metric Reporting Discipline

Cross-run metric tables must use at-best-step reads of WandB history, not raw
`run.summary[...]` loop-logged values.

Use:

```bash
python documents/code/general/wandb_tool.py --project <project> compare <run_ids...> \
  --preset retriever_eval retriever_train \
  --at-best-step \
  --anchor-metric both
```

or:

```bash
python documents/code/general/wandb_tool.py --project <project> db-compare <run_ids...> \
  --refresh \
  --anchor-metric both
```

Every cross-run table must include both bundles:

```text
primary   = best/step
secondary = best_secondary/step
```

Allowed direct summary keys:

```text
best/metric_value
best/step
best_secondary/metric_value
best_secondary/step
verdict
verdict_metrics
```

Forbidden direct summary keys:

```text
eval_acl6060/*
eval_dev/*
by_paper/*
train/*
train/tcm_pos_viol_rate
train/tcm_neg_viol_rate
train/loss*
```

Rule of thumb: if a key is logged inside the training loop, raw
`run.summary[key]` is a last-step snapshot, not a best-checkpoint metric.

Running runs may appear in comparisons only if explicitly labelled as running
and not final.

## 11. Data Leakage And Calibration Constraints

ACL is held-out test/readout data.  Do not use ACL to choose:

- tau
- model checkpoint
- threshold
- hyperparameters
- variant winner

For tau/model selection:

1. Use dev calibration first.
2. Freeze the selection rule.
3. Report ACL only as held-out readout.

If recall retention matters, use explicit recall-retention constraints rather
than F-score-only ranking.

When the user asks for reviewer-defensible calibration, state the rule before
showing held-out results.

## 12. Query And Backfill Behavior

When asked:

```text
Which script/data/eval belongs to run X?
Which evals descend from model X?
What files were used for this run?
```

First query:

```bash
python documents/code/general/experiment_event.py files <run_id-or-event_id>
python documents/code/general/experiment_event.py show <run_id-or-event_id>
python documents/code/general/wandb_tool.py --project <project> db-show <run_id> --config --notes
```

If no manifest exists, say provenance is incomplete.  Then backfill a manifest
from evidence:

- WandB config
- WandB notes
- launcher references
- Slurm logs
- filesystem paths
- Git history when needed

Label backfilled facts as evidence-derived.  Do not present guesses as verified
lineage.

## 13. Human And Agent Division Of Labor

The human owns:

- canonical run decisions
- deprecated run decisions
- reviewer-facing protocol
- experiment family taxonomy
- final judgment about which results matter

The agent owns:

- manifest creation and updates
- launcher hash capture
- command capture
- Slurm job id capture
- WandB run id linking
- artifact path extraction
- SQLite registration and sync
- lineage queries
- orphan launcher and missing-manifest audits

The agent should not freely create new directory trees.  New files should go
under fixed module folders:

```text
launchers/YYYY/MM/
notes/YYYY/MM/
manifests/YYYY/MM/
reports/
archive/
```

## 14. Legacy Runs

A run is historical debt if it has no WandB record or lacks required schema.
Do not use historical-debt runs as authoritative baselines unless the user
explicitly asks for a legacy comparison and the limitations are stated.

For legacy recovery:

1. Read WandB `git_commit` and config if available.
2. Locate likely launcher and logs from filesystem/Git evidence.
3. Create a backfill event manifest.
4. Register the manifest.
5. Sync WandB if a run id exists.
6. Mark unresolved fields as `null` or `unknown`, not guessed.

## 15. RAG Policy

Do not build or trust RAG as the first layer of experiment tracking.

RAG may sit on top of the structured ledger, but answers must return stable
identifiers and paths:

```text
event_id
wandb_run_id
launcher_path
artifact path
manifest path
```

If RAG retrieves a plausible script without a manifest or DB link, treat it as
a candidate only and verify before use.

## 16. Practical Defaults

For new retriever training under `documents/code/train/term_train/`:

- create notes under `notes/YYYY/MM/`
- create launcher under `launchers/YYYY/MM/`
- create manifest under `manifests/YYYY/MM/`
- include `qwen3_glossary_neg_train.py` as `train_script`
- include common launcher body as `common_launcher` when sourced
- include all train/dev/ACL/medicine JSONL files as artifacts
- include all eval glossaries as artifacts
- include output checkpoint dir and Slurm log dir
- register manifest before launch
- re-register manifest after WandB run id appears

For new data-prep under `documents/code/data_pre/<module>/`:

- include raw input dirs/files
- include source scripts
- include output JSONL/glossary/audio/index dirs
- include any filler or exclusion glossary
- include downstream training manifest links if known


## 17. Fail-Fast And No Silent Fallback

Default behavior must be fail-fast.  Do not silently fallback.

If data, files, paths, configs, checkpoints, metrics, manifests, WandB runs, or
intermediate artifacts are missing, malformed, inconsistent, or ambiguous, the
script or agent must either:

1. raise a clear error,
2. explicitly filter the affected samples and report the count/reason, or
3. use a fallback only when the fallback strategy is explicitly enabled and
   documented in the launcher, WandB config, logs, and event manifest.

Silent fallback is forbidden.

Examples of forbidden behavior:

```text
missing audio -> use dummy audio silently
missing MFA span -> use full-chunk MaxSim silently
missing checkpoint -> use latest checkpoint silently
missing metric -> use raw summary or another metric silently
missing glossary -> use another glossary silently
invalid JSONL row -> skip silently
path not found -> try another path silently
WandB init failure -> continue training locally

If filtering is used, the script must log at least:

```text
total rows
kept rows
dropped rows
drop rate
drop reasons

If fallback is used, it must be explicitly named, enabled by a flag or config,
and recorded in:

```text
launcher command
WandB config
event manifest metadata
logs
final report or verdict

Agents must not invent or apply fallback policies on their own.  They may
propose a fallback or filtering rule, but should not apply it unless the user
approves it or it is already declared in the manifest/launcher.

## 18. Hard Do Not Rules

Do not:

- quote final metrics from chat memory
- quote cross-run metrics from raw `run.summary[...]`
- hand-write metric truth into markdown
- manually edit SQLite metric tables
- launch training or checkpoint-producing eval without WandB
- let training or checkpoint-producing eval continue after WandB init failure
- treat standalone eval W&B failure as metric failure when validated TSV/log
  artifacts already exist
- create a new folder for each ablation
- rely on notes alone for provenance
- use ACL to select tau/checkpoint/hyperparameters
- guess which launcher produced a run
- mix data paths from similar launchers without manifest evidence
- silently fallback to another file, checkpoint, glossary, metric, or path
- silently use dummy data, dummy labels, or dummy metrics
- silently skip malformed samples without counters and reasons
- silently downgrade eval/training behavior when required inputs are missing


## 19. Minimal Checklist Before Answering Experiment Questions

Before answering a factual experiment question:

1. Identify whether the question is about metrics, provenance, or interpretation.
2. For training/checkpoint metrics, use WandB CLI at-best-step reads; for
   standalone eval metrics, use verified TSV/log artifacts linked from the
   manifest.
3. For provenance, use event manifests and `experiment_event.py`.
4. For interpretation, cite the rule or manifest evidence used.
5. If evidence is missing, say so and propose a backfill.

## 20. Long-Running Job Safety Rule

For any long-running data-prep, training, eval, simuleval, or analysis job,
never run it in the foreground of an interactive shell.

Default launch policy:

1. Prefer Slurm `sbatch` or `experiment_event.py launch`.
2. If Slurm is not used, launch with `setsid` or equivalent detached execution.
3. Always redirect stdout/stderr to persistent log files.
4. Always write a PID file or Slurm job id.
5. Always run from a stable cross-node absolute path.
6. Never treat `.tmp`, partial shard files, or missing stats as successful output.
7. Training must not start until data-prep outputs are atomically finalized and
   diagnostics/stats pass.

Forbidden:

- running long jobs directly in foreground
- relying on an interactive SSH / VSCode / Codex terminal to keep jobs alive
- using plain `cmd &` without detach, logs, and pid tracking
- continuing from `.tmp` files as if they were completed shards
- starting training before data validation succeeds

If a detached job is required, use a pattern like:

```bash
setsid bash -lc '<command>' > <log>.out 2> <log>.err < /dev/null &
echo $! > <log>.pid
```

Memory may guide workflow defaults, but metrics/status/provenance must be verified from WandB, event manifests, SQLite, filesystem, or Slurm before being reported.

## 21. Data Storage Path Defaults

For new generated datasets, indexes, checkpoints, eval outputs, transfer
packages, or other large experiment artifacts, prefer storing data under:

```text
/mnt/gemini/data1
```

Do not default large outputs to the repo tree, `$HOME`, `/tmp`, or the root
filesystem unless the user explicitly requests it or the artifact is small and
repo-local by design.

Before launching storage-heavy data-prep, training, eval, or packaging jobs,
check available space with:

```bash
df -h
```

If `/mnt/gemini/data1` is close to full or unsuitable for the expected output,
use `df -h` to choose a different mounted storage path with enough free space.
Record the chosen path in the launcher, event manifest artifacts, notes, and
logs where applicable.  Do not silently switch storage roots without making the
choice visible in provenance.

## 22. Completion Notification Rule

When a Codex task or substantive reply is complete, schedule a short delayed
notification immediately before the final in-chat response whenever a real
notification channel is configured.  The delay gives the chat UI time to render
the final result before the user clicks through from Slack.

Do not rely on Codex Slack connector self-DMs as the primary notification path:
the connector may post as the user's own Slack account, and Slack usually does
not generate a push notification for messages authored by the same user.

Preferred notification path:

```text
Slack Incoming Webhook or Slack bot posting as a non-user identity to a
dedicated notification channel, for example #codex-notify.
```

The message should be short and actionable: state that Codex finished, name the
task, and include the most important status or artifact path when useful.  On
this host, prefer a detached 5-10 second delay; the default delay target is 8
seconds:

```bash
~/bin/codex-notify --delay 8 --detach "Codex finished: <task/status/artifact>"
```

When working from a repository or VS Code Remote-SSH workspace, include the
workspace path so Slack renders a one-click editor link:

```bash
~/bin/codex-notify --delay 8 --detach --workspace "$PWD" "Codex finished: <task/status/artifact>"
```

The helper reads its webhook from `~/.config/codex_notify.env`, which should stay
outside the repo and should not be printed back into chat or logs.

If only the Slack connector is available, it may still be used as a best-effort
activity log, but it should be labelled as non-guaranteed for push
notifications.  If notification send fails, do not block the task; mention the
failure briefly in the final response.

For detached Slurm or background jobs that need notification after Codex is no
longer active, use a job-local Slack webhook or another detached notifier,
because the Codex Slack connector only works while Codex is running.
