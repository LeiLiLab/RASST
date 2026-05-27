# Curated RASST Wrappers

These wrappers provide a stable public path interface over the frozen legacy code. They do not rewrite legacy launchers. Each wrapper resolves the selected legacy target from `RASST_LEGACY_CODE_ROOT`, prints the command in `--dry-run`, and only launches when `RASST_ALLOW_LAUNCH=1`.

Override any default target by setting the matching environment variable, for example:

```bash
RASST_ACL_EVAL_TARGET=documents/code/simuleval/launchers/2026/05/example.sh \
  bash code/rasst/scripts/eval_acl.sh --dry-run
```
