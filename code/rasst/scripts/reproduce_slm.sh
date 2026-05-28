#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${SCRIPT_DIR}/../common/env.sh"

MANIFEST="${RASST_SLM_MANIFEST:-${RASST_ROOT}/code/rasst/manifests/slm_training.cap16_denoise_budget_ttag.json}"
LANG_SELECT="all"
STAGE_SELECT="all"
DO_LAUNCH=0

usage() {
  cat <<'EOF'
Usage: bash code/rasst/scripts/reproduce_slm.sh [OPTIONS]

Options:
  --lang de|ja|zh|all        Language to reproduce. Default: all.
  --stage prepare|train|all  Reproduction stage. Default: all.
  --manifest PATH            Override the SLM reproduction manifest.
  --launch                   Detach-launch selected steps. Requires RASST_ALLOW_LAUNCH=1.
  --dry-run                  Print commands only. This is the default.
  -h, --help                 Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --lang)
      LANG_SELECT="${2:?missing --lang value}"
      shift 2
      ;;
    --stage)
      STAGE_SELECT="${2:?missing --stage value}"
      shift 2
      ;;
    --manifest)
      MANIFEST="${2:?missing --manifest value}"
      shift 2
      ;;
    --launch)
      DO_LAUNCH=1
      shift
      ;;
    --dry-run)
      DO_LAUNCH=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${LANG_SELECT}" != "all" && "${LANG_SELECT}" != "de" && "${LANG_SELECT}" != "ja" && "${LANG_SELECT}" != "zh" ]]; then
  printf 'Invalid --lang: %s\n' "${LANG_SELECT}" >&2
  exit 2
fi
if [[ "${STAGE_SELECT}" != "all" && "${STAGE_SELECT}" != "prepare" && "${STAGE_SELECT}" != "train" ]]; then
  printf 'Invalid --stage: %s\n' "${STAGE_SELECT}" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  printf 'Missing manifest: %s\n' "${MANIFEST}" >&2
  exit 3
fi

mapfile -t SELECTED_STEPS < <(
  python3 - "${MANIFEST}" "${RASST_ROOT}" "${LANG_SELECT}" "${STAGE_SELECT}" <<'PY'
import base64
import json
import shlex
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
root = Path(sys.argv[2])
lang_select = sys.argv[3]
stage_select = sys.argv[4]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
code_root = Path(manifest.get("code_root", str(root / "code/rasst")))
legacy_root = Path(manifest.get("legacy_root", str(root / "code/legacy")))

langs = ["de", "ja", "zh"] if lang_select == "all" else [lang_select]
stages = ["prepare", "train"] if stage_select == "all" else [stage_select]

def expand(value: str) -> str:
    return value.format(root=str(root), code_root=str(code_root), legacy_root=str(legacy_root))

def abs_from_code(path_text: str) -> Path:
    path = Path(expand(path_text))
    return path if path.is_absolute() else code_root / path

for stage in stages:
    for lang in langs:
        try:
            spec = manifest["languages"][lang][stage]
        except KeyError as exc:
            raise SystemExit(f"Missing manifest entry for {lang}/{stage}") from exc
        launcher = abs_from_code(spec["launcher"])
        if not launcher.exists():
            raise SystemExit(f"Missing launcher for {lang}/{stage}: {launcher}")
        env = {
            "ROOT_DIR_OVERRIDE": str(code_root),
            "RASST_ROOT": str(root),
            "RASST_ACTIVE_CODE_ROOT": str(code_root),
        }
        env.update({key: expand(str(value)) for key, value in spec.get("env", {}).items()})
        env_text = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
        command = f"{env_text} bash {shlex.quote(str(launcher))}"
        cwd = str(code_root)
        log_slug = f"reproduce_slm_{lang}_{stage}"
        encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
        print("\t".join([lang, stage, cwd, log_slug, encoded]))
PY
)

if ((${#SELECTED_STEPS[@]} == 0)); then
  echo "No SLM reproduction steps selected." >&2
  exit 4
fi

printf 'manifest=%s\n' "${MANIFEST}"
printf 'selected_steps=%s\n' "${#SELECTED_STEPS[@]}"
printf 'launch_requested=%s\n' "${DO_LAUNCH}"

for row in "${SELECTED_STEPS[@]}"; do
  IFS=$'\t' read -r lang stage cwd log_slug encoded_command <<< "${row}"
  command="$(
    python3 - "${encoded_command}" <<'PY'
import base64
import sys
print(base64.b64decode(sys.argv[1]).decode("utf-8"))
PY
  )"

  printf '\n[SLM] lang=%s stage=%s\n' "${lang}" "${stage}"
  printf 'cwd=%s\n' "${cwd}"
  printf 'command=%s\n' "${command}"

  if [[ "${DO_LAUNCH}" != "1" || "${RASST_ALLOW_LAUNCH:-0}" != "1" ]]; then
    printf 'status=dry_run_only\n'
    printf 'Set RASST_ALLOW_LAUNCH=1 and pass --launch to detach-launch this step.\n'
    continue
  fi

  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log_dir="${RASST_LOG_ROOT}/curated/slm_reproduction"
  mkdir -p "${log_dir}"
  out_log="${log_dir}/${stamp}__${log_slug}.out"
  err_log="${log_dir}/${stamp}__${log_slug}.err"
  pid_file="${log_dir}/${stamp}__${log_slug}.pid"
  printf -v quoted_cwd '%q' "${cwd}"
  setsid bash -lc "cd ${quoted_cwd} && ${command}" >"${out_log}" 2>"${err_log}" < /dev/null &
  printf '%s\n' "$!" >"${pid_file}"
  printf 'status=launched_detached\n'
  printf 'pid_file=%s\n' "${pid_file}"
  printf 'stdout=%s\n' "${out_log}"
  printf 'stderr=%s\n' "${err_log}"
done
