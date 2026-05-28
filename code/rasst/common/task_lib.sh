#!/usr/bin/env bash
set -euo pipefail

TASK_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${TASK_LIB_DIR}/env.sh"

RASST_DRY_RUN=0
RASST_WRAPPER_ARGS=()

parse_common_args() {
  while (($#)); do
    case "$1" in
      --dry-run)
        RASST_DRY_RUN=1
        shift
        ;;
      --)
        shift
        RASST_WRAPPER_ARGS+=("$@")
        break
        ;;
      *)
        RASST_WRAPPER_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

legacy_path() {
  local rel_path="$1"
  local abs_path="${RASST_LEGACY_CODE_ROOT}/${rel_path}"
  if [[ ! -e "${abs_path}" ]]; then
    printf 'Missing legacy target: %s\n' "${abs_path}" >&2
    return 2
  fi
  printf '%s\n' "${abs_path}"
}

rasst_code_path() {
  local rel_path="$1"
  local abs_path="${RASST_ROOT}/code/rasst/${rel_path}"
  if [[ ! -e "${abs_path}" ]]; then
    printf 'Missing RASST code target: %s\n' "${abs_path}" >&2
    return 2
  fi
  printf '%s\n' "${abs_path}"
}

shell_join() {
  local out=""
  local part
  for part in "$@"; do
    printf -v part '%q' "${part}"
    out+="${part} "
  done
  printf '%s\n' "${out% }"
}

run_or_dry_run() {
  local task_name="$1"
  local cwd="$2"
  local log_slug="$3"
  shift 3

  local command_line
  command_line="$(shell_join "$@")"

  printf 'task=%s\n' "${task_name}"
  printf 'cwd=%s\n' "${cwd}"
  printf 'RASST_ROOT=%s\n' "${RASST_ROOT}"
  printf 'RASST_DATA_ROOT=%s\n' "${RASST_DATA_ROOT}"
  printf 'RASST_OUTPUT_ROOT=%s\n' "${RASST_OUTPUT_ROOT}"
  printf 'RASST_CHECKPOINT_ROOT=%s\n' "${RASST_CHECKPOINT_ROOT}"
  printf 'EVAL_TMPDIR=%s\n' "${EVAL_TMPDIR}"
  printf 'command=%s\n' "${command_line}"

  if [[ "${RASST_DRY_RUN}" == "1" || "${RASST_ALLOW_LAUNCH:-0}" != "1" ]]; then
    printf 'status=dry_run_only\n'
    printf 'Set RASST_ALLOW_LAUNCH=1 and omit --dry-run to launch detached.\n'
    return 0
  fi

  local stamp log_dir out_log err_log pid_file quoted_cwd
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log_dir="${RASST_LOG_ROOT}/curated"
  mkdir -p "${log_dir}"
  out_log="${log_dir}/${stamp}__${log_slug}.out"
  err_log="${log_dir}/${stamp}__${log_slug}.err"
  pid_file="${log_dir}/${stamp}__${log_slug}.pid"
  printf -v quoted_cwd '%q' "${cwd}"

  setsid bash -lc "cd ${quoted_cwd} && ${command_line}" >"${out_log}" 2>"${err_log}" < /dev/null &
  printf '%s\n' "$!" >"${pid_file}"
  printf 'status=launched_detached\n'
  printf 'pid_file=%s\n' "${pid_file}"
  printf 'stdout=%s\n' "${out_log}"
  printf 'stderr=%s\n' "${err_log}"
}
