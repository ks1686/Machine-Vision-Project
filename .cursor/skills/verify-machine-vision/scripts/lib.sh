#!/bin/bash
# Shared helpers for verify-machine-vision. Source this file. Do not execute it.
# Run state is a list of key=value lines. Never source or eval it.

LIB_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return 1
GUARD="$LIB_DIR/guard.py"
SPAWN_PID=
SPAWN_LSTART=
SPAWN_CMD=
SPAWN_UID=
RUN_STATE=
RUN_EVIDENCE=
RUN_ID=
STATUS=
FEATURE=
CREATED_AT=
APP_PID=
APP_LSTART=
APP_COMMAND=
APP_UID=
APP_STATUS=0
CAP_X=
CAP_Y=
CAP_W=
CAP_H=

if [ -z "${VMV_REAL_HOME:-}" ]; then
  VMV_REAL_HOME=${HOME:-}
fi
if [ -z "${VMV_ORIGINAL_PATH:-}" ]; then
  VMV_ORIGINAL_PATH=${PATH:-}
fi

die() {
  printf 'error: %s\n' "$1" >&2
  exit "${2:-1}"
}

trim() {
  local value="$1"
  value=${value#"${value%%[![:space:]]*}"}
  value=${value%"${value##*[![:space:]]}"}
  printf '%s' "$value"
}

valid_run_id() {
  local id="$1"
  case "$id" in
    ''|.|..) return 1 ;;
  esac
  case "$id" in
    *[!A-Za-z0-9_-]*) return 1 ;;
  esac
  return 0
}

our_scratch() {
  case "$1" in
    /tmp/vmvrun.*|/tmp/vmvevd.*|/private/tmp/vmvrun.*|/private/tmp/vmvevd.*)
      return 0
      ;;
  esac
  return 1
}

require_supported_platform() {
  local os
  os=$(uname -s) || return 1
  case "$os" in
    Linux|Darwin) return 0 ;;
  esac
  printf 'error: unsupported platform: %s\n' "$os" >&2
  return 1
}

read_one_line() {
  local file="$1" line
  [ -f "$file" ] || return 1
  [ -L "$file" ] && return 1
  IFS= read -r line < "$file" || return 1
  printf '%s' "$line"
}

parse_run_env() {
  local file="$1" line key value
  RUN_ID=
  STATUS=
  FEATURE=
  CREATED_AT=
  APP_PID=
  APP_LSTART=
  APP_COMMAND=
  APP_UID=
  [ -f "$file" ] || return 1
  [ -L "$file" ] && return 1
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    case "$line" in
      *=*) ;;
      *) return 1 ;;
    esac
    key=${line%%=*}
    value=${line#*=}
    case "$key" in
      RUN_ID|STATUS|FEATURE|CREATED_AT|APP_PID|APP_LSTART|APP_COMMAND|APP_UID) ;;
      *) continue ;;
    esac
    case "$value" in
      *'$'*|*\`*|*'('*|*')'*|*';'*|*'|'*|*'&'*|*'<'*|*'>'*)
        return 1
        ;;
    esac
    case "$key" in
      RUN_ID) RUN_ID=$value ;;
      STATUS) STATUS=$value ;;
      FEATURE) FEATURE=$value ;;
      CREATED_AT) CREATED_AT=$value ;;
      APP_PID) APP_PID=$value ;;
      APP_LSTART) APP_LSTART=$value ;;
      APP_COMMAND) APP_COMMAND=$value ;;
      APP_UID) APP_UID=$value ;;
    esac
  done < "$file"
  [ -n "$RUN_ID" ] || return 1
  valid_run_id "$RUN_ID" || return 1
}

write_env_line() {
  local dest="$1" key="$2" value="$3"
  case "$value" in
    *$'\n'*|*'$'*|*\`*|*'('*|*')'*|*';'*|*'|'*|*'&'*|*'<'*|*'>'*)
      return 1
      ;;
  esac
  printf '%s=%s\n' "$key" "$value" >> "$dest"
}

write_run_env() {
  local dest="$1" tmp
  tmp="${dest}.tmp.$$"
  : > "$tmp" || return 1
  chmod 600 "$tmp" || return 1
  write_env_line "$tmp" RUN_ID "$RUN_ID" || return 1
  write_env_line "$tmp" STATUS "$STATUS" || return 1
  write_env_line "$tmp" FEATURE "$FEATURE" || return 1
  write_env_line "$tmp" CREATED_AT "$CREATED_AT" || return 1
  write_env_line "$tmp" APP_PID "$APP_PID" || return 1
  write_env_line "$tmp" APP_LSTART "$APP_LSTART" || return 1
  write_env_line "$tmp" APP_COMMAND "$APP_COMMAND" || return 1
  write_env_line "$tmp" APP_UID "$APP_UID" || return 1
  mv -f "$tmp" "$dest" || return 1
  chmod 600 "$dest" || return 1
}

load_run_env() {
  parse_run_env "$RUN_STATE/run.env" || return 1
}

ps_one() {
  local field="$1" pid="$2" out
  [ -n "$pid" ] || return 1
  if [ "$field" = "lstart" ]; then
    out=$(TZ=UTC0 LC_ALL=C ps -o lstart= -p "$pid") || return 1
  else
    out=$(ps -ww -o "$field"= -p "$pid") || return 1
  fi
  case "$out" in
    *$'\n'*) return 1 ;;
  esac
  out=$(trim "$out") || return 1
  [ -n "$out" ] || return 1
  printf '%s' "$out"
}

ps_lstart() { ps_one lstart "$1"; }
ps_command() { ps_one command "$1"; }
ps_uid() { ps_one uid "$1"; }
ps_pgid() { ps_one pgid "$1"; }

ps_self_ok() {
  ps -o pid= -p $$ >/dev/null 2>&1
}

# The process group comes from the live ps row. Nothing in run.env is a pgid.
signal_live_group() {
  local pid="$1" pgid self_pgid
  pgid=$(ps_pgid "$pid") || return 1
  self_pgid=$(ps_pgid $$) || return 1
  [ "$pgid" = "$pid" ] || return 1
  [ "$pgid" != "$self_pgid" ] || return 1
  case "$pgid" in
    ''|*[!0-9]*|0|1) return 1 ;;
  esac
  kill -TERM -- "-$pgid" || return 1
}

stop_recorded_process() {
  local pid="$1" exp_lstart="$2" exp_cmd="$3" exp_uid="$4"
  local lstart cmd uid
  ps_self_ok || return 1
  if ! lstart=$(ps_lstart "$pid"); then
    return 0
  fi
  cmd=$(ps_command "$pid") || return 1
  uid=$(ps_uid "$pid") || return 1
  [ "$lstart" = "$exp_lstart" ] || return 1
  [ "$cmd" = "$exp_cmd" ] || return 1
  [ "$uid" = "$exp_uid" ] || return 1
  signal_live_group "$pid" || return 1
}

signal_verified_pid() {
  local pid="$1" exp_lstart="$2" exp_cmd="$3" exp_uid="$4"
  local lstart cmd uid
  ps_self_ok || return 1
  lstart=$(ps_lstart "$pid") || return 1
  cmd=$(ps_command "$pid") || return 1
  uid=$(ps_uid "$pid") || return 1
  [ "$lstart" = "$exp_lstart" ] || return 1
  [ "$cmd" = "$exp_cmd" ] || return 1
  [ "$uid" = "$exp_uid" ] || return 1
  case "$pid" in
    ''|*[!0-9]*|0|1) return 1 ;;
  esac
  kill -TERM -- "$pid" || return 1
}

settle_spawn() {
  local pid="${SPAWN_PID:-}" last="" cmd tries=50
  [ -n "$pid" ] || return 1
  while [ "$tries" -gt 0 ]; do
    if ! cmd=$(ps_command "$pid"); then
      return 1
    fi
    if [ -n "$last" ] && [ "$cmd" = "$last" ]; then
      SPAWN_CMD=$cmd
      SPAWN_LSTART=$(ps_lstart "$pid") || return 1
      SPAWN_UID=$(ps_uid "$pid") || return 1
      return 0
    fi
    last=$cmd
    tries=$((tries - 1))
    sleep 0.05
  done
  return 1
}

spawn_setsid() {
  [ "$#" -gt 0 ] || return 1
  python3 -c 'import os, sys
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])' "$@" &
  SPAWN_PID=$!
  if ! settle_spawn; then
    abort_spawn || true
    return 1
  fi
}

abort_spawn() {
  local pid="${SPAWN_PID:-}"
  [ -n "$pid" ] || return 0
  if [ -z "${SPAWN_LSTART:-}" ]; then
    SPAWN_LSTART=$(ps_lstart "$pid") || return 1
    SPAWN_CMD=$(ps_command "$pid") || return 1
    SPAWN_UID=$(ps_uid "$pid") || return 1
  fi
  stop_recorded_process "$pid" "$SPAWN_LSTART" "$SPAWN_CMD" "$SPAWN_UID"
}

guard_side_effects() {
  local log="$1" base arg
  shift
  [ "$#" -gt 0 ] || return 0
  base=$(basename -- "$1") || return 1
  case "$base" in
    sendmail|mail|mutt|msmtp|deploy)
      printf '%s\n' "$base" >> "$log" || return 1
      return 86
      ;;
  esac
  if [ "$base" = "git" ] && [ "${2:-}" = "push" ]; then
    printf '%s\n' "git push" >> "$log" || return 1
    return 86
  fi
  for arg in "$@"; do
    case "$arg" in
      tel:*|sms:*|mailto:*|facetime:*)
        printf '%s\n' "$arg" >> "$log" || return 1
        return 86
        ;;
    esac
  done
  return 0
}

save_cache_line() {
  local file="$1" value="$2"
  [ -n "$value" ] || return 0
  printf '%s\n' "$value" > "$file" || return 1
  chmod 600 "$file" || return 1
}

bake_tool_cache() {
  local state="$1" real_home="$2" value
  value=${GOMODCACHE:-}
  if [ -z "$value" ] && [ -d "$real_home/go/pkg/mod" ]; then
    value="$real_home/go/pkg/mod"
  fi
  save_cache_line "$state/cache/GOMODCACHE" "$value" || return 1
  value=${GOENV:-}
  if [ -z "$value" ] && [ -e "$real_home/.config/go/env" ]; then
    value="$real_home/.config/go/env"
  fi
  save_cache_line "$state/cache/GOENV" "$value" || return 1
  value=${PLAYWRIGHT_BROWSERS_PATH:-}
  if [ -z "$value" ] && [ -d "$real_home/.cache/ms-playwright" ]; then
    value="$real_home/.cache/ms-playwright"
  fi
  save_cache_line "$state/cache/PLAYWRIGHT_BROWSERS_PATH" "$value" || return 1
  value=${NPM_CONFIG_CACHE:-${npm_config_cache:-}}
  if [ -z "$value" ] && [ -d "$real_home/.npm" ]; then
    value="$real_home/.npm"
  fi
  save_cache_line "$state/cache/NPM_CONFIG_CACHE" "$value" || return 1
  value=${UV_CACHE_DIR:-}
  if [ -z "$value" ] && [ -d "$real_home/.cache/uv" ]; then
    value="$real_home/.cache/uv"
  fi
  save_cache_line "$state/cache/UV_CACHE_DIR" "$value" || return 1
  value=${XAUTHORITY:-}
  if [ -z "$value" ] && [ -f "$real_home/.Xauthority" ]; then
    value="$real_home/.Xauthority"
  fi
  save_cache_line "$state/cache/XAUTHORITY" "$value" || return 1
}

apply_isolated_env() {
  local state="$1" value name
  export HOME="$state/home"
  export XDG_RUNTIME_DIR="$state/runtime"
  export GOTELEMETRY=off
  export PYTHONUNBUFFERED=1
  for name in GOMODCACHE GOENV PLAYWRIGHT_BROWSERS_PATH NPM_CONFIG_CACHE UV_CACHE_DIR XAUTHORITY; do
    if value=$(read_one_line "$state/cache/$name"); then
      export "$name=$value"
    fi
  done
  if value=$(read_one_line "$state/cache/NPM_CONFIG_CACHE"); then
    export npm_config_cache="$value"
  fi
  if [ "$(uname -s)" = "Darwin" ]; then
    export CFFIXED_USER_HOME="$HOME"
  fi
  if [ "$(uname -s)" = "Linux" ] && [ -n "${DISPLAY:-}" ]; then
    export QT_QPA_PLATFORM=xcb
  fi
  export PATH="$state/bin:${VMV_ORIGINAL_PATH:-$PATH}"
}

copy_git_ls_files() {
  local repo="$1" dest="$2" file dir list status=0
  list="$dest/.vmv-ls-files"
  git -C "$repo" ls-files -z > "$list" || {
    rm -f "$list"
    return 1
  }
  while IFS= read -r -d '' file; do
    [ -n "$file" ] || continue
    dir=$(dirname -- "$file") || status=1
    if [ "$status" -ne 0 ]; then
      break
    fi
    mkdir -p "$dest/$dir" || status=1
    if [ "$status" -ne 0 ]; then
      break
    fi
    if [ -L "$repo/$file" ]; then
      cp -P "$repo/$file" "$dest/$file" || status=1
    else
      cp -p "$repo/$file" "$dest/$file" || status=1
    fi
    if [ "$status" -ne 0 ]; then
      break
    fi
  done < "$list"
  rm -f "$list"
  return "$status"
}

new_run_id() {
  local rand id
  rand=$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n') || return 1
  id="$(date -u +%Y%m%dT%H%M%SZ)-$$-${rand}"
  valid_run_id "$id" || return 1
  printf '%s\n' "$id"
}

assert_private_or_drop() {
  local dir="$1"
  our_scratch "$dir" || return 1
  if ! python3 "$GUARD" private-dir "$dir" >/dev/null; then
    rm -rf -- "$dir"
    return 1
  fi
}

locate_run() {
  local id="$1"
  valid_run_id "$id" || return 1
  RUN_STATE=$(python3 "$GUARD" find-run "vmvrun." "$id") || return 1
  RUN_EVIDENCE=$(python3 "$GUARD" find-run "vmvevd." "$id") || return 1
  our_scratch "$RUN_STATE" || return 1
  our_scratch "$RUN_EVIDENCE" || return 1
}

record_app_identity() {
  load_run_env || return 1
  APP_PID=$SPAWN_PID
  APP_LSTART=$SPAWN_LSTART
  APP_COMMAND=$SPAWN_CMD
  APP_UID=$SPAWN_UID
  STATUS=running
  write_run_env "$RUN_STATE/run.env" || return 1
}

clear_app_identity() {
  load_run_env || return 1
  APP_PID=
  APP_LSTART=
  APP_COMMAND=
  APP_UID=
  STATUS=ready
  write_run_env "$RUN_STATE/run.env" || return 1
}

publish_file() {
  local name="$1" src="$2"
  python3 "$GUARD" write-evidence "$RUN_EVIDENCE" "$name" < "$src" >/dev/null || return 1
}

exact_window_id() {
  local title="$1" wid name ids rest line
  command -v xdotool >/dev/null 2>&1 || return 1
  ids=$(xdotool search --name "$title" 2>/dev/null) || return 1
  [ -n "$ids" ] || return 1
  rest=$ids
  while [ -n "$rest" ]; do
    line=${rest%%$'\n'*}
    if [ "$line" = "$rest" ]; then
      rest=
    else
      rest=${rest#*$'\n'}
    fi
    wid=$line
    [ -n "$wid" ] || continue
    name=$(xdotool getwindowname "$wid" 2>/dev/null) || continue
    if [ "$name" = "$title" ]; then
      printf '%s\n' "$wid"
      return 0
    fi
  done
  return 1
}

parse_geometry() {
  local geom="$1" line key value rest
  CAP_X=
  CAP_Y=
  CAP_W=
  CAP_H=
  rest=$geom
  while [ -n "$rest" ]; do
    line=${rest%%$'\n'*}
    if [ "$line" = "$rest" ]; then
      rest=
    else
      rest=${rest#*$'\n'}
    fi
    key=${line%%=*}
    value=${line#*=}
    case "$value" in
      ''|*[!0-9]*) return 1 ;;
    esac
    case "$key" in
      X) CAP_X=$value ;;
      Y) CAP_Y=$value ;;
      WIDTH) CAP_W=$value ;;
      HEIGHT) CAP_H=$value ;;
    esac
  done
  [ -n "$CAP_X" ] && [ -n "$CAP_Y" ] && [ -n "$CAP_W" ] && [ -n "$CAP_H" ]
}

screen_size() {
  local dims
  if command -v xdpyinfo >/dev/null 2>&1; then
    dims=$(xdpyinfo 2>/dev/null | awk '/dimensions:/ { print $2; exit }') || true
  fi
  case "$dims" in
    [0-9]*x[0-9]*)
      SCREEN_W=${dims%%x*}
      SCREEN_H=${dims#*x}
      ;;
    *)
      SCREEN_W=1920
      SCREEN_H=1200
      ;;
  esac
}

clamp_capture() {
  local right bottom
  screen_size || return 1
  case "$CAP_X$CAP_Y$CAP_W$CAP_H" in
    *[!0-9]*) return 1 ;;
  esac
  if [ "$CAP_X" -ge "$SCREEN_W" ] || [ "$CAP_Y" -ge "$SCREEN_H" ]; then
    return 1
  fi
  right=$((CAP_X + CAP_W))
  bottom=$((CAP_Y + CAP_H))
  if [ "$right" -gt "$SCREEN_W" ]; then
    CAP_W=$((SCREEN_W - CAP_X))
  fi
  if [ "$bottom" -gt "$SCREEN_H" ]; then
    CAP_H=$((SCREEN_H - CAP_Y))
  fi
  [ "$CAP_W" -gt 0 ] && [ "$CAP_H" -gt 0 ]
}

release_stuck_keys() {
  command -v xdotool >/dev/null 2>&1 || return 0
  [ -n "${DISPLAY:-}" ] || return 0
  xdotool keyup space >/dev/null 2>&1 || true
}

capture_window() {
  local title="$1" name="$2" wid geom raw
  release_stuck_keys || true
  wid=$(exact_window_id "$title") || return 1
  xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || true
  xdotool windowmove "$wid" 40 40 >/dev/null 2>&1 || true
  xdotool windowsize "$wid" 960 540 >/dev/null 2>&1 || true
  sleep 0.4
  geom=$(xdotool getwindowgeometry --shell "$wid") || return 1
  parse_geometry "$geom" || return 1
  clamp_capture || return 1
  raw="$RUN_STATE/scratch/$name"
  ffmpeg -y -loglevel error -f x11grab -video_size "${CAP_W}x${CAP_H}" -i "${DISPLAY}+${CAP_X},${CAP_Y}" -frames:v 1 "$raw" || return 1
  publish_file "$name" "$raw" || return 1
}

wait_for_window() {
  local title="$1" pid="$2" tries="$3" wid
  while [ "$tries" -gt 0 ]; do
    if ! ps_lstart "$pid" >/dev/null 2>&1; then
      return 1
    fi
    if wid=$(exact_window_id "$title"); then
      if [ -n "$wid" ]; then
        printf '%s\n' "$wid"
        return 0
      fi
    fi
    sleep 0.5
    tries=$((tries - 1))
  done
  return 1
}

wait_for_exit() {
  local pid="$1" seconds="$2" elapsed=0
  while [ "$elapsed" -lt "$seconds" ]; do
    if ! ps_lstart "$pid" >/dev/null 2>&1; then
      set +e
      wait "$pid"
      APP_STATUS=$?
      set -e
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  return 1
}

join_if_any() {
  if [ "$#" -eq 0 ]; then
    return 0
  fi
  printf '%s\n' "$@"
}
