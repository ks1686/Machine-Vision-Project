#!/usr/bin/env bash
# Proves the verification harness checks. Run it from any directory.
# It does not use sudo, does not read a hard-coded home, and removes its scratch.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
CLI="$SCRIPT_DIR/verify-machine-vision"
GUARD="$SCRIPT_DIR/guard.py"
LIB="$SCRIPT_DIR/lib.sh"
SKILL_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd) || exit 1
# shellcheck source=lib.sh
. "$LIB"

REAL_PATH=${PATH:-}
REAL_HOME=${HOME:-}
REAL_PS=$(command -v ps) || exit 1
cd / || exit 1

REGRESS=$(mktemp -d /tmp/vmv-regress.XXXXXXXX) || exit 1
mkdir -m 700 "$REGRESS/bin" "$REGRESS/pid" || exit 1
: > "$REGRESS/dirs"

note_dir() {
  printf '%s\n' "$1" >> "$REGRESS/dirs"
}

note_spawn() {
  local dir="$REGRESS/pid/$SPAWN_PID"
  printf '%s\n' "$SPAWN_LSTART" > "$dir.lstart" || return 1
  printf '%s\n' "$SPAWN_UID" > "$dir.uid" || return 1
  printf '%s\n' "$SPAWN_CMD" > "$dir.cmd" || return 1
}

scratch_prefix() {
  case "$1" in
    /tmp/vmvrun.*|/tmp/vmvevd.*|/tmp/vmv-regress.*|/private/tmp/vmvrun.*|/private/tmp/vmvevd.*|/private/tmp/vmv-regress.*)
      return 0
      ;;
  esac
  return 1
}

loosen() {
  local path="$1"
  python3 - "$path" << 'PY' || return 1
import os
import stat
import sys
def fix(path):
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        return
    os.chmod(path, stat.S_IMODE(info.st_mode) | 0o700)
    if stat.S_ISDIR(info.st_mode):
        for name in os.listdir(path):
            fix(os.path.join(path, name))
fix(sys.argv[1])
PY
}

remove_recorded() {
  local dir pid lstart cmd uid
  PATH=$REAL_PATH
  if [ -d "$REGRESS/pid" ]; then
    for pid in "$REGRESS/pid/"*.lstart; do
      [ -e "$pid" ] || continue
      pid=${pid##*/}
      pid=${pid%.lstart}
      lstart=$(cat "$REGRESS/pid/$pid.lstart") || continue
      uid=$(cat "$REGRESS/pid/$pid.uid") || continue
      cmd=$(cat "$REGRESS/pid/$pid.cmd") || continue
      stop_recorded_process "$pid" "$lstart" "$cmd" "$uid" || true
    done
  fi
  if [ -f "$REGRESS/dirs" ]; then
    while IFS= read -r dir; do
      [ -n "$dir" ] || continue
      scratch_prefix "$dir" || continue
      if [ -d "$dir" ]; then
        loosen "$dir" || true
        rm -rf -- "$dir"
      fi
    done < "$REGRESS/dirs"
  fi
}

list_leaks() {
  local base
  for base in /tmp /private/tmp; do
    [ -d "$base" ] || continue
    find "$base" -maxdepth 1 \( -name 'vmvrun.*' -o -name 'vmvevd.*' -o -name 'vmv-regress.*' \) -print
  done
}

finish() {
  local status=$?
  trap - EXIT
  PATH=$REAL_PATH
  if [ -n "${REGRESS:-}" ] && [ -d "$REGRESS" ]; then
    remove_recorded || true
    loosen "$REGRESS" || true
    rm -rf -- "$REGRESS"
  fi
  if [ "$status" -eq 0 ]; then
    if list_leaks | grep -q .; then
      printf 'FAIL leftover scratch in /tmp\n' >&2
      list_leaks >&2 || true
      exit 1
    fi
  fi
  exit "$status"
}
trap finish EXIT

fail() {
  printf 'fail: %s\n' "$1" >&2
  return 1
}

capture() {
  CAP_STATUS=0
  CAP_OUT=$("$@" 2>&1) && CAP_STATUS=0 || CAP_STATUS=$?
}

field() {
  printf '%s\n' "$1" | sed -n "s/^$2=//p"
}

make_repo() {
  local repo="$1"
  git init -q "$repo" || return 1
  printf 'tracked\n' > "$repo/tracked.txt" || return 1
  printf 'junk\n' > "$repo/junk.txt" || return 1
  mkdir -p "$repo/node_modules" || return 1
  printf 'x\n' > "$repo/node_modules/a.js" || return 1
  git -C "$repo" add tracked.txt || return 1
  git -C "$repo" -c user.email=vmv@example.com -c user.name=vmv commit -q -m init || return 1
}

do_launch() {
  local out
  out=$("$CLI" launch --repo "$REPO") || return 1
  LAST_ID=$(field "$out" run_id) || return 1
  LAST_STATE=$(field "$out" state) || return 1
  LAST_EVIDENCE=$(field "$out" evidence) || return 1
  [ -n "$LAST_ID" ] && [ -n "$LAST_STATE" ] && [ -n "$LAST_EVIDENCE" ] || return 1
  note_dir "$LAST_STATE"
  note_dir "$LAST_EVIDENCE"
}

mode_of() {
  python3 "$GUARD" stat-mode "$1" || return 1
}

run_test() {
  local name="$1"
  shift
  if ! "$@"; then
    printf 'FAIL %s\n' "$name" >&2
    exit 1
  fi
  printf 'ok %s\n' "$name"
}

test_unsupported_platform() {
  local bin marker
  bin="$REGRESS/fake-platform"
  mkdir -p "$bin" || return 1
  marker="$REGRESS/mktemp-called"
  cat > "$bin/uname" << 'EOF' || return 1
#!/bin/sh
echo Haiku
EOF
  cat > "$bin/mktemp" << EOF || return 1
#!/bin/sh
touch "$marker"
exit 1
EOF
  chmod +x "$bin/uname" "$bin/mktemp" || return 1
  capture env PATH="$bin:$REAL_PATH" "$CLI" launch --repo "$REPO"
  [ "$CAP_STATUS" -ne 0 ] || fail "unsupported platform returned 0"
  [ ! -e "$marker" ] || fail "mktemp ran on an unsupported platform"
}

test_private_dirs() {
  local saved out mode
  saved=$(umask)
  umask 0 || return 1
  do_launch || { umask "$saved"; return 1; }
  umask "$saved" || return 1
  mode=$(mode_of "$LAST_STATE") || return 1
  case "$mode" in
    700\ *) ;;
    *) fail "state mode $mode" ;;
  esac
  mode=$(mode_of "$LAST_EVIDENCE") || return 1
  case "$mode" in
    700\ *) ;;
    *) fail "evidence mode $mode" ;;
  esac
  bin="$REGRESS/chmod-spy"
  mkdir -p "$bin" || return 1
  cat > "$bin/chmod" << EOF || return 1
#!/bin/sh
printf '%s\n' "\$*" >> "$REGRESS/chmod.log"
exec /bin/chmod "\$@"
EOF
  chmod +x "$bin/chmod" || return 1
  : > "$REGRESS/chmod.log" || return 1
  capture env PATH="$bin:$REAL_PATH" "$CLI" launch --repo "$REPO"
  [ "$CAP_STATUS" -eq 0 ] || fail "spy launch failed: $CAP_OUT"
  SPY_STATE=$(field "$CAP_OUT" state) || return 1
  SPY_EVIDENCE=$(field "$CAP_OUT" evidence) || return 1
  note_dir "$SPY_STATE"
  note_dir "$SPY_EVIDENCE"
  if grep -F -x -e "$SPY_STATE" -e "700 $SPY_STATE" -e "0700 $SPY_STATE" "$REGRESS/chmod.log"; then
    fail "launch chmod the state directory"
  fi
  open=$(mktemp -d /tmp/vmv-regress.XXXXXXXX) || return 1
  note_dir "$open"
  chmod 0777 "$open" || return 1
  capture "$CLI" check-private "$open"
  [ "$CAP_STATUS" -ne 0 ] || fail "accepted a group writable dir"
  mode=$(mode_of "$open") || return 1
  case "$mode" in
    777\ *) ;;
    *) fail "check-private changed mode to $mode" ;;
  esac
  before=$(mode_of /etc) || return 1
  capture "$CLI" check-private /etc
  [ "$CAP_STATUS" -ne 0 ] || fail "accepted /etc"
  after=$(mode_of /etc) || return 1
  [ "$before" = "$after" ] || fail "check-private changed /etc"
}

test_run_ids() {
  local id
  for id in '.' '..' 'a/b' 'a.b' '' 'a b'; do
    capture "$CLI" check-id "$id"
    [ "$CAP_STATUS" -ne 0 ] || fail "accepted id $id"
  done
  do_launch || return 1
  FIRST=$LAST_ID
  do_launch || return 1
  [ "$FIRST" != "$LAST_ID" ] || fail "run ids were not unique"
  sleep 30 &
  SLEEP_PID=$!
  capture "$CLI" cleanup
  [ "$CAP_STATUS" -ne 0 ] || fail "cleanup without an id returned 0"
  if printf '%s\n' "$CAP_OUT" | grep -F -x -q -- cleaned; then
    fail "cleanup without an id said cleaned"
  fi
  if ! "$REAL_PS" -p "$SLEEP_PID" >/dev/null; then
    fail "cleanup without an id killed a process"
  fi
  kill "$SLEEP_PID" || true
  wait "$SLEEP_PID" || true
  do_launch || return 1
  KEEP_STATE=$LAST_STATE
  capture "$CLI" cleanup --id "missing000"
  [ "$CAP_STATUS" -ne 0 ] || fail "cleanup guessed an id"
  [ -d "$KEEP_STATE" ] || fail "cleanup removed a different run"
  if printf '%s\n' "$CAP_OUT" | grep -F -x -q -- cleaned; then
    fail "wrong id said cleaned"
  fi
}

test_run_env_and_status() {
  local sentinel home_out
  do_launch || return 1
  sentinel="$REGRESS/sentinel"
  printf 'HOME=$(touch %s)\nUV=/tmp/evil\n' "$sentinel" >> "$LAST_STATE/run.env" || return 1
  capture "$CLI" run --id "$LAST_ID" -- printenv HOME
  [ "$CAP_STATUS" -eq 0 ] || fail "printenv failed: $CAP_OUT"
  [ "$CAP_OUT" = "$LAST_STATE/home" ] || fail "HOME came from run.env: $CAP_OUT"
  [ ! -e "$sentinel" ] || fail "run.env command substitution ran"
  capture "$CLI" run --id "$LAST_ID" -- printenv PATH
  [ "$CAP_STATUS" -eq 0 ] || fail "PATH print failed"
  case "$CAP_OUT" in
    "$LAST_STATE/bin:"*) ;;
    *) fail "tool path was not derived from the state dir" ;;
  esac
  case "$CAP_OUT" in
    */tmp/evil*) fail "UV from run.env leaked into PATH" ;;
  esac
  capture "$CLI" run --id "$LAST_ID" -- false
  [ "$CAP_STATUS" -eq 1 ] || fail "false returned $CAP_STATUS"
  capture "$CLI" run --id "$LAST_ID" -- bash -c 'exit 3'
  [ "$CAP_STATUS" -eq 3 ] || fail "exit 3 returned $CAP_STATUS"
}

test_side_effects() {
  local bin sentinel log
  do_launch || return 1
  bin="$REGRESS/side-bin"
  mkdir -p "$bin" || return 1
  sentinel="$REGRESS/side-sentinel"
  cat > "$bin/sendmail" << EOF || return 1
#!/bin/sh
touch "$sentinel"
EOF
  chmod +x "$bin/sendmail" || return 1
  capture env PATH="$bin:$REAL_PATH" "$CLI" run --id "$LAST_ID" -- sendmail -t
  [ "$CAP_STATUS" -eq 86 ] || fail "sendmail returned $CAP_STATUS"
  capture env PATH="$bin:$REAL_PATH" "$CLI" run --id "$LAST_ID" -- true tel:123
  [ "$CAP_STATUS" -eq 86 ] || fail "tel returned $CAP_STATUS"
  capture env PATH="$bin:$REAL_PATH" "$CLI" run --id "$LAST_ID" -- git push origin main
  [ "$CAP_STATUS" -eq 86 ] || fail "git push returned $CAP_STATUS"
  [ ! -e "$sentinel" ] || fail "side effect command ran"
  log=$(cat "$LAST_STATE/side-effects.log") || return 1
  [ "$log" = $'sendmail\ntel:123\ngit push' ] || fail "side effect order: $log"
}

test_evidence_and_paths() {
  local outside mode
  do_launch || return 1
  ln -s "$LAST_EVIDENCE/real.txt" "$LAST_EVIDENCE/link.txt" || return 1
  capture bash -c 'printf hi | "$1" evidence --id "$2" --name link.txt' bash "$CLI" "$LAST_ID"
  [ "$CAP_STATUS" -ne 0 ] || fail "followed an evidence symlink"
  ln -s "$LAST_EVIDENCE/missing" "$LAST_EVIDENCE/dangling.txt" || return 1
  capture bash -c 'printf hi | "$1" evidence --id "$2" --name dangling.txt' bash "$CLI" "$LAST_ID"
  [ "$CAP_STATUS" -ne 0 ] || fail "accepted a dangling symlink"
  [ -L "$LAST_EVIDENCE/dangling.txt" ] || fail "dangling symlink was replaced"
  capture bash -c 'printf hi | "$1" evidence --id "$2" --name run.env' bash "$CLI" "$LAST_ID"
  [ "$CAP_STATUS" -ne 0 ] || fail "reserved name was accepted"
  capture bash -c 'printf hi | "$1" evidence --id "$2" --name ../escape.txt' bash "$CLI" "$LAST_ID"
  [ "$CAP_STATUS" -ne 0 ] || fail "parent escape was accepted"
  outside=$(mktemp -d /tmp/vmv-regress.XXXXXXXX) || return 1
  note_dir "$outside"
  ln -s "$outside" "$LAST_EVIDENCE/jump" || return 1
  capture bash -c 'printf hi | "$1" evidence --id "$2" --name jump' bash "$CLI" "$LAST_ID"
  [ "$CAP_STATUS" -ne 0 ] || fail "symlink swap was accepted"
  [ ! -e "$outside/jump" ] || fail "write escaped through the symlink"
  saved=$(umask)
  umask 0 || return 1
  capture bash -c 'printf hello | "$1" evidence --id "$2" --name note.txt' bash "$CLI" "$LAST_ID"
  umask "$saved" || return 1
  [ "$CAP_STATUS" -eq 0 ] || fail "evidence write failed: $CAP_OUT"
  mode=$(mode_of "$LAST_EVIDENCE/note.txt") || return 1
  case "$mode" in
    600\ *) ;;
    *) fail "evidence mode $mode" ;;
  esac
  capture bash -c 'printf hello | "$1" evidence --id "$2" --name note.txt' bash "$CLI" "$LAST_ID"
  [ "$CAP_STATUS" -ne 0 ] || fail "second evidence write was accepted"
  root=$(mktemp -d /tmp/vmv-regress.XXXXXXXX) || return 1
  note_dir "$root"
  capture python3 "$GUARD" scratch-ok "$root"
  [ "$CAP_STATUS" -eq 0 ] || fail "/tmp prefix refused"
  capture python3 "$GUARD" scratch-ok "/private/tmp/vmv-example"
  [ "$CAP_STATUS" -eq 0 ] || fail "/private/tmp prefix refused"
  capture python3 "$GUARD" scratch-ok /etc
  [ "$CAP_STATUS" -ne 0 ] || fail "/etc prefix accepted"
  ln -s /etc "$root/etc-link" || return 1
  capture python3 "$GUARD" scratch-ok "$root/etc-link"
  [ "$CAP_STATUS" -ne 0 ] || fail "realpath did not reject a symlink escape"
  capture python3 "$GUARD" inside "$root" "$root/child.txt"
  [ "$CAP_STATUS" -eq 0 ] || fail "child of a private dir was refused"
  capture python3 "$GUARD" inside "$root" /etc/passwd
  [ "$CAP_STATUS" -ne 0 ] || fail "inside accepted /etc"
  capture python3 "$GUARD" inside "$root" "$root/etc-link"
  [ "$CAP_STATUS" -ne 0 ] || fail "inside accepted a symlink"
}

test_hash_snapshot_explicit() {
  local dir locked out
  dir=$(mktemp -d /tmp/vmv-regress.XXXXXXXX) || return 1
  note_dir "$dir"
  printf 'secret\n' > "$dir/secret" || return 1
  chmod 000 "$dir/secret" || return 1
  capture bash -c 'set -euo pipefail; "$1" hash --root "$2"; echo NEXT' bash "$CLI" "$dir"
  [ "$CAP_STATUS" -ne 0 ] || fail "hash of an unreadable file returned 0"
  case "$CAP_OUT" in
    *NEXT*) fail "hash failure continued" ;;
  esac
  chmod 600 "$dir/secret" || return 1
  mkdir "$dir/locked" || return 1
  chmod 000 "$dir/locked" || return 1
  capture bash -c 'set -euo pipefail; "$1" hash --root "$2"; echo NEXT' bash "$CLI" "$dir"
  [ "$CAP_STATUS" -ne 0 ] || fail "hash of an unreadable dir returned 0"
  case "$CAP_OUT" in
    *NEXT*) fail "dir hash failure continued" ;;
  esac
  chmod 000 "$dir/locked" || return 1
  capture bash -c 'set -euo pipefail; "$1" snapshot --root "$2" --dest "$3"; echo NEXT' bash "$CLI" "$dir" "$dir/snap.txt"
  chmod 700 "$dir/locked" || return 1
  [ "$CAP_STATUS" -ne 0 ] || fail "snapshot of an unreadable dir returned 0"
  case "$CAP_OUT" in
    *NEXT*) fail "snapshot failure continued" ;;
  esac
  chmod 700 "$dir/locked" || return 1
  left="$REGRESS/left.txt"
  right="$REGRESS/right.txt"
  printf 'same\n' > "$left" || return 1
  printf 'same\n' > "$right" || return 1
  chmod 000 "$right" || return 1
  capture "$CLI" compare "$left" "$right"
  [ "$CAP_STATUS" -ne 0 ] || fail "compare of an unreadable file returned 0"
  case "$CAP_OUT" in
    *unchanged*) fail "unreadable compare said unchanged" ;;
  esac
  case "$CAP_OUT" in
    *unreadable*) ;;
    *) fail "unreadable compare did not say unreadable: $CAP_OUT" ;;
  esac
  chmod 600 "$right" || return 1
}

test_process_identity() {
  local lstart other cmd uid
  spawn_setsid sleep 30 || return 1
  note_spawn || return 1
  lstart=$(ps_lstart "$SPAWN_PID") || return 1
  other=$(TZ=America/New_York LC_ALL=C "$REAL_PS" -o lstart= -p "$SPAWN_PID") || return 1
  other=$(trim "$other") || return 1
  [ "$lstart" != "$other" ] || fail "UTC lstart matched local time"
  direct=$(TZ=UTC0 LC_ALL=C "$REAL_PS" -o lstart= -p "$SPAWN_PID") || return 1
  direct=$(trim "$direct") || return 1
  [ "$lstart" = "$direct" ] || fail "lstart was not TZ=UTC0 LC_ALL=C"
  cmd=$(ps_command "$SPAWN_PID") || return 1
  uid=$(ps_uid "$SPAWN_PID") || return 1
  stop_recorded_process "$SPAWN_PID" "wrong" "$cmd" "$uid" && fail "wrong lstart was signaled"
  if ! "$REAL_PS" -p "$SPAWN_PID" >/dev/null; then
    fail "wrong lstart killed the process"
  fi
  bin="$REGRESS/fake-ps"
  mkdir -p "$bin" || return 1
  cat > "$bin/ps" << 'EOF' || return 1
#!/bin/sh
exit 1
EOF
  chmod +x "$bin/ps" || return 1
  PATH="$bin:$REAL_PATH" stop_recorded_process "$SPAWN_PID" "$lstart" "$cmd" "$uid" && fail "ps failure was not closed"
  PATH=$REAL_PATH
  if ! "$REAL_PS" -p "$SPAWN_PID" >/dev/null; then
    fail "ps failure still signaled"
  fi
  if ! "$REAL_PS" -p 1 >/dev/null; then
    fail "pid 1 is not alive"
  fi
  printf '1\n' > "$REGRESS/PGID" || return 1
  stop_recorded_process "$SPAWN_PID" "$lstart" "$cmd" "$uid" || fail "live group kill failed"
  if "$REAL_PS" -p "$SPAWN_PID" >/dev/null; then
    fail "group kill left the process"
  fi
  if ! "$REAL_PS" -p 1 >/dev/null; then
    fail "group kill signaled pid 1"
  fi
  [ "$(cat "$REGRESS/PGID")" = "1" ] || fail "kill rewrote the poisoned file"
  do_launch || return 1
  spawn_setsid sleep 30 || return 1
  note_spawn || return 1
  RUN_STATE=$LAST_STATE
  load_run_env || return 1
  APP_PID=$SPAWN_PID
  APP_LSTART=$SPAWN_LSTART
  APP_COMMAND=$SPAWN_CMD
  APP_UID=$SPAWN_UID
  write_run_env "$RUN_STATE/run.env" || return 1
  bin="$REGRESS/fake-ps"
  mkdir -p "$bin" || return 1
  cat > "$bin/ps" << 'EOF' || return 1
#!/bin/sh
exit 1
EOF
  chmod +x "$bin/ps" || return 1
  capture env PATH="$bin:$REAL_PATH" "$CLI" cleanup --id "$LAST_ID"
  [ "$CAP_STATUS" -ne 0 ] || fail "ps failure during cleanup returned 0"
  [ -d "$LAST_STATE" ] || fail "ps failure during cleanup removed state"
  if printf '%s\n' "$CAP_OUT" | grep -F -x -q -- cleaned; then
    fail "ps failure during cleanup said cleaned"
  fi
  if ! "$REAL_PS" -p "$SPAWN_PID" >/dev/null; then
    fail "ps failure during cleanup killed the process"
  fi
  PATH=$REAL_PATH
  abort_spawn || return 1
  spawn_setsid sleep 30 || return 1
  note_spawn || return 1
  abort_spawn || fail "failed launch did not kill the child"
  if "$REAL_PS" -p "$SPAWN_PID" >/dev/null; then
    fail "abort left the child running"
  fi
}

test_cleanup_identity() {
  local lstart cmd uid out
  do_launch || return 1
  [ -f "$LAST_STATE/handlers.before" ] || fail "handlers were not snapshotted at launch"
  spawn_setsid sleep 30 || return 1
  note_spawn || return 1
  lstart=$(ps_lstart "$SPAWN_PID") || return 1
  cmd=$(ps_command "$SPAWN_PID") || return 1
  uid=$(ps_uid "$SPAWN_PID") || return 1
  RUN_STATE=$LAST_STATE
  load_run_env || return 1
  APP_PID=$SPAWN_PID
  APP_LSTART=wrong
  APP_COMMAND=$cmd
  APP_UID=$uid
  write_run_env "$RUN_STATE/run.env" || return 1
  capture "$CLI" cleanup --id "$LAST_ID"
  [ "$CAP_STATUS" -ne 0 ] || fail "identity mismatch returned 0"
  [ -d "$LAST_STATE" ] || fail "identity mismatch removed state"
  if printf '%s\n' "$CAP_OUT" | grep -F -x -q -- cleaned; then
    fail "identity mismatch said cleaned"
  fi
  case "$CAP_OUT" in
    *recovery:*) ;;
    *) fail "identity mismatch omitted recovery: $CAP_OUT" ;;
  esac
  if ! "$REAL_PS" -p "$SPAWN_PID" >/dev/null; then
    fail "identity mismatch killed the process"
  fi
  abort_spawn || return 1
  printf 'tampered\n' >> "$LAST_STATE/handlers.before" || return 1
  capture "$CLI" cleanup --id "$LAST_ID"
  [ "$CAP_STATUS" -ne 0 ] || fail "tampered handlers were cleaned"
  [ -d "$LAST_STATE" ] || fail "tampered handlers removed state"
  if printf '%s\n' "$CAP_OUT" | grep -F -x -q -- cleaned; then
    fail "tampered handlers said cleaned"
  fi
  do_launch || return 1
  spawn_setsid sleep 30 || return 1
  note_spawn || return 1
  RUN_STATE=$LAST_STATE
  load_run_env || return 1
  APP_PID=$SPAWN_PID
  APP_LSTART=$SPAWN_LSTART
  APP_COMMAND=$SPAWN_CMD
  APP_UID=$SPAWN_UID
  write_run_env "$RUN_STATE/run.env" || return 1
  mkdir "$LAST_STATE/scratch/ro" || return 1
  chmod 0555 "$LAST_STATE/scratch/ro" || return 1
  EVIDENCE=$LAST_EVIDENCE
  capture "$CLI" cleanup --id "$LAST_ID"
  [ "$CAP_STATUS" -eq 0 ] || fail "cleanup failed: $CAP_OUT"
  if "$REAL_PS" -p "$SPAWN_PID" >/dev/null; then
    fail "cleanup left the recorded process"
  fi
  [ ! -d "$LAST_STATE" ] || fail "cleanup left state"
  [ -d "$EVIDENCE" ] || fail "cleanup removed evidence"
  if ! printf '%s\n' "$CAP_OUT" | grep -F -x -q -- cleaned; then
    fail "cleanup did not say cleaned: $CAP_OUT"
  fi
}

test_git_copy_and_isolation() {
  local sandbox home prefs before after
  do_launch || return 1
  sandbox="$LAST_STATE/sandbox"
  [ -f "$sandbox/tracked.txt" ] || fail "tracked file was not copied"
  [ ! -e "$sandbox/junk.txt" ] || fail "untracked file was copied"
  [ ! -e "$sandbox/node_modules" ] || fail "node_modules was copied"
  GOMODCACHE="$REGRESS/gomod"
  GOENV="$REGRESS/goenv"
  PLAYWRIGHT_BROWSERS_PATH="$REGRESS/pw"
  NPM_CONFIG_CACHE="$REGRESS/npm"
  UV_CACHE_DIR="$REGRESS/uv"
  mkdir -p "$GOMODCACHE" "$PLAYWRIGHT_BROWSERS_PATH" "$NPM_CONFIG_CACHE" "$UV_CACHE_DIR" || return 1
  printf 'go\n' > "$GOENV" || return 1
  export GOMODCACHE GOENV PLAYWRIGHT_BROWSERS_PATH NPM_CONFIG_CACHE UV_CACHE_DIR
  do_launch || return 1
  capture "$CLI" run --id "$LAST_ID" -- printenv GOMODCACHE
  [ "$CAP_OUT" = "$GOMODCACHE" ] || fail "GOMODCACHE was not passed through"
  capture "$CLI" run --id "$LAST_ID" -- printenv GOENV
  [ "$CAP_OUT" = "$GOENV" ] || fail "GOENV was not passed through"
  capture "$CLI" run --id "$LAST_ID" -- printenv PLAYWRIGHT_BROWSERS_PATH
  [ "$CAP_OUT" = "$PLAYWRIGHT_BROWSERS_PATH" ] || fail "PLAYWRIGHT_BROWSERS_PATH was not passed through"
  capture "$CLI" run --id "$LAST_ID" -- printenv NPM_CONFIG_CACHE
  [ "$CAP_OUT" = "$NPM_CONFIG_CACHE" ] || fail "NPM_CONFIG_CACHE was not passed through"
  capture "$CLI" run --id "$LAST_ID" -- printenv UV_CACHE_DIR
  [ "$CAP_OUT" = "$UV_CACHE_DIR" ] || fail "UV_CACHE_DIR was not passed through"
  capture "$CLI" run --id "$LAST_ID" -- printenv GOTELEMETRY
  [ "$CAP_OUT" = "off" ] || fail "GOTELEMETRY was $CAP_OUT"
  home="$REGRESS/real-home"
  mkdir -p "$home/Library/Preferences" || return 1
  prefs="$home/Library/Preferences/com.vmv.test.plist"
  printf 'original\n' > "$prefs" || return 1
  before=$(python3 "$GUARD" hash-tree "$home") || return 1
  bin="$REGRESS/darwin-uname"
  mkdir -p "$bin" || return 1
  cat > "$bin/uname" << 'EOF' || return 1
#!/bin/sh
echo Darwin
EOF
  chmod +x "$bin/uname" || return 1
  capture env HOME="$home" PATH="$bin:$REAL_PATH" "$CLI" launch --repo "$REPO"
  [ "$CAP_STATUS" -eq 0 ] || fail "darwin launch failed: $CAP_OUT"
  LAST_ID=$(field "$CAP_OUT" run_id) || return 1
  LAST_STATE=$(field "$CAP_OUT" state) || return 1
  LAST_EVIDENCE=$(field "$CAP_OUT" evidence) || return 1
  note_dir "$LAST_STATE"
  note_dir "$LAST_EVIDENCE"
  capture env HOME="$home" PATH="$bin:$REAL_PATH" "$CLI" run --id "$LAST_ID" -- printenv CFFIXED_USER_HOME
  [ "$CAP_OUT" = "$LAST_STATE/home" ] || fail "CFFIXED_USER_HOME was $CAP_OUT"
  capture env HOME="$home" PATH="$bin:$REAL_PATH" "$CLI" run --id "$LAST_ID" -- python3 -c 'import os; p=os.path.join(os.environ["HOME"], "Library", "Preferences", "com.vmv.test.plist"); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write("changed")'
  [ "$CAP_STATUS" -eq 0 ] || fail "isolated write failed: $CAP_OUT"
  after=$(python3 "$GUARD" hash-tree "$home") || return 1
  [ "$before" = "$after" ] || fail "isolated HOME wrote the real preferences"
  unset GOMODCACHE GOENV PLAYWRIGHT_BROWSERS_PATH NPM_CONFIG_CACHE UV_CACHE_DIR
}

test_watch_chmod_handlers() {
  local root dest out outside mode home pref snap1 snap2 snap3
  root=$(mktemp -d /tmp/vmv-regress.XXXXXXXX) || return 1
  note_dir "$root"
  printf 'a\n' > "$root/keep.txt" || return 1
  dest="$REGRESS/manifest.txt"
  python3 "$GUARD" snapshot-tree "$root" "$dest" || return 1
  printf 'finder\n' > "$root/.DS_Store" || return 1
  printf 'c\n' > "$root/new.txt" || return 1
  printf 'changed\n' > "$root/keep.txt" || return 1
  out="$REGRESS/watch.txt"
  python3 "$GUARD" watch-new "$root" "$dest" "$out" || return 1
  grep -F -x -q "keep.txt" "$out" || fail "watcher missed a changed file"
  grep -F -x -q "new.txt" "$out" || fail "watcher missed a new file"
  if grep -F -q ".DS_Store" "$out"; then
    fail "watcher reported .DS_Store"
  fi
  do_launch || return 1
  outside=$(mktemp -d /tmp/vmv-regress.XXXXXXXX) || return 1
  note_dir "$outside"
  chmod 0555 "$outside" || return 1
  capture "$CLI" chmod-inside --id "$LAST_ID" --path "$outside"
  [ "$CAP_STATUS" -eq 2 ] || fail "chmod-inside returned $CAP_STATUS"
  mode=$(mode_of "$outside") || return 1
  case "$mode" in
    555\ *) ;;
    *) fail "chmod-inside changed an outside dir to $mode" ;;
  esac
  chmod 700 "$outside" || return 1
  home="$REGRESS/darwin-home"
  mkdir -p "$home/Library/Preferences" || return 1
  pref="$home/Library/Preferences/com.vmv.handler.plist"
  printf 'one\n' > "$pref" || return 1
  snap1="$REGRESS/h1"
  snap2="$REGRESS/h2"
  snap3="$REGRESS/h3"
  VMV_PLATFORM=Darwin python3 "$GUARD" handlers-snapshot "$snap1" "$home" || return 1
  grep -F -q "$pref" "$snap1" || fail "handler snapshot did not record the full path"
  VMV_PLATFORM=Darwin python3 "$GUARD" handlers-snapshot "$snap2" "$home" || return 1
  capture python3 "$GUARD" compare "$snap1" "$snap2"
  [ "$CAP_STATUS" -eq 0 ] || fail "handler compare failed"
  case "$CAP_OUT" in
    *unchanged*) ;;
    *) fail "matching handlers were not unchanged" ;;
  esac
  printf 'two\n' > "$pref" || return 1
  VMV_PLATFORM=Darwin python3 "$GUARD" handlers-snapshot "$snap3" "$home" || return 1
  capture python3 "$GUARD" compare "$snap1" "$snap3"
  [ "$CAP_STATUS" -eq 2 ] || fail "changed handlers returned $CAP_STATUS"
  case "$CAP_OUT" in
    *unchanged*) fail "changed handlers said unchanged" ;;
  esac
  chmod 000 "$pref" || return 1
  capture bash -c 'set -euo pipefail; VMV_PLATFORM=Darwin python3 "$1" handlers-snapshot "$2" "$3"; echo NEXT' bash "$GUARD" "$REGRESS/h4" "$home"
  [ "$CAP_STATUS" -ne 0 ] || fail "unreadable handler snapshot returned 0"
  case "$CAP_OUT" in
    *NEXT*) fail "unreadable handler snapshot continued" ;;
    *unchanged*) fail "unreadable handler snapshot said unchanged" ;;
  esac
  chmod 600 "$pref" || return 1
}

test_names_urls_and_die() {
  capture python3 "$GUARD" exact-name "Verify Machine Vision Probe" "Verify Machine Vision Probe"
  [ "$CAP_STATUS" -eq 0 ] || fail "exact name rejected itself"
  capture python3 "$GUARD" exact-name "Verify Machine Vision Probe" "Verify Machine Vision Probe extra"
  [ "$CAP_STATUS" -ne 0 ] || fail "exact name accepted a suffix"
  capture python3 "$GUARD" exact-name "Verify Machine Vision Probe" "Probe"
  [ "$CAP_STATUS" -ne 0 ] || fail "exact name accepted a fragment"
  capture python3 "$GUARD" exact-url-path "/notes" "http://localhost:3000/notes"
  [ "$CAP_STATUS" -eq 0 ] || fail "exact url rejected /notes"
  capture python3 "$GUARD" exact-url-path "/notes" "http://localhost:3000/notes/extra"
  [ "$CAP_STATUS" -ne 0 ] || fail "exact url accepted a longer path"
  capture python3 "$GUARD" exact-url-path "/notes" "http://localhost:3000/notes/"
  [ "$CAP_STATUS" -ne 0 ] || fail "exact url accepted a trailing slash"
  capture bash -c '
set -euo pipefail
helper_die() { echo boom >&2; return 1; }
f() {
  set -e
  x="$(helper_die)" || return 1
  echo survived
}
f
echo after
'
  [ "$CAP_STATUS" -ne 0 ] || fail "helper failure returned 0"
  case "$CAP_OUT" in
    *survived*|*after*) fail "helper failure continued: $CAP_OUT" ;;
  esac
  bash -c '
set -u
empty=()
. "$1"
join_if_any "${empty[@]+"${empty[@]}"}"
' bash "$LIB" || fail "empty array tripped set -u"
  if [ -n "${DISPLAY:-}" ] && command -v xdotool >/dev/null 2>&1 && python3 -c 'import tkinter' >/dev/null 2>&1; then
    python3 - << 'PY' > "$REGRESS/tk.pid" &
import tkinter as tk
root = tk.Tk()
root.title("Verify Machine Vision Probe")
root.geometry("240x120+40+40")
root.update_idletasks()
root.update()
print(root.winfo_id(), flush=True)
root.after(8000, root.destroy)
root.mainloop()
PY
    TK_PID=$!
    sleep 1
    rendered=""
    ids=$(xdotool search --name "Verify Machine Vision Probe" 2>/dev/null || true)
    rest=$ids
    while [ -n "$rest" ]; do
      line=${rest%%$'\n'*}
      if [ "$line" = "$rest" ]; then
        rest=
      else
        rest=${rest#*$'\n'}
      fi
      [ -n "$line" ] || continue
      got=$(xdotool getwindowname "$line" 2>/dev/null || true)
      if [ "$got" = "Verify Machine Vision Probe" ]; then
        rendered=$got
        break
      fi
    done
    kill "$TK_PID" >/dev/null 2>&1 || true
    wait "$TK_PID" || true
    [ "$rendered" = "Verify Machine Vision Probe" ] || fail "rendered window name was $rendered"
  fi
}

test_source_shape() {
  local token file
  token=$(printf '%s%s' 'map' 'file')
  if grep -F -q -- "$token" "$SCRIPT_DIR"/*; then
    fail "scripts use a forbidden line reader"
  fi
  token=$(printf '%s%s' 'read' 'array')
  if grep -F -q -- "$token" "$SCRIPT_DIR"/*; then
    fail "scripts use a forbidden array reader"
  fi
  token=$(printf '%s%s' 'declare ' '-A')
  if grep -F -q -- "$token" "$SCRIPT_DIR"/*; then
    fail "scripts use forbidden associative arrays"
  fi
  python3 - "$SKILL_DIR" << 'PY' || return 1
import pathlib
import re
import sys
root = pathlib.Path(sys.argv[1])
home = "/" + "home" + "/"
users = "/" + "Users" + "/"
blocked = "su" + "do" + " "
for path in root.rglob("*"):
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    if "\u2014" in text or "\u2013" in text:
        raise SystemExit("punctuation " + str(path))
    if re.search(r"\$\{[A-Za-z_][A-Za-z0-9_]*,,", text):
        raise SystemExit("lowercase expansion " + str(path))
    if home in text or users in text or blocked in text:
        raise SystemExit("forbidden " + str(path))
PY
}

REPO="$REGRESS/repo"
make_repo "$REPO" || {
  printf 'FAIL repo\n' >&2
  exit 1
}
note_dir "$REPO"

run_test unsupported-platform test_unsupported_platform
run_test private-dirs test_private_dirs
run_test run-ids test_run_ids
run_test run-env-and-status test_run_env_and_status
run_test side-effects test_side_effects
run_test evidence-and-paths test_evidence_and_paths
run_test hash-snapshot test_hash_snapshot_explicit
run_test process-identity test_process_identity
run_test cleanup-identity test_cleanup_identity
run_test git-copy-and-isolation test_git_copy_and_isolation
run_test watch-chmod-handlers test_watch_chmod_handlers
run_test names-urls-and-die test_names_urls_and_die
run_test source-shape test_source_shape

printf 'regression ok\n'
