#!/bin/bash
# Feature drives. Sourced by verify-machine-vision after lib.sh.

USER_ID=synth_a
WIN_REGISTER="Guided Continuous Capture"
WIN_AUTH="Capture Authentication Image"
WIN_MESH="synth_a_face.obj"

require_app() {
  [ -x "$RUN_STATE/bin/uv" ] || return 1
  [ -f "$RUN_STATE/sandbox/face_registration.py" ] || return 1
  [ -f "$RUN_STATE/sandbox/pyproject.toml" ] || return 1
}

require_gui() {
  [ -n "${DISPLAY:-}" ] || return 1
  command -v xdotool >/dev/null 2>&1 || return 1
  command -v ffmpeg >/dev/null 2>&1 || return 1
}

require_template() {
  [ -f "$RUN_STATE/sandbox/registered_faces/$USER_ID/${USER_ID}_avgface.png" ]
}

probe_darwin_camera() {
  local script status
  script="$RUN_STATE/scratch/avfoundation_probe.py"
  mkdir -p "$RUN_STATE/scratch" || return 1
  cat > "$script" << 'PY' || return 1
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
opened = bool(cap.isOpened())
ok = False
frame = None
if opened:
    ok, frame = cap.read()
cap.release()
if not opened or not ok or frame is None or getattr(frame, "ndim", 0) != 3:
    raise SystemExit(3)
if frame.shape[0] < 1 or frame.shape[1] < 1:
    raise SystemExit(3)
print("shape={0}".format(tuple(int(v) for v in frame.shape)))
PY
  chmod 600 "$script" || return 1
  : > "$RUN_STATE/scratch/avfoundation.shape" || return 1
  : > "$RUN_STATE/scratch/avfoundation.err" || return 1
  status=0
  (
    apply_isolated_env "$RUN_STATE" || exit 1
    unset LD_PRELOAD
    unset FAKEVIDEO_PPM
    unset FAKEVIDEO_DEVICE
    cd "$RUN_STATE/sandbox" || exit 1
    "$RUN_STATE/bin/uv" run --frozen --directory "$RUN_STATE/sandbox" python -u "$script"
  ) > "$RUN_STATE/scratch/avfoundation.shape" 2> "$RUN_STATE/scratch/avfoundation.err" || status=$?
  return "$status"
}

camera_on() {
  local ppm="$1" os status
  unset DYLD_INSERT_LIBRARIES
  unset LD_PRELOAD
  os=$(uname -s) || return 1
  case "$os" in
    Darwin)
      unset LD_PRELOAD
      unset FAKEVIDEO_PPM
      unset FAKEVIDEO_DEVICE
      status=0
      probe_darwin_camera || status=$?
      if [ "$status" -ne 0 ]; then
        printf 'error: Darwin has no /dev/video0 camera node and this harness will not inject a library\n' >&2
        return 3
      fi
      return 0
      ;;
    Linux)
      if [ -e /dev/video0 ]; then
        unset LD_PRELOAD
        unset FAKEVIDEO_PPM
        unset FAKEVIDEO_DEVICE
        return 0
      fi
      if [ ! -f "$RUN_STATE/bin/libfakevideo.so" ] || [ ! -f "$ppm" ]; then
        printf 'error: camera shim missing\n' >&2
        return 1
      fi
      export LD_PRELOAD="$RUN_STATE/bin/libfakevideo.so"
      export FAKEVIDEO_PPM="$ppm"
      export FAKEVIDEO_DEVICE=/dev/video0
      return 0
      ;;
    *)
      printf 'error: unsupported platform: %s\n' "$os" >&2
      return 1
      ;;
  esac
}

camera_off() {
  unset LD_PRELOAD
  unset FAKEVIDEO_PPM
  unset FAKEVIDEO_DEVICE
  unset DYLD_INSERT_LIBRARIES
}

make_frames() {
  apply_isolated_env "$RUN_STATE" || return 1
  "$RUN_STATE/bin/uv" run --frozen --directory "$RUN_STATE/sandbox" python -u \
    "$LIB_DIR/synthetic_face.py" --kind frontal --ppm "$RUN_STATE/frames/frontal.ppm" || return 1
  "$RUN_STATE/bin/uv" run --frozen --directory "$RUN_STATE/sandbox" python -u \
    "$LIB_DIR/synthetic_face.py" --kind shifted --ppm "$RUN_STATE/frames/shifted.ppm" || return 1
}

run_py() {
  local log="$1"
  shift
  apply_isolated_env "$RUN_STATE" || return 1
  set +e
  (
    cd "$RUN_STATE/sandbox" || exit 1
    "$RUN_STATE/bin/uv" run --frozen --directory "$RUN_STATE/sandbox" python -u "$@"
  ) >"$log" 2>&1
  APP_STATUS=$?
  set -e
  printf 'exit_code=%s\n' "$APP_STATUS" >> "$log" || return 1
}

run_py_bg() {
  local log="$1"
  shift
  apply_isolated_env "$RUN_STATE" || return 1
  (
    cd "$RUN_STATE/sandbox" || exit 1
    exec python3 -c 'import os, sys
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])' \
      "$RUN_STATE/bin/uv" run --frozen --directory "$RUN_STATE/sandbox" python -u "$@"
  ) >"$log" 2>&1 &
  SPAWN_PID=$!
  if ! settle_spawn; then
    abort_spawn || true
    return 1
  fi
  record_app_identity || {
    abort_spawn || true
    return 1
  }
}

stop_app() {
  load_run_env || return 1
  if [ -z "${APP_PID:-}" ]; then
    return 0
  fi
  stop_recorded_process "$APP_PID" "$APP_LSTART" "$APP_COMMAND" "$APP_UID" || return 1
  set +e
  wait "$APP_PID"
  set -e
  clear_app_identity || return 1
}

finish_bg() {
  local log="$1" timeout="$2"
  if ! wait_for_exit "$SPAWN_PID" "$timeout"; then
    stop_app || true
    printf 'exit_code=timeout\n' >> "$log" || true
    return 1
  fi
  printf 'exit_code=%s\n' "$APP_STATUS" >> "$log" || return 1
  clear_app_identity || return 1
  return "$APP_STATUS"
}

send_space_until_exit() {
  local title="$1" pid="$2" tries="$3" wid
  release_stuck_keys || true
  while [ "$tries" -gt 0 ]; do
    if ! ps_lstart "$pid" >/dev/null 2>&1; then
      return 0
    fi
    if wid=$(exact_window_id "$title"); then
      if [ -n "$wid" ]; then
        xdotool windowactivate --sync "$wid" >/dev/null 2>&1 || true
        xdotool key --window "$wid" space >/dev/null 2>&1 || true
      fi
    fi
    sleep 0.4
    tries=$((tries - 1))
  done
  return 0
}

note_manifest() {
  local name="$1"
  python3 "$GUARD" snapshot-tree "$RUN_STATE/sandbox" "$RUN_STATE/$name" || return 1
}

publish_new_files() {
  local manifest="$1" capture="$2"
  python3 "$GUARD" watch-new "$RUN_STATE/sandbox" "$RUN_STATE/$manifest" "$RUN_STATE/scratch/new-files.txt" || return 1
  publish_file "$capture" "$RUN_STATE/scratch/new-files.txt" || return 1
  rm -f "$RUN_STATE/scratch/new-files.txt"
}

drive_register_face() {
  local log saved
  require_app || return 1
  require_gui || return 1
  make_frames || return 1
  note_manifest manifest.register || return 1
  camera_on "$RUN_STATE/frames/frontal.ppm" || return $?
  log="$RUN_STATE/scratch/register.log"
  run_py_bg "$log" face_registration.py "$USER_ID" || {
    camera_off
    return 1
  }
  if ! wait_for_window "$WIN_REGISTER" "$SPAWN_PID" 90 >/dev/null; then
    stop_app || true
    camera_off
    return 1
  fi
  sleep 1
  if ! capture_window "$WIN_REGISTER" "register-window.png"; then
    stop_app || true
    camera_off
    return 1
  fi
  if ! finish_bg "$log" 120; then
    publish_file "register-transcript.txt" "$log" || true
    camera_off
    return 1
  fi
  camera_off
  publish_file "register-transcript.txt" "$log" || return 1
  log_has() { grep -F -q -- "$2" "$1"; }
  log_has "$log" "Guided continuous capture complete." || return 1
  saved=$(sed -n 's/.*Total saved: \([0-9][0-9]*\).*/\1/p' "$log" | tail -n 1) || return 1
  [ -n "$saved" ] || return 1
  [ "$saved" -gt 0 ] || return 1
  log="$RUN_STATE/scratch/check-capture.log"
  run_py "$log" check_capture.py "$USER_ID" || return 1
  publish_file "check-capture.txt" "$log" || return 1
  [ "$APP_STATUS" -eq 0 ] || return "$APP_STATUS"
  grep -F -q -- "Passed" "$log" || return 1
  log="$RUN_STATE/scratch/average.log"
  run_py "$log" make_average_face.py "$USER_ID" || return 1
  publish_file "average-face.txt" "$log" || return 1
  [ "$APP_STATUS" -eq 0 ] || return "$APP_STATUS"
  grep -F -q -- "[average] wrote:" "$log" || return 1
  publish_new_files manifest.register register-new-files.txt || return 1
  grep -F -q -- "${USER_ID}_avgface.png" "$RUN_EVIDENCE/register-new-files.txt" || return 1
}

stop_key_helper() {
  local pid="$1" lstart="$2" cmd="$3" uid="$4"
  if [ -z "$pid" ]; then
    return 0
  fi
  if [ -z "$lstart" ] || [ -z "$cmd" ] || [ -z "$uid" ]; then
    if ! ps_lstart "$pid" >/dev/null 2>&1; then
      set +e
      wait "$pid"
      set -e
      return 0
    fi
    return 1
  fi
  reap_verified_pid "$pid" "$lstart" "$cmd" "$uid"
}

capture_one_auth() {
  local ppm="$1" transcript="$2" shot="$3" log key_pid key_lstart key_cmd key_uid
  release_stuck_keys || true
  camera_on "$ppm" || return $?
  log="$RUN_STATE/scratch/auth.log"
  : > "$log" || return 1
  run_py_bg "$log" authenticate.py "$USER_ID" || {
    camera_off
    return 1
  }
  if ! wait_for_window "$WIN_AUTH" "$SPAWN_PID" 80 >/dev/null; then
    stop_app || true
    camera_off
    return 1
  fi
  sleep 1
  if ! capture_window "$WIN_AUTH" "$shot"; then
    stop_app || true
    camera_off
    return 1
  fi
  send_space_until_exit "$WIN_AUTH" "$SPAWN_PID" 120 &
  key_pid=$!
  key_lstart=$(ps_lstart "$key_pid" 2>/dev/null) || key_lstart=
  key_cmd=$(ps_command "$key_pid" 2>/dev/null) || key_cmd=
  key_uid=$(ps_uid "$key_pid" 2>/dev/null) || key_uid=
  if ! finish_bg "$log" 40; then
    stop_key_helper "$key_pid" "$key_lstart" "$key_cmd" "$key_uid" || true
    publish_file "$transcript" "$log" || true
    camera_off
    return 1
  fi
  stop_key_helper "$key_pid" "$key_lstart" "$key_cmd" "$key_uid" || true
  camera_off
  publish_file "$transcript" "$log" || return 1
  cp "$log" "$RUN_STATE/scratch/last-auth.log" || return 1
  return 0
}

drive_authenticate() {
  require_app || return 1
  require_gui || return 1
  require_template || return 1
  make_frames || return 1
  capture_one_auth "$RUN_STATE/frames/frontal.ppm" "authenticate-accept.txt" "authenticate-accept.png" || return $?
  grep -F -q -- "Authentication successful" "$RUN_STATE/scratch/last-auth.log" || return 1
  grep -F -q -- "Identified as: $USER_ID" "$RUN_STATE/scratch/last-auth.log" || return 1
  capture_one_auth "$RUN_STATE/frames/shifted.ppm" "authenticate-reject.txt" "authenticate-reject.png" || return $?
  grep -F -q -- "Authentication failed - No match found" "$RUN_STATE/scratch/last-auth.log" || return 1
  grep -F -q -- "below minimum threshold 78.0%" "$RUN_STATE/scratch/last-auth.log" || return 1
  grep -F -q -- "Final confidence: 0.0%" "$RUN_STATE/scratch/last-auth.log" || return 1
}

drive_compare_faces() {
  require_app || return 1
  require_gui || return 1
  require_template || return 1
  make_frames || return 1
  capture_one_auth "$RUN_STATE/frames/frontal.ppm" "compare-transcript.txt" "compare-window.png" || return $?
  grep -F -q -- "Comparison for $USER_ID" "$RUN_STATE/scratch/last-auth.log" || return 1
  grep -F -q -- "Final confidence:" "$RUN_STATE/scratch/last-auth.log" || return 1
  grep -F -q -- "Pose difference:" "$RUN_STATE/scratch/last-auth.log" || return 1
  grep -F -q -- "Auth depth variance:" "$RUN_STATE/scratch/last-auth.log" || return 1
  grep -F -q -- "Authentication successful" "$RUN_STATE/scratch/last-auth.log" || return 1
  grep -F -q -- "Identified as: $USER_ID" "$RUN_STATE/scratch/last-auth.log" || return 1
}

mode_is() {
  local path="$1" expect="$2" got
  got=$(python3 "$GUARD" stat-mode "$path") || return 1
  case "$got" in
    "$expect "*) return 0 ;;
  esac
  return 1
}

drive_secure_storage() {
  local avg rootdir userdir log report
  require_app || return 1
  require_gui || return 1
  require_template || return 1
  make_frames || return 1
  avg="$RUN_STATE/sandbox/registered_faces/$USER_ID/${USER_ID}_avgface.png"
  userdir="$RUN_STATE/sandbox/registered_faces/$USER_ID"
  rootdir="$RUN_STATE/sandbox/registered_faces"
  mode_is "$avg" 600 || return 1
  mode_is "$userdir" 700 || return 1
  mode_is "$rootdir" 700 || return 1
  python3 - "$avg" << 'PY' || return 1
import sys
magic = open(sys.argv[1], "rb").read(8)
if magic != b"\x89PNG\r\n\x1a\n":
    raise SystemExit(1)
PY
  report="$RUN_STATE/scratch/storage.txt"
  {
    printf 'encrypted=no\n' || return 1
    printf 'png_magic=yes\n' || return 1
    printf 'integrity_mac=absent\n' || return 1
    python3 "$GUARD" stat-mode "$avg" || return 1
    python3 "$GUARD" stat-mode "$userdir" || return 1
  } > "$report" || return 1
  log="$RUN_STATE/scratch/invalid-id.log"
  : > "$log" || return 1
  if ! run_py_bg "$log" face_registration.py '../nope'; then
    if [ -z "${SPAWN_PID:-}" ]; then
      return 1
    fi
    set +e
    wait "$SPAWN_PID"
    APP_STATUS=$?
    set -e
    if ! grep -F -q -- "Invalid user ID" "$log"; then
      return 1
    fi
    printf 'exit_code=%s\n' "$APP_STATUS" >> "$log" || return 1
  else
    finish_bg "$log" 15 || true
  fi
  [ "$APP_STATUS" -eq 1 ] || return 1
  publish_file "storage-invalid-id.txt" "$log" || return 1
  grep -F -q -- "Invalid user ID" "$log" || return 1
  if [ -e "$RUN_STATE/nope" ] || [ -e "$RUN_STATE/sandbox/nope" ]; then
    return 1
  fi
  cp -p "$avg" "$RUN_STATE/scratch/avg.good.png" || return 1
  cat > "$RUN_STATE/scratch/edit_pixels.py" << 'PY' || return 1
import sys
import cv2
path = sys.argv[1]
image = cv2.imread(path)
if image is None:
    raise SystemExit(1)
image[0:40, 0:40] = (0, 0, 255)
if not cv2.imwrite(path, image):
    raise SystemExit(1)
PY
  apply_isolated_env "$RUN_STATE" || return 1
  "$RUN_STATE/bin/uv" run --frozen --directory "$RUN_STATE/sandbox" python -u \
    "$RUN_STATE/scratch/edit_pixels.py" "$avg" || return 1
  capture_one_auth "$RUN_STATE/frames/frontal.ppm" "storage-edit.txt" "storage-edit.png" || return $?
  if ! grep -F -q -- "Authentication successful" "$RUN_STATE/scratch/last-auth.log"; then
    return 1
  fi
  printf 'decodable_edit=accepted\n' >> "$report" || return 1
  cp -p "$RUN_STATE/scratch/avg.good.png" "$avg" || return 1
  python3 - "$avg" << 'PY' || return 1
import sys
path = sys.argv[1]
data = bytearray(open(path, "rb").read())
if len(data) < 64:
    raise SystemExit(1)
data[40] ^= 0xFF
open(path, "wb").write(data)
PY
  capture_one_auth "$RUN_STATE/frames/frontal.ppm" "storage-corrupt.txt" "storage-corrupt.png" || return $?
  grep -F -q -- "Failed to load averaged face" "$RUN_STATE/scratch/last-auth.log" || return 1
  if grep -F -q -- "Authentication successful" "$RUN_STATE/scratch/last-auth.log"; then
    return 1
  fi
  printf 'corrupt_png=rejected_at_decode\n' >> "$report" || return 1
  publish_file "storage-at-rest.txt" "$report" || return 1
}

drive_view_mesh() {
  local log
  require_app || return 1
  require_gui || return 1
  require_template || return 1
  log="$RUN_STATE/scratch/mesh-textured.log"
  run_py "$log" mesh_textured.py "$USER_ID" || return 1
  publish_file "mesh-textured.txt" "$log" || return 1
  [ "$APP_STATUS" -eq 0 ] || return "$APP_STATUS"
  grep -F -q -- "[mesh_textured] wrote:" "$log" || return 1
  mode_is "$RUN_STATE/sandbox/models/${USER_ID}_face.obj" 600 || return 1
  log="$RUN_STATE/scratch/mesh-viewer.log"
  run_py_bg "$log" mesh_viewer.py "$USER_ID" || return 1
  if ! wait_for_window "$WIN_MESH" "$SPAWN_PID" 60 >/dev/null; then
    stop_app || true
    return 1
  fi
  sleep 2
  if ! capture_window "$WIN_MESH" "mesh-window.png"; then
    stop_app || true
    return 1
  fi
  stop_app || return 1
  printf 'exit_code=signaled\n' >> "$log" || return 1
  publish_file "mesh-viewer.txt" "$log" || return 1
}

drive_feature() {
  local feature="$1"
  case "$feature" in
    register-face) drive_register_face ;;
    authenticate) drive_authenticate ;;
    compare-faces) drive_compare_faces ;;
    secure-storage) drive_secure_storage ;;
    view-mesh) drive_view_mesh ;;
    *)
      printf 'error: unknown feature %s\n' "$feature" >&2
      return 1
      ;;
  esac
}
