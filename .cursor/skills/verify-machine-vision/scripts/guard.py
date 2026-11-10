#!/usr/bin/env python3
"""Path, evidence, and snapshot guards for verify-machine-vision.

Bash calls this instead of sourcing run.env or chmod-ing a directory the
caller named. Failures print a reason and exit nonzero.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys

TMP_PREFIXES = ("/tmp", "/private/tmp")
RESERVED_NAMES = {
    "run.env",
    "run.id",
    "handlers.before",
    "handlers.after",
    "manifest.before",
    "lib.sh",
    "drive.sh",
    "guard.py",
    "regression.sh",
    "verify-machine-vision",
    "fakevideo.c",
    "synthetic_face.py",
    "side-effects.log",
    "recover.txt",
}


def die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def under_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    for prefix in prefixes:
        if path == prefix or path.startswith(prefix + os.sep):
            return True
    return False


def real_whole(path: str) -> str:
    """Resolve the entire path in one call. Do not resolve only the tail."""
    return os.path.realpath(path)


def scratch_ok(path: str) -> bool:
    resolved = real_whole(path)
    return under_prefix(resolved, TMP_PREFIXES)


def symlink_component(path: str) -> bool:
    if not path.startswith(os.sep):
        path = os.path.abspath(path)
    current = os.sep
    parts = [part for part in path.split(os.sep) if part]
    for part in parts:
        current = os.path.join(current, part)
        if os.path.islink(current):
            return True
    return False


def assert_private_dir(path: str) -> None:
    if symlink_component(path) or os.path.islink(path):
        die(f"refusing symlink dir: {path}")
    try:
        st = os.lstat(path)
    except OSError as exc:
        die(f"unreadable: {path}: {exc}")
    if not stat.S_ISDIR(st.st_mode):
        die(f"not a directory: {path}")
    if st.st_uid != os.getuid():
        die(f"refusing dir we do not own: {path}")
    if stat.S_IMODE(st.st_mode) & 0o022:
        die(f"refusing group or world writable dir: {path}")


def inside(root: str, path: str) -> bool:
    assert_private_dir(root)
    root_r = real_whole(root)
    if os.path.lexists(path) and os.path.islink(path):
        return False
    if symlink_component(path):
        return False
    if os.path.lexists(path):
        path_r = real_whole(path)
    else:
        parent = os.path.dirname(path)
        if parent == "":
            return False
        parent_r = real_whole(parent)
        base = os.path.basename(path)
        if base in ("", ".", "..") or os.sep in base or (os.altsep and os.altsep in base):
            return False
        path_r = os.path.join(parent_r, base)
    return path_r == root_r or path_r.startswith(root_r + os.sep)


def walk_or_die(root: str):
    def onerror(err: OSError) -> None:
        raise err

    yield from os.walk(root, followlinks=False, onerror=onerror)


def hash_file(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        die(f"unreadable: {path}: {exc}")
    return digest.hexdigest()


def iter_entries(root: str):
    root_r = real_whole(root)
    try:
        for dirpath, dirnames, filenames in walk_or_die(root):
            dirnames.sort()
            filenames.sort()
            for name in dirnames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.lstat(path)
                except OSError as exc:
                    die(f"unreadable: {path}: {exc}")
                yield path, st
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.lstat(path)
                except OSError as exc:
                    die(f"unreadable: {path}: {exc}")
                yield path, st
    except OSError as exc:
        die(f"unreadable: {root_r}: {exc}")


def hash_tree(root: str) -> str:
    assert_private_dir(root)
    digest = hashlib.sha256()
    for path, st in iter_entries(root):
        rel = os.path.relpath(path, root)
        digest.update(rel.encode())
        digest.update(str(stat.S_IMODE(st.st_mode)).encode())
        if stat.S_ISREG(st.st_mode):
            digest.update(hash_file(path).encode())
        elif stat.S_ISLNK(st.st_mode):
            try:
                digest.update(os.readlink(path).encode())
            except OSError as exc:
                die(f"unreadable: {path}: {exc}")
    return digest.hexdigest()


def snapshot_tree(root: str, dest: str) -> None:
    assert_private_dir(root)
    lines = []
    for path, st in iter_entries(root):
        rel = os.path.relpath(path, root)
        if stat.S_ISREG(st.st_mode):
            file_hash = hash_file(path)
        elif stat.S_ISLNK(st.st_mode):
            try:
                file_hash = "link:" + os.readlink(path)
            except OSError as exc:
                die(f"unreadable: {path}: {exc}")
        else:
            file_hash = "-"
        lines.append(f"{rel}\t{stat.S_IMODE(st.st_mode):04o}\t{st.st_size}\t{file_hash}")
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode()
    write_bytes_no_follow(dest, payload)


def write_bytes_replace(path: str, payload: bytes) -> None:
    """Replace a cleanup state file. Temp file plus rename, never O_EXCL on the final path."""
    parent = os.path.dirname(path) or "."
    assert_private_dir(parent)
    if symlink_component(parent) or (os.path.lexists(path) and os.path.islink(path)):
        die(f"refusing symlink: {path}")
    base = os.path.basename(path)
    if base in ("", ".", ".."):
        die(f"refusing state name: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = None
    tmp = ""
    n = 0
    while n < 20:
        candidate = os.path.join(parent, f".{base}.{os.getpid()}.{n}.tmp")
        if os.path.islink(candidate):
            die(f"refusing symlink: {candidate}")
        try:
            fd = os.open(candidate, flags, 0o600)
            tmp = candidate
            break
        except FileExistsError:
            n += 1
            continue
        except OSError as exc:
            die(f"refusing state write: {path}: {exc}")
    if fd is None or not tmp:
        die(f"refusing state write: {path}: could not create a temp file")
    try:
        os.write(fd, payload)
        os.fchmod(fd, 0o600)
        os.fsync(fd)
        file_stat = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        os.replace(tmp, path)
    except OSError as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        die(f"refusing state write: {path}: {exc}")
    try:
        path_stat = os.lstat(path)
    except OSError as exc:
        die(f"unreadable: {path}: {exc}")
    if stat.S_ISLNK(path_stat.st_mode) or path_stat.st_ino != file_stat.st_ino:
        die(f"state path changed during write: {path}")
    if stat.S_IMODE(path_stat.st_mode) != 0o600:
        die(f"state mode is not 600: {path}")


def write_bytes_no_follow(path: str, payload: bytes) -> None:
    parent = os.path.dirname(path)
    if symlink_component(parent) or os.path.islink(path):
        die(f"refusing symlink: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        die(f"refusing evidence write: {path}: {exc}")
    try:
        os.write(fd, payload)
        os.fchmod(fd, 0o600)
        file_stat = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        path_stat = os.lstat(path)
    except OSError as exc:
        die(f"unreadable: {path}: {exc}")
    if stat.S_ISLNK(path_stat.st_mode) or path_stat.st_ino != file_stat.st_ino:
        die(f"evidence path changed during write: {path}")
    if stat.S_IMODE(path_stat.st_mode) != 0o600:
        die(f"evidence mode is not 600: {path}")


def valid_capture_name(name: str) -> bool:
    if name in RESERVED_NAMES or name.startswith("."):
        return False
    if name in (".", "..") or "/" in name or "\\" in name or ".." in name:
        return False
    if not name or len(name) > 80:
        return False
    first = name[0]
    if not (first.isalnum()):
        return False
    for char in name:
        if not (char.isalnum() or char in "._-"):
            return False
    return True


def write_evidence(evidence_dir: str, name: str) -> None:
    assert_private_dir(evidence_dir)
    if not valid_capture_name(name):
        die(f"refusing capture name: {name}")
    target = os.path.join(real_whole(evidence_dir), name)
    if not inside(evidence_dir, target):
        die(f"evidence target escaped the evidence dir: {target}")
    if os.path.lexists(target):
        die(f"refusing existing evidence path: {target}")
    payload = sys.stdin.buffer.read()
    write_bytes_no_follow(target, payload)
    print(target)


def compare_files(left: str, right: str) -> None:
    try:
        with open(left, "rb") as handle:
            left_bytes = handle.read()
        with open(right, "rb") as handle:
            right_bytes = handle.read()
    except OSError as exc:
        print("unreadable")
        die(f"unreadable: {exc}")
    if left_bytes == right_bytes:
        print("unchanged")
        return
    print("changed")
    raise SystemExit(2)


def chmod_writable_inside(root: str, path: str) -> None:
    """Make a tree we created writable before rm. Never follow a symlink."""
    if os.path.islink(path):
        die(f"refusing to chmod a symlink: {path}", 2)

    def one(current: str) -> None:
        if not inside(root, current):
            die(f"refusing to chmod outside the run root: {current}", 2)
        try:
            info = os.lstat(current)
        except OSError as exc:
            die(f"unreadable: {current}: {exc}")
        if stat.S_ISLNK(info.st_mode):
            return
        mode = stat.S_IMODE(info.st_mode) | stat.S_IWUSR
        if stat.S_ISDIR(info.st_mode):
            mode |= stat.S_IXUSR
        os.chmod(current, mode)

    def descend(current: str) -> None:
        if os.path.islink(current):
            return
        one(current)
        if not os.path.isdir(current):
            return
        try:
            names = os.listdir(current)
        except OSError as exc:
            die(f"unreadable: {current}: {exc}")
        for name in names:
            descend(os.path.join(current, name))

    descend(path)


def watch_new(root: str, manifest: str, output: str) -> None:
    assert_private_dir(root)
    previous: dict[str, str] = {}
    try:
        with open(manifest, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                rel, _mode, _size, file_hash = line.split("\t", 3)
                previous[rel] = file_hash
    except OSError as exc:
        die(f"unreadable: {manifest}: {exc}")
    found = []
    for path, st in iter_entries(root):
        rel = os.path.relpath(path, root)
        if os.path.basename(rel) == ".DS_Store":
            continue
        if stat.S_ISREG(st.st_mode):
            file_hash = hash_file(path)
        elif stat.S_ISLNK(st.st_mode):
            try:
                file_hash = "link:" + os.readlink(path)
            except OSError as exc:
                die(f"unreadable: {path}: {exc}")
        else:
            file_hash = "-"
        if rel not in previous or previous[rel] != file_hash:
            found.append(rel)
    payload = ("\n".join(found) + ("\n" if found else "")).encode()
    write_bytes_no_follow(output, payload)


def resolve_desktop(desktop_id: str) -> str:
    if not desktop_id or "/" in desktop_id or desktop_id.startswith("."):
        return ""
    bases = []
    data_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share")
    for entry in data_dirs.split(":"):
        if entry:
            bases.append(os.path.join(entry, "applications"))
    home = os.environ.get("HOME", "")
    if home:
        bases.append(os.path.join(home, ".local", "share", "applications"))
    for base in bases:
        candidate = os.path.join(base, desktop_id)
        if os.path.islink(candidate):
            continue
        if os.path.isfile(candidate):
            return real_whole(candidate)
    return ""


def handlers_snapshot(dest: str, real_home: str) -> None:
    system = os.environ.get("VMV_PLATFORM") or os.uname().sysname
    lines = []
    if system == "Linux":
        if shutil.which("xdg-mime") is None:
            lines.append("xdg-mime\tabsent\t")
        else:
            for mime in (
                "x-scheme-handler/http",
                "x-scheme-handler/https",
                "image/png",
            ):
                proc = subprocess.run(
                    ["xdg-mime", "query", "default", mime],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    die(f"xdg-mime failed for {mime}: {proc.stderr.strip()}")
                desktop = proc.stdout.strip()
                full = resolve_desktop(desktop) if desktop else ""
                lines.append(f"{mime}\t{desktop}\t{full}")
    elif system == "Darwin":
        pref = os.path.join(real_home, "Library", "Preferences")
        if not os.path.isdir(pref):
            lines.append(f"preferences\tabsent\t{pref}")
        else:
            for path, st in iter_entries(pref):
                if os.path.basename(path) == ".DS_Store":
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                # Compare handler files by full path, not by bundle id.
                lines.append(f"pref\t{real_whole(path)}\t{hash_file(path)}")
    else:
        die(f"unsupported platform: {system}")
    payload = ("\n".join(lines) + "\n").encode()
    parent = os.path.dirname(dest)
    if parent and not os.path.isdir(parent):
        die(f"missing snapshot dir: {parent}")
    write_bytes_replace(dest, payload)


def find_runs(prefix: str, run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id or ""):
        die("refusing run id")
    bases = ["/tmp", "/private/tmp"]
    seen = set()
    found = []
    for base in bases:
        if not os.path.isdir(base):
            continue
        try:
            entries = list(os.scandir(base))
        except OSError as exc:
            die(f"unreadable: {base}: {exc}")
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not entry.name.startswith(prefix):
                    continue
                st = entry.stat(follow_symlinks=False)
            except OSError as exc:
                die(f"unreadable: {entry.path}: {exc}")
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            if st.st_uid != os.getuid():
                continue
            if stat.S_IMODE(st.st_mode) & 0o022:
                continue
            id_path = os.path.join(entry.path, "run.id")
            if not os.path.isfile(id_path) or os.path.islink(id_path):
                continue
            try:
                text = open(id_path, "r", encoding="utf-8").read()
            except OSError as exc:
                die(f"unreadable: {id_path}: {exc}")
            if text not in (run_id, run_id + "\n"):
                continue
            found.append(entry.path)
    if len(found) != 1:
        die(f"expected one {prefix} dir for {run_id}, found {len(found)}")
    print(found[0])


def stat_mode(path: str) -> None:
    try:
        st = os.lstat(path)
    except OSError as exc:
        die(f"unreadable: {path}: {exc}")
    print(f"{stat.S_IMODE(st.st_mode):03o} {st.st_uid}")


def exact_name(expected: str, actual: str) -> None:
    if expected != actual:
        die(f"accessible name mismatch: {actual!r} != {expected!r}")
    print("match")


def exact_url_path(expected: str, actual: str) -> None:
    from urllib.parse import urlsplit

    expected_path = urlsplit(expected).path if "://" in expected else expected
    actual_path = urlsplit(actual).path if "://" in actual else actual
    if expected_path != actual_path:
        die(f"url path mismatch: {actual_path!r} != {expected_path!r}")
    print("match")


def main(argv: list[str]) -> None:
    if not argv:
        die("usage: guard.py COMMAND ...")
    command = argv[0]
    args = argv[1:]
    if command == "scratch-ok":
        if scratch_ok(args[0]):
            print("ok")
            return
        die("outside scratch prefixes")
    if command == "inside":
        if inside(args[0], args[1]):
            print("ok")
            return
        die("outside root")
    if command == "private-dir":
        assert_private_dir(args[0])
        print("ok")
        return
    if command == "hash-tree":
        print(hash_tree(args[0]))
        return
    if command == "snapshot-tree":
        snapshot_tree(args[0], args[1])
        return
    if command == "compare":
        compare_files(args[0], args[1])
        return
    if command == "write-evidence":
        write_evidence(args[0], args[1])
        return
    if command == "write-replace":
        payload = sys.stdin.buffer.read()
        write_bytes_replace(args[0], payload)
        return
    if command == "chmod-inside":
        chmod_writable_inside(args[0], args[1])
        print("ok")
        return
    if command == "watch-new":
        watch_new(args[0], args[1], args[2])
        return
    if command == "handlers-snapshot":
        handlers_snapshot(args[0], args[1])
        return
    if command == "find-run":
        find_runs(args[0], args[1])
        return
    if command == "stat-mode":
        stat_mode(args[0])
        return
    if command == "exact-name":
        exact_name(args[0], args[1])
        return
    if command == "exact-url-path":
        exact_url_path(args[0], args[1])
        return
    die(f"unknown guard command: {command}")


if __name__ == "__main__":
    main(sys.argv[1:])
