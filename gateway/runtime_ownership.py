"""Nonblocking, all-or-nothing reservations of canonical profile homes.

A held lock inode is never removed: unlinking it creates a second owner. The one exception is a
stale inode this user cannot write (left by a gateway that ran as root, #42685) that a flock through
a read-only descriptor proves unheld; it is replaced once, and every winner checks that the path
still names the inode it locked.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import stat
import threading
from hermes_dir_acl import make_user_dir

_LOGGER = logging.getLogger(__name__)


class OwnershipConflict(RuntimeError):
    def __init__(self, message: str, *, home: Path | None = None):
        super().__init__(message)
        self.home = home


def canonical_home(home: Path) -> Path:
    return Path(os.path.normcase(str(Path(home).expanduser().resolve())))


def tighten_lock_mode(fd: int) -> None:
    """Validate an open ``gateway.lock`` descriptor and narrow an upgraded lock to 0600.

    Main's ``acquire_gateway_runtime_lock`` created the inode with ``open(path, "a+")`` under the
    umask (0644 on most hosts), and lock inodes are never replaced, so an upgraded home keeps that
    mode forever and discovery refuses it. Only a regular, single-link inode this user owns is
    chmodded, through the already-open NOFOLLOW descriptor (no path race); anything else is refused.
    """
    info = os.fstat(fd)
    if os.name == 'nt':
        if not stat.S_ISREG(info.st_mode):
            raise PermissionError('unsafe gateway lock owner or type')
        return
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():  # windows-footgun: ok — POSIX branch
        raise PermissionError('unsafe gateway lock owner or type')
    if stat.S_IMODE(info.st_mode) & 0o077:
        if info.st_nlink != 1:
            raise PermissionError('unsafe gateway lock link count')
        os.fchmod(fd, 0o600)  # windows-footgun: ok — POSIX branch


_LOCK_FLAGS = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)


def _open_checked(path: Path, flags: int) -> int:
    fd = os.open(path, flags, 0o600)
    try:
        tighten_lock_mode(fd)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _unlink_stale_lock(path: Path) -> bool:
    """Unlink a ``gateway.lock`` this user cannot write (another uid's) once a flock taken through a
    read-only descriptor proves it unheld. Symlinks, hardlinks, non-regular files, a home whose
    ``gateway.pid`` names a live process and anything not provably unheld are left in place (False).
    An inode this user cannot even open is refused with the recovery command: an unheld snapshot
    (``/proc/locks``) does not stop a holder locking it between the read and the unlink."""
    from gateway.status import _live_pid_from_record, _read_pid_record
    if os.name == 'nt':
        return False  # a sharing violation is a live holder there
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        return False
    if _live_pid_from_record(_read_pid_record(path.with_name('gateway.pid'))) is not None:
        return False
    try:
        probe = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK)
    except PermissionError:
        raise PermissionError(foreign_lock_recovery(path)) from None
    import fcntl
    try:
        if not _same_inode(os.fstat(probe), info):
            return False
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        # Unlinked while still locked: a late locker of the old inode never shares our lock.
        return _unlink_if_same(path, info)
    finally:
        os.close(probe)


def foreign_lock_recovery(path: Path) -> str:
    return (f'foreign_stale_lock: {path} is not readable by this user (left by a gateway run as root?) '
            f'and cannot be proven unheld. If `sudo lsof {path}` lists no process, run `sudo rm {path}`.')


def unreadable_lock_recovery(path: Path) -> str | None:
    """The recovery a client prints for a ``gateway.lock`` it cannot even open read-only."""
    try:
        os.close(os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK))
    except PermissionError:
        return foreign_lock_recovery(path)
    except OSError:
        pass
    return None


def _unlink_if_same(path: Path, info) -> bool:
    if not _same_inode(os.lstat(path), info):
        return False
    os.unlink(path)
    return True


def _same_inode(left, right) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _open_lock(path: Path) -> int:
    """Descriptor for *path*, replacing a provably stale inode this user cannot use once."""
    try:
        return _open_checked(path, _LOCK_FLAGS)
    except PermissionError:
        if not _unlink_stale_lock(path):
            raise
    # A racer may create the fresh inode first; both then contend for its flock as usual.
    return _open_checked(path, _LOCK_FLAGS)


def _still_names(path: Path, fd: int) -> bool:
    """True when *path* is still the inode *fd* locked (a stale-lock replacement unlinks it)."""
    if os.name == 'nt':
        return True
    try:
        return _same_inode(os.lstat(path), os.fstat(fd))
    except FileNotFoundError:
        return False


def _safe_log_value(value, *, limit: int = 80) -> str:
    """Keep lock-record labels concise and free of control characters in logs."""
    text = str(value or '')
    safe = ''.join(char for char in text if char.isalnum() or char in '._:-')[:limit]
    return safe or 'unknown'


def _proc_flock_owner_pid(lock_path: Path) -> int | None:
    """Find the kernel-reported flock owner for an unpublished or stale lock record."""
    if os.name == 'nt' or not Path('/proc/locks').is_file():
        return None
    try:
        info = lock_path.stat()
        expected = (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
        for line in Path('/proc/locks').read_text(encoding='utf-8').splitlines():
            fields = line.split()
            try:
                index = fields.index('FLOCK')
                pid = int(fields[index + 3])
                major, minor, inode = fields[index + 4].split(':', 2)
                observed = (int(major, 16), int(minor, 16), int(inode))
            except (ValueError, IndexError):
                continue
            if observed == expected and pid > 0:
                return pid
    except OSError:
        pass
    return None


def _ownership_conflict_snapshot(home: Path) -> dict[str, str]:
    """Return a redacted, read-only owner snapshot; never include argv or environment."""
    snapshot = {
        'lock_record_pid': 'unknown', 'owner_pid': 'unknown', 'owner_pid_source': 'unavailable',
        'owner_kind': 'unknown', 'owner_purpose': 'unknown', 'owner_start_time': 'unknown',
        'owner_comm': 'unknown', 'owner_exe': 'unknown', 'owner_unit': 'unknown',
    }
    try:
        from gateway.status import _live_pid_from_record, _read_gateway_lock_record
        record = _read_gateway_lock_record(home / 'gateway.lock') or {}
        raw_pid = record.get('pid')
        if isinstance(raw_pid, (int, str)) and str(raw_pid).isdigit():
            snapshot['lock_record_pid'] = str(raw_pid)
        snapshot['owner_kind'] = _safe_log_value(record.get('kind'))
        snapshot['owner_purpose'] = _safe_log_value(record.get('purpose'))
        start_time = record.get('start_time')
        if isinstance(start_time, (int, float, str)):
            snapshot['owner_start_time'] = _safe_log_value(start_time)
        pid = _live_pid_from_record(record)
        if pid is not None:
            snapshot['owner_pid_source'] = 'gateway_lock_record'
        else:
            pid = _proc_flock_owner_pid(home / 'gateway.lock')
            if pid is not None:
                snapshot['owner_pid_source'] = 'proc_locks'
        if pid is None:
            return snapshot
        snapshot['owner_pid'] = str(pid)
        if os.name != 'nt':
            proc = Path('/proc') / str(pid)
            try:
                snapshot['owner_comm'] = _safe_log_value((proc / 'comm').read_text(encoding='utf-8').strip())
            except OSError:
                pass
            try:
                snapshot['owner_exe'] = str(os.readlink(proc / 'exe'))[:240]
            except OSError:
                pass
            try:
                cgroup = (proc / 'cgroup').read_text(encoding='utf-8')
                units = [part for line in cgroup.splitlines() for part in line.split('/') if part.endswith('.service')]
                if units:
                    snapshot['owner_unit'] = _safe_log_value(units[-1])
            except OSError:
                pass
    except Exception:
        # Diagnostics must never change the lock refusal or its safety semantics.
        pass
    return snapshot



def _log_ownership_conflict(home: Path, purpose: str | None) -> None:
    snapshot = _ownership_conflict_snapshot(home)
    _LOGGER.warning(
        'exclusive maintenance refused profile=%s attempted_purpose=%s '
        'lock_record_pid=%s owner_pid=%s owner_pid_source=%s owner_kind=%s owner_purpose=%s '
        'owner_start_time=%s owner_comm=%r owner_exe=%r owner_unit=%s',
        home, _safe_log_value(purpose), snapshot['lock_record_pid'], snapshot['owner_pid'],
        snapshot['owner_pid_source'], snapshot['owner_kind'], snapshot['owner_purpose'], snapshot['owner_start_time'],
        snapshot['owner_comm'], snapshot['owner_exe'], snapshot['owner_unit'],
    )


class ProfileOwnership:
    def __init__(self):
        self._handles: dict[Path, object] = {}
        self._writers: list[threading.Thread] = []
        self._mutex = threading.RLock()

    @property
    def homes(self) -> tuple[Path, ...]:
        with self._mutex:
            return tuple(self._handles)

    def reserve(self, homes, *, purpose: str | None = None) -> None:
        """``purpose`` marks a maintenance reservation in the lock record, so a contender can tell
        a short maintenance holder from a gateway runtime owner (:func:`live_runtime_owner`)."""
        from gateway.status import _try_acquire_file_lock, _build_pid_record
        with self._mutex:
            added = []
            try:
                for home in sorted({canonical_home(p) for p in homes}):
                    if home in self._handles:
                        continue
                    make_user_dir(home, mode=0o700, parents=True, exist_ok=True)
                    path = home / 'gateway.lock'
                    fd = _open_lock(path)
                    handle = os.fdopen(fd, 'r+', encoding='utf-8')  # windows-footgun: ok — write-only
                    try:
                        if not _try_acquire_file_lock(handle) or not _still_names(path, handle.fileno()):
                            raise OwnershipConflict(f'Gateway runtime already owns profile {home}', home=home)
                        record = {**_build_pid_record(), 'hermes_home': str(home)}
                        if purpose:
                            record['purpose'] = purpose
                        handle.seek(0)
                        handle.truncate()
                        json.dump(record, handle)
                        handle.flush()
                        os.fsync(handle.fileno())
                    except BaseException:
                        handle.close()
                        raise
                    self._handles[home] = handle
                    added.append(home)
            except BaseException:
                for home in reversed(added):
                    self.release(home)
                raise

    def owns(self, home: Path) -> bool:
        with self._mutex:
            return canonical_home(home) in self._handles

    def release(self, home: Path) -> None:
        from gateway.status import _release_file_lock
        with self._mutex:
            handle = self._handles.pop(canonical_home(home), None)
            if handle is not None:
                _release_file_lock(handle)
                handle.close()

    def start_writer(self, thread: threading.Thread) -> None:
        """Register before starting so partial startup cannot forget a live writer."""
        with self._mutex:
            self._writers.append(thread)
            thread.start()

    def close(self) -> None:
        with self._mutex:
            # A timed-out daemon can still write during finally/atexit. Keep the
            # handles alive; process exit releases them atomically in the OS.
            if any(thread.is_alive() for thread in self._writers):
                return
            self._writers.clear()
            for home in reversed(self.homes):
                self.release(home)


process_ownership = ProfileOwnership()
_maintenance = threading.local()


def maintenance_holder(home: Path, purpose: str) -> dict | None:
    """The live holder record of *home*'s ``gateway.lock`` when it is a ``purpose`` maintenance
    reservation (see :meth:`ProfileOwnership.reserve`), else None. Advisory only: it names the
    holder for a refusal or a short wait, it never grants the lock."""
    from gateway.status import _live_pid_from_record, _read_gateway_lock_record
    record = _read_gateway_lock_record(canonical_home(home) / 'gateway.lock')
    if not record or record.get('purpose') != purpose or _live_pid_from_record(record) is None:
        return None
    return record


def lock_record_unpublished(home: Path) -> bool:
    """True when *home*'s ``gateway.lock`` record names no live holder yet: empty, torn, or a dead
    previous holder's. :meth:`ProfileOwnership.reserve` takes the flock BEFORE it writes the record,
    so a contender that loses the flock in that window cannot tell maintenance from a gateway; it
    must retry (bounded) instead of reporting a runtime owner. Advisory only, never grants the lock."""
    from gateway.status import _live_pid_from_record, _read_gateway_lock_record
    record = _read_gateway_lock_record(canonical_home(home) / 'gateway.lock')
    return not record or _live_pid_from_record(record) is None


def live_runtime_owner(home: Path) -> int | None:
    """PID of a live runtime owner of *home* (a gateway, or a reservation that is not marked as
    maintenance), else None. A purpose-marked maintenance holder is never an owner."""
    from gateway.status import _live_pid_from_record, _read_gateway_lock_record, get_running_pid
    home = canonical_home(home)
    record = _read_gateway_lock_record(home / 'gateway.lock')
    live = _live_pid_from_record(record) if record else None
    if live is not None and record.get('purpose'):
        return None  # the flock holder is maintenance, so no gateway can hold it too
    return get_running_pid(home / 'gateway.pid', cleanup_stale=False) or live


@contextmanager
def exclusive_maintenance(homes, *, purpose: str | None = None):
    """Reserve the authority's exact lock before touching managed state.

    This is not an owner-presence check: the reservation remains held through
    publication, excluding startup even before its PID/control socket exists.
    Only nested synchronous maintenance in this thread may reuse a reservation;
    being the runtime owner (even in this process) never grants maintenance.
    Lock inodes must not be restored, removed, or replaced by callers.
    """
    state = getattr(_maintenance, 'state', None)
    outermost = state is None or state[0] != os.getpid()
    owner = state[1] if state is not None and not outermost else ProfileOwnership()
    previous = set(owner.homes)
    try:
        owner.reserve(homes, purpose=purpose)
        if outermost:
            _maintenance.state = (os.getpid(), owner)
        yield
    except OwnershipConflict as exc:
        conflict_home = canonical_home(exc.home) if exc.home is not None else None
        if conflict_home is not None:
            _log_ownership_conflict(conflict_home, purpose)
        raise OwnershipConflict(
            f'Exclusive maintenance refused: {exc}. Drain and stop the gateway, then retry.',
            home=conflict_home,
        ) from exc
    finally:
        for home in reversed(owner.homes):
            if home not in previous:
                owner.release(home)
        if outermost:
            _maintenance.state = None
