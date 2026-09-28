"""Cron: import path of the restart-safe external worker.

The worker is spawned as ``sys.executable -m cron.scheduler``. Its entry module is
``cron.scheduler``, not ``hermes_cli.main``, so nothing bootstraps the gateway's checkout
onto its ``sys.path``; historically it imported ``cron`` only through the implicit ``-m``
cwd entry. That entry is gone under ``PYTHONSAFEPATH`` and useless when the venv's
editable install maps a moved/deleted checkout -- the worker then dies with
"No module named 'cron'" before its ownership ack (#112729, hypothesised cause).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because user
children must not see our tree. This child IS Hermes, so the pin is applied *after* the
env is built, on the sanitized env, and restores both halves of what the sanitizer took:

* the checkout, so ``cron``/``hermes_*`` import; and
* the dependency environment this process activated, because on a self-managed
  (shell-installer / PM) install the spawned ``sys.executable`` is the PM *store* Python,
  which owns no third-party dependencies -- they reach the parent only by *path*, through
  the committed generation ``activate_dependencies`` put on this process's ``sys.path``.

Restoring the tree alone left that child with no dependency path at all, so it died at
its first import (``No module named 'ruamel'``) before publishing its ownership
acknowledgement and every scheduled job on such an install was recorded failed
(#122222). The interpreter's own ``purelib`` is excluded from the restoration, so
runners that own their dependencies (wheel / pipx / test / developer venv) behave
exactly as before.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

_SITE_PACKAGES_DIRS = frozenset({"site-packages", "dist-packages"})


def _installed_purelib() -> Path | None:
    try:
        return Path(sysconfig.get_paths()["purelib"]).resolve()
    except (KeyError, OSError):
        return None


def _in_real_venv(site_packages: Path) -> bool:
    """True when an ancestor of *site_packages* is a real venv (``pyvenv.cfg`` present).

    Keeps an unrelated directory that merely *is named* ``site-packages`` off the child's
    import path.
    """
    return any((parent / "pyvenv.cfg").is_file() for parent in site_packages.parents)


def _activated_dependency_site_packages() -> list[Path]:
    """Dependency ``site-packages`` directories this process activated, in ``sys.path`` order.

    Read from this process's own ``sys.path`` (the signal ``pm.environments``'s
    ``activate_dependencies``/``running_from_selected_environment`` publish) and not from
    PM's install records, so the child-spawn path does no home-scoped filesystem reads.
    An interpreter that owns its dependencies contributes nothing here: its own
    ``purelib`` is skipped and the child inherits that same interpreter anyway.
    """
    purelib = _installed_purelib()
    found: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            path = Path(entry).resolve()
        except OSError:
            continue
        if path.name not in _SITE_PACKAGES_DIRS or path == purelib:
            continue
        if not _in_real_venv(path):
            continue
        if path not in found:
            found.append(path)
    return found


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` and the activated dependency environment to the worker env's
    own PYTHONPATH (never ``os.environ``'s).

    Order follows what the parent itself imports through: the checkout, then the
    dependency generation, then whatever the sanitizer chose to keep. When no activated
    dependency environment is found, the previous tree-only behaviour stands.

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    pinned = [root, *(str(path) for path in _activated_dependency_site_packages())]
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*pinned, *existing]))
    return worker_env
