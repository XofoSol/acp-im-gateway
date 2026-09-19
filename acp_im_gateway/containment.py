"""Containment gate.

A chat may only bind a directory *inside* an allowed root. This is a code gate,
not a convention: every candidate path is resolved with ``os.path.realpath``
before it is compared, so ``..`` traversal and symlink escapes are rejected.

    resolve_root("/home/tmp/root", ["/home/me/Projects"])  -> raises ContainmentError
    resolve_root("/home/me/Projects/app", ["/home/me/Projects"]) -> PosixPath('/home/me/Projects/app')
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


class ContainmentError(Exception):
    """A path is missing, not a directory, or outside every allowed root."""


def resolve_path(path: str | os.PathLike[str]) -> Path:
    """``~``-expand and fully resolve ``path`` without requiring it to exist.

    Resolution (``realpath``) is what makes the gate safe: it collapses ``..``
    and follows every symlink, so the returned path is the real location.
    """
    expanded = os.path.expanduser(os.fspath(path))
    return Path(os.path.realpath(expanded))


def is_within_root(root: str | os.PathLike[str], path: str | os.PathLike[str]) -> bool:
    """True when ``path`` is ``root`` itself or lives underneath it (realpaths).

    Both sides are resolved first. Comparison is component-wise via
    ``os.path.commonpath`` so ``/srv/app-2`` is *not* treated as inside ``/srv/app``.
    """
    real_root = os.path.realpath(os.fspath(root))
    real_path = os.path.realpath(os.fspath(path))
    try:
        return os.path.commonpath([real_root, real_path]) == real_root
    except ValueError:
        # Different drives / relative vs absolute mix: not contained.
        return False


def roots_as_text(roots: Iterable[str | os.PathLike[str]]) -> str:
    """Comma-joined realpaths, for error messages and ``/status`` output."""
    return ", ".join(os.path.realpath(os.fspath(root)) for root in roots)


def resolve_root(
    candidate: str | os.PathLike[str],
    roots: Iterable[str | os.PathLike[str]],
    *,
    what: str = "project root",
    must_exist: bool = True,
) -> Path:
    """Return the realpath of ``candidate`` if it is contained in one of ``roots``.

    Raises :class:`ContainmentError` for an empty path, a missing path, a file
    instead of a directory, or a path outside every allowed root.
    """
    text = os.fspath(candidate).strip()
    if not text:
        raise ContainmentError(f"{what} is empty")

    root_list = [os.path.realpath(os.fspath(root)) for root in roots]
    if not root_list:
        raise ContainmentError(f"no allowed roots configured, refusing to bind {text!r}")

    real = resolve_path(text)

    if must_exist and not real.exists():
        raise ContainmentError(f"{what} does not exist: {real}")
    if must_exist and not real.is_dir():
        raise ContainmentError(f"{what} is not a directory: {real}")

    if not any(is_within_root(root, real) for root in root_list):
        raise ContainmentError(
            f"{what} {real} is outside the allowed roots ({', '.join(root_list)})"
        )
    return real
