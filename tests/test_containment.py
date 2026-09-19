"""Containment gate: a chat may only bind a directory inside an allowed root."""

from __future__ import annotations

from pathlib import Path

import pytest

from acp_im_gateway.containment import (
    ContainmentError,
    is_within_root,
    resolve_path,
    resolve_root,
)


def test_inside_root_is_allowed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    project = root / "app"
    project.mkdir(parents=True)
    assert resolve_root(project, [root]) == project.resolve()


def test_root_itself_is_allowed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert resolve_root(root, [root]) == root.resolve()


def test_outside_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(ContainmentError, match="outside the allowed roots"):
        resolve_root(outside, [root])


def test_sibling_with_shared_prefix_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "app"
    sibling = tmp_path / "app-2"
    root.mkdir()
    sibling.mkdir()
    assert is_within_root(root, sibling) is False
    with pytest.raises(ContainmentError):
        resolve_root(sibling, [root])


def test_parent_traversal_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret"
    secret.mkdir()
    sneaky = root / ".." / "secret"
    with pytest.raises(ContainmentError, match="outside the allowed roots"):
        resolve_root(sneaky, [root])


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside)
    with pytest.raises(ContainmentError, match="outside the allowed roots"):
        resolve_root(root / "escape", [root])


def test_symlink_inside_root_is_allowed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    real = root / "real"
    real.mkdir(parents=True)
    (root / "link").symlink_to(real)
    assert resolve_root(root / "link", [root]) == real.resolve()


def test_symlinked_root_still_contains_its_real_children(tmp_path: Path) -> None:
    real_root = tmp_path / "real-root"
    project = real_root / "app"
    project.mkdir(parents=True)
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root)
    assert resolve_root(project, [linked_root]) == project.resolve()


def test_missing_path_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ContainmentError, match="does not exist"):
        resolve_root(root / "ghost", [root])


def test_file_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "README.md").write_text("hi", encoding="utf-8")
    with pytest.raises(ContainmentError, match="not a directory"):
        resolve_root(root / "README.md", [root])


def test_no_allowed_roots_means_no_binding(tmp_path: Path) -> None:
    project = tmp_path / "app"
    project.mkdir()
    with pytest.raises(ContainmentError, match="no allowed roots"):
        resolve_root(project, [])


def test_empty_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ContainmentError, match="empty"):
        resolve_root("   ", [tmp_path])


def test_second_root_is_accepted(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    project = second / "app"
    project.mkdir(parents=True)
    first.mkdir()
    assert resolve_root(project, [first, second]) == project.resolve()


def test_resolve_path_expands_user_and_dot_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = resolve_path("~/a/../b")
    assert resolved == (Path.home() / "b").resolve()
