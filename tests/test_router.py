"""Router: discovery, the containment gate on bind, persistence and queueing."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from acp_im_gateway.access import AccessPolicy
from acp_im_gateway.containment import ContainmentError
from acp_im_gateway.router import (
    ChatRuntime,
    DiscoveryError,
    ProjectDiscovery,
    Router,
    StateStore,
    decode_index_name,
)

from .helpers import make_project


def build_router(
    projects_root: Path,
    *,
    state_file: Path,
    agent_index_dir: Path | None = None,
    allowed_roots: tuple[Path, ...] | None = None,
    allowed_user_ids: set[int] | None = None,
) -> Router:
    discovery = ProjectDiscovery(
        projects_root,
        agent_index_dir=agent_index_dir or projects_root.parent / "no-index",
        max_depth=1,
        allowed_roots=allowed_roots if allowed_roots is not None else (projects_root,),
    )
    access = AccessPolicy(allowed_user_ids=set(allowed_user_ids or {900}))
    router = Router(
        store=StateStore(state_file),
        discovery=discovery,
        access=access,
        allowed_roots=allowed_roots if allowed_roots is not None else (projects_root,),
    )
    router.load()
    return router


# --------------------------------------------------------------------------- discovery


def test_discovers_directories_containing_git(projects_root: Path, tmp_path: Path) -> None:
    router = build_router(projects_root, state_file=tmp_path / "state.json")
    names = [row["name"] for row in router.list_projects()]
    assert names == ["alpha", "beta"]  # "not-a-repo" is skipped, no hand-written list


def test_discovery_depth_can_reach_nested_projects(projects_root: Path, tmp_path: Path) -> None:
    nested = projects_root / "group" / "deep"
    nested.mkdir(parents=True)
    (nested / ".git").mkdir()
    shallow = ProjectDiscovery(projects_root, max_depth=1)
    deep = ProjectDiscovery(projects_root, max_depth=2)
    assert "deep" not in [project.name for project in shallow.discover()]
    assert "deep" in [project.name for project in deep.discover()]


def test_agent_index_entries_are_hints() -> None:
    # The encoding turns dashes into path separators, so use a dash-free location.
    with tempfile.TemporaryDirectory(prefix="acpgw") as raw:
        base = Path(raw)
        real_project = make_project(base, "myapp")
        index = base / "agentindex"
        encoded = "-" + str(real_project).lstrip("/").replace("/", "-")
        (index / encoded).mkdir(parents=True)
        (index / "-does-not-exist-anywhere").mkdir(parents=True)

        discovery = ProjectDiscovery(
            base / "Projects",
            agent_index_dir=index,
            allowed_roots=(base,),
        )
        names = [project.name for project in discovery.discover()]
        assert "myapp" in names
        assert "does-not-exist-anywhere" not in names  # missing paths are ignored


def test_decode_index_name_round_trips() -> None:
    assert str(decode_index_name("-home-dev-Projects-myapp")) == "/home/dev/Projects/myapp"
    assert str(decode_index_name("-srv-work")) == "/srv/work"


def test_missing_index_directory_is_tolerated(tmp_path: Path) -> None:
    discovery = ProjectDiscovery(tmp_path / "Projects", agent_index_dir=tmp_path / "nope")
    assert discovery.from_agent_index() == []


def test_git_file_worktree_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "Projects"
    worktree = root / "worktree"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text("gitdir: /somewhere\n", encoding="utf-8")
    discovery = ProjectDiscovery(root)
    assert [project.name for project in discovery.discover()] == ["worktree"]


# --------------------------------------------------------------------------- binding


def test_bind_by_discovered_name(projects_root: Path, tmp_path: Path) -> None:
    router = build_router(projects_root, state_file=tmp_path / "state.json")
    binding = router.bind(111, "alpha")
    assert binding.project_root == str((projects_root / "alpha").resolve())
    assert binding.project_name == "alpha"
    assert router.get(111) is binding


def test_bind_by_explicit_path(projects_root: Path, tmp_path: Path) -> None:
    extra = make_project(projects_root, "gamma")
    router = build_router(projects_root, state_file=tmp_path / "state.json")
    binding = router.bind(111, "custom-name", str(extra))
    assert binding.project_name == "custom-name"
    assert binding.project_root == str(extra.resolve())


def test_bind_outside_allowed_roots_is_refused(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = make_project(tmp_path / "outside", "secret-project")
    router = build_router(allowed, state_file=tmp_path / "state.json")
    with pytest.raises(ContainmentError):
        router.bind(111, "whatever", str(outside))
    assert router.get(111) is None


def test_bind_through_a_symlink_escape_is_refused(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = make_project(tmp_path / "outside", "secret-project")
    (allowed / "trap").symlink_to(outside)
    router = build_router(allowed, state_file=tmp_path / "state.json")
    with pytest.raises(ContainmentError):
        router.bind(111, "trap")


def test_unknown_name_is_reported(projects_root: Path, tmp_path: Path) -> None:
    router = build_router(projects_root, state_file=tmp_path / "state.json")
    with pytest.raises(DiscoveryError, match="no project named"):
        router.bind(111, "ghost")


def test_ambiguous_name_lists_candidates(tmp_path: Path) -> None:
    make_project(tmp_path / "one", "dup")
    make_project(tmp_path / "two", "dup")
    discovery = ProjectDiscovery(tmp_path, max_depth=2, allowed_roots=(tmp_path,))
    router = Router(
        store=StateStore(tmp_path / "state.json"),
        discovery=discovery,
        access=AccessPolicy(allowed_user_ids={900}),
        allowed_roots=(tmp_path,),
    )
    router.load()
    with pytest.raises(DiscoveryError, match="matches several projects"):
        router.bind(111, "dup")


def test_unbind(projects_root: Path, tmp_path: Path) -> None:
    router = build_router(projects_root, state_file=tmp_path / "state.json")
    router.bind(111, "alpha")
    assert router.unbind(111) is True
    assert router.get(111) is None
    assert router.unbind(111) is False


# --------------------------------------------------------------------------- persistence


def test_binding_round_trip_survives_restart(projects_root: Path, tmp_path: Path) -> None:
    state_file = tmp_path / "state" / "state.json"
    first = build_router(projects_root, state_file=state_file)
    first.bind(111, "alpha")
    first.set_session(111, "sess-42", model="fake-model", mode="normal", approval="ask")
    first.telegram_offset = 4242
    first.save()

    assert state_file.is_file()
    assert oct(state_file.stat().st_mode & 0o777) == "0o600"
    with state_file.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    assert raw["telegram_offset"] == 4242
    assert raw["bindings"]["111"]["session_id"] == "sess-42"

    second = build_router(projects_root, state_file=state_file)
    binding = second.get(111)
    assert binding is not None
    assert binding.project_root == str((projects_root / "alpha").resolve())
    assert binding.session_id == "sess-42"
    assert binding.model == "fake-model"
    assert binding.mode == "normal"
    assert binding.approval == "ask"
    assert second.telegram_offset == 4242


def test_corrupt_state_file_is_tolerated(projects_root: Path, tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text("{not json", encoding="utf-8")
    router = build_router(projects_root, state_file=state_file)
    assert router.bindings == {}
    router.bind(111, "alpha")  # and it recovers by overwriting
    assert json.loads(state_file.read_text(encoding="utf-8"))["bindings"]["111"]["project_name"] == "alpha"


def test_state_file_round_trips_allowlist_and_pairing(projects_root: Path, tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    router = build_router(projects_root, state_file=state_file, allowed_user_ids={900})
    request = router.access.request_pairing(901, 111, "newcomer")
    assert request is not None
    router.save()

    other = build_router(projects_root, state_file=state_file, allowed_user_ids={900})
    assert other.access.is_user_allowed(901) is False
    assert other.access.pending[request.code].user_id == 901
    other.access.approve(request.code)
    other.save()

    third = build_router(projects_root, state_file=state_file, allowed_user_ids={900})
    assert third.access.is_user_allowed(901) is True


def test_merge_external_adopts_cli_changes(projects_root: Path, tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    gateway_side = build_router(projects_root, state_file=state_file, allowed_user_ids={900})
    gateway_side.bind(111, "alpha")
    gateway_side.save()

    cli_side = build_router(projects_root, state_file=state_file, allowed_user_ids={900})
    cli_side.access.allowed_user_ids.add(777)
    cli_side.bind(222, "beta")
    cli_side.save()

    # Force a distinct mtime so the in-memory router notices the CLI's write.
    future = os.stat(state_file).st_mtime + 10
    os.utime(state_file, (future, future))

    assert gateway_side.merge_external() is True
    assert gateway_side.access.is_user_allowed(777) is True
    assert gateway_side.get(222) is not None  # adopted: we had never seen that chat
    assert gateway_side.get(111).project_name == "alpha"  # our own binding wins
    assert gateway_side.merge_external() is False


def test_list_projects_marks_binding_and_bindability(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    make_project(allowed, "inside")
    make_project(tmp_path, "sibling")  # discovered, but outside the allowed roots
    discovery = ProjectDiscovery(tmp_path, max_depth=2, allowed_roots=(allowed,))
    router = Router(
        store=StateStore(tmp_path / "state.json"),
        discovery=discovery,
        access=AccessPolicy(allowed_user_ids={900}),
        allowed_roots=(allowed,),
    )
    router.load()
    router.bind(111, "inside")
    rows = {row["name"]: row for row in router.list_projects()}
    assert rows["inside"]["bound_chat_ids"] == [111]
    assert rows["inside"]["bindable"] is True
    assert rows["sibling"]["bindable"] is False


# --------------------------------------------------------------------------- per chat


def test_chat_runtime_queue_and_labels() -> None:
    runtime = ChatRuntime(chat_id=111)
    assert runtime.queue_size == 0
    assert runtime.state_label() == "idle"
    assert runtime.enqueue("first") == 1
    assert runtime.enqueue("second") == 2
    assert runtime.dequeue() == "first"
    assert runtime.queue_size == 1
    runtime.busy = True
    assert runtime.state_label() == "busy (queue: 1)"
    assert runtime.dequeue() == "second"
    assert runtime.dequeue() is None
    assert runtime.state_label() == "busy"


def test_router_runtime_is_stable_per_chat(projects_root: Path, tmp_path: Path) -> None:
    router = build_router(projects_root, state_file=tmp_path / "state.json")
    assert router.runtime(111) is router.runtime(111)
    assert router.runtime(111) is not router.runtime(222)
    assert router.busy_chat_ids() == []
    router.runtime(111).busy = True
    assert router.busy_chat_ids() == [111]
