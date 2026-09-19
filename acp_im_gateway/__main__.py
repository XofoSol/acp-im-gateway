"""Command line entry point.

    python -m acp_im_gateway run                     # start the gateway
    python -m acp_im_gateway pairing list            # pending pairing codes
    python -m acp_im_gateway pairing approve <code>  # allow a sender
    python -m acp_im_gateway pairing reject <code>   # drop a pairing code
    python -m acp_im_gateway bindings list           # chat -> project bindings
    python -m acp_im_gateway bindings add <chat_id> <path>
    python -m acp_im_gateway bindings remove <chat_id>
    python -m acp_im_gateway projects                # discovered project candidates
    python -m acp_im_gateway config                  # effective configuration

``pairing`` / ``bindings`` / ``projects`` only touch the state file and the
filesystem, so they are safe to run while the gateway is up: the running process
picks allowlist changes up on its next poll and adopts bindings it has not seen.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .access import AccessPolicy, PairingError
from .acp import AcpError
from .config import Config, ConfigError, parse_id_list, parse_path_list, split_command
from .containment import ContainmentError
from .gateway import Gateway
from .router import DiscoveryError, ProjectDiscovery, Router, StateStore

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2


# --------------------------------------------------------------------------- helpers


def _common_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Flags accepted both before and after the subcommand."""
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=default,
        help="print instead of calling Telegram",
    )
    parser.add_argument("--log-level", default=default, help="DEBUG | INFO | WARNING | ERROR")
    parser.add_argument(
        "--state-file", default=default, help="state JSON path (default GATEWAY_STATE_FILE)"
    )
    parser.add_argument(
        "--projects-root", default=default, help="root scanned for project candidates"
    )
    parser.add_argument(
        "--allowed-roots", default=default, help="comma-separated roots a chat may bind inside"
    )
    parser.add_argument(
        "--allowed-user-ids", default=default, help="comma-separated Telegram user ids"
    )
    parser.add_argument(
        "--allowed-chat-ids", default=default, help="comma-separated group chat ids"
    )
    parser.add_argument(
        "--agent-cmd", default=default, help="agent command, e.g. 'reasonix acp'"
    )


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    _common_flags(common, suppress=True)

    parser = argparse.ArgumentParser(
        prog="python -m acp_im_gateway",
        parents=[common],
        description=(
            "Drive an ACP-speaking coding agent (Reasonix today, any ACP agent "
            "tomorrow) from Telegram."
        ),
        epilog=(
            "Runs on the Python standard library only. Configuration comes from "
            "./.env, the environment, or an optional TOML file — see .env.example."
        ),
    )
    parser.add_argument("--version", action="version", version=f"acp-im-gateway {__version__}")

    subparsers = parser.add_subparsers(
        dest="command", metavar="command", required=True, title="commands"
    )

    run = subparsers.add_parser(
        "run", parents=[common], help="start the gateway (long polling)"
    )
    run.add_argument("--max-polls", type=int, default=None, help=argparse.SUPPRESS)

    pairing = subparsers.add_parser(
        "pairing", parents=[common], help="manage pairing codes (approve a sender)"
    )
    pairing_sub = pairing.add_subparsers(dest="pairing_command", metavar="action", required=True)
    pairing_sub.add_parser("list", help="list pending pairing codes")
    approve = pairing_sub.add_parser("approve", help="add the code's sender to the allowlist")
    approve.add_argument("code", help="pairing code printed in the gateway log")
    reject = pairing_sub.add_parser("reject", help="discard a pairing code")
    reject.add_argument("code", help="pairing code printed in the gateway log")

    bindings = subparsers.add_parser(
        "bindings", parents=[common], help="inspect or edit chat -> project bindings"
    )
    bindings_sub = bindings.add_subparsers(
        dest="bindings_command", metavar="action", required=True
    )
    bindings_sub.add_parser("list", help="list bindings")
    add = bindings_sub.add_parser("add", help="bind a chat to a project path")
    add.add_argument("chat_id", type=int)
    add.add_argument("path", help="directory inside an allowed root")
    remove = bindings_sub.add_parser("remove", help="drop a binding")
    remove.add_argument("chat_id", type=int)

    subparsers.add_parser(
        "projects", parents=[common], help="list discovered project candidates"
    )
    subparsers.add_parser(
        "config", parents=[common], help="show the effective configuration (token redacted)"
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    overrides: dict[str, Any] = {}
    if getattr(args, "state_file", None):
        overrides["state_file"] = Path(args.state_file).expanduser()
    if getattr(args, "allowed_roots", None):
        overrides["allowed_roots"] = parse_path_list(args.allowed_roots)
    if getattr(args, "allowed_user_ids", None):
        overrides["allowed_user_ids"] = parse_id_list(args.allowed_user_ids)
    if getattr(args, "allowed_chat_ids", None):
        overrides["allowed_chat_ids"] = parse_id_list(args.allowed_chat_ids)
    if getattr(args, "agent_cmd", None):
        overrides["agent_cmd"] = split_command(args.agent_cmd)
    if getattr(args, "projects_root", None):
        overrides["projects_root"] = Path(args.projects_root).expanduser()
    if getattr(args, "log_level", None):
        overrides["log_level"] = str(args.log_level).upper()
    if getattr(args, "dry_run", False):
        overrides["dry_run"] = True
    return Config.from_env(overrides=overrides)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        stream=sys.stderr,
    )


def _router_from_config(config: Config) -> Router:
    access = AccessPolicy.from_config(
        allowed_user_ids=config.allowed_user_ids,
        allowed_chat_ids=config.allowed_chat_ids,
        pairing_ttl=config.pairing_ttl,
    )
    store = StateStore(config.state_file)
    discovery = ProjectDiscovery(
        config.projects_root,
        agent_index_dir=config.agent_index_dir,
        max_depth=config.discovery_depth,
        allowed_roots=config.resolved_roots(),
    )
    router = Router(
        store=store,
        discovery=discovery,
        access=access,
        allowed_roots=config.resolved_roots(),
    )
    router.load()
    return router


# --------------------------------------------------------------------------- commands


def _cmd_run(args: argparse.Namespace) -> int:
    config = _config_from_args(args).with_overrides(
        dry_run=True if getattr(args, "dry_run", False) else None
    )
    config.validate(require_token=not config.dry_run)
    _setup_logging(config.log_level)
    log = logging.getLogger("acp_im_gateway")
    gateway = Gateway(config, log=log)

    def _handle_signal(signum: int, _frame: Any) -> None:
        log.warning("signal %s received; stopping (current poll finishes first)", signum)
        gateway.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except ValueError:  # pragma: no cover - not the main thread
            pass

    log.info(
        "starting: agent=%s projects_root=%s roots=%s state=%s dry_run=%s",
        " ".join(config.agent_cmd),
        config.projects_root,
        ",".join(str(root) for root in config.resolved_roots()),
        config.state_file,
        config.dry_run,
    )
    try:
        gateway.run(max_iterations=getattr(args, "max_polls", None))
    except AcpError as exc:
        log.error("agent error: %s", exc)
        return EXIT_ERROR
    except ConfigError as exc:
        log.error("%s", exc)
        return EXIT_CONFIG
    except Exception as exc:  # keep the failure legible for systemd
        log.error("gateway failed: %s", exc)
        log.debug("traceback", exc_info=True)
        return EXIT_ERROR
    finally:
        gateway.stop()
    return EXIT_OK


def _cmd_pairing(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    router = _router_from_config(config)
    action = args.pairing_command

    if action == "list":
        pending = router.access.pending_requests()
        if not pending:
            print("No pending pairing codes.")
            return EXIT_OK
        now = time.time()
        print(f"{'code':<10} {'user_id':<14} {'chat_id':<14} {'username':<18} expires in")
        for request in pending:
            remaining = max(0, int(request.expires_at - now))
            status = "approved" if request.approved_at is not None else f"{remaining}s"
            print(
                f"{request.code:<10} {request.user_id:<14} {request.chat_id:<14} "
                f"{(request.username or '-'):<18} {status}"
            )
        return EXIT_OK

    code = args.code
    try:
        if action == "approve":
            request = router.access.approve(code)
        else:
            request = router.access.reject(code)
    except PairingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    router.save()
    if action == "approve":
        print(
            f"approved user {request.user_id} ({request.username or 'no username'}) "
            f"in chat {request.chat_id}"
        )
        print(
            "The running gateway picks this up on its next poll and will tell the user. "
            "It is persisted, so it also survives a restart."
        )
    else:
        print(f"rejected pairing code {request.code}")
    return EXIT_OK


def _cmd_bindings(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    router = _router_from_config(config)
    action = args.bindings_command

    if action == "list":
        if not router.bindings:
            print("No bindings yet. Use `bindings add <chat_id> <path>` or /bind in Telegram.")
            return EXIT_OK
        print(f"{'chat_id':<16} {'project':<24} {'session':<10} path")
        for chat_id, binding in sorted(router.bindings.items()):
            print(
                f"{chat_id:<16} {binding.project_name:<24} {binding.short_session():<10} "
                f"{binding.project_root}"
            )
        return EXIT_OK

    if action == "add":
        try:
            binding = router.bind(args.chat_id, args.path)
        except ContainmentError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        except DiscoveryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(f"bound chat {binding.chat_id} to {binding.project_root}")
        return EXIT_OK

    removed = router.unbind(args.chat_id)
    print(f"removed binding for chat {args.chat_id}" if removed else "no such binding")
    return EXIT_OK if removed else EXIT_ERROR


def _cmd_projects(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    router = _router_from_config(config)
    rows = router.list_projects()
    if not rows:
        print(
            f"No projects discovered.\n"
            f"  scanned: {config.projects_root} (depth {config.discovery_depth})\n"
            f"  agent index: {config.agent_index_dir}"
        )
        return EXIT_OK
    print(f"Allowed roots: {router.roots_label()}")
    for row in rows:
        flags = []
        if row["bound_chat_ids"]:
            flags.append("bound by " + ",".join(str(chat) for chat in row["bound_chat_ids"]))
        if not row["bindable"]:
            flags.append("outside allowed roots")
        suffix = f"  [{'; '.join(flags)}]" if flags else ""
        print(f"  {row['name']:<28} {row['path']}{suffix}")
    return EXIT_OK


def _cmd_config(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    print(f"acp-im-gateway {__version__}\n")
    width = max(len(key) for key, _ in config.safe_table())
    for key, value in config.safe_table():
        print(f"  {key:<{width}}  {value}")
    return EXIT_OK


# --------------------------------------------------------------------------- entry


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command
    handlers = {
        "run": _cmd_run,
        "pairing": _cmd_pairing,
        "bindings": _cmd_bindings,
        "projects": _cmd_projects,
        "config": _cmd_config,
    }
    try:
        return handlers[command](args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except ContainmentError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
