#!/usr/bin/env python3
"""A scriptable fake ACP agent used by the unit tests.

It speaks newline-delimited JSON-RPC 2.0 on stdio, exactly like a real ACP agent,
but its behaviour is controlled by flags so tests can force the awkward cases:
non-JSON stdout lines, unknown notifications, inbound requests, mid-turn
permission prompts, crashes and respawns.

Every request it receives is appended to ``--log <path>`` as one JSON object per
line, which is how tests assert what actually crossed the wire.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Mapping


def emit(obj: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


class FakeAgent:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sessions: dict[str, str] = {}
        self.handled = 0
        self.permission_seq = 1000
        self.waiting_for: dict[int, str] = {}  # response id -> "permission" | "unknown"
        self.pending_prompts: dict[str, int] = {}  # sessionId -> request id
        self.cancelled: set[str] = set()

    # ------------------------------------------------------------------ logging

    def log(self, event: str, **fields: Any) -> None:
        if not self.args.log:
            return
        with open(self.args.log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **fields}, default=str) + "\n")

    # ------------------------------------------------------------------ payloads

    def capabilities(self) -> dict[str, Any]:
        caps: dict[str, Any] = {
            "loadSession": bool(self.args.load_session),
            "sessionCapabilities": {"list": {}, "resume": {}, "close": {}, "delete": {}},
            "promptCapabilities": {"embeddedContext": True, "image": False, "audio": False},
        }
        if self.args.steer:
            caps["_meta"] = {"reasonix.io": {"sessionSteer": {"method": "_reasonix.io/session/steer"}}}
        return caps

    def config_options(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "model",
                "name": "Model",
                "category": "model",
                "type": "select",
                "currentValue": "fake-model",
                "options": [{"value": "fake-model", "name": "fake-model"}],
            },
            {
                "id": "tool_approval",
                "name": "Tool Approval",
                "category": "tool_approval",
                "type": "select",
                "currentValue": "ask",
                "options": [
                    {"value": "ask", "name": "Ask"},
                    {"value": "auto", "name": "Auto"},
                ],
            },
        ]

    def respond(self, message: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        emit({"jsonrpc": "2.0", "id": message.get("id"), "result": dict(result)})

    def respond_error(self, message: Mapping[str, Any], code: int, text: str) -> None:
        emit({"jsonrpc": "2.0", "id": message.get("id"), "error": {"code": code, "message": text}})

    def notify_update(self, session_id: str, session_update: str, **fields: Any) -> None:
        emit(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "update": {"sessionUpdate": session_update, **fields},
                },
            }
        )

    # ------------------------------------------------------------------ handling

    def handle_request(self, message: Mapping[str, Any]) -> None:
        method = str(message.get("method"))
        params: Mapping[str, Any] = message.get("params") or {}
        self.handled += 1
        self.log("request", method=method, params=params, pid=os.getpid())

        if self.args.junk:
            sys.stdout.write("this line is not JSON at all\n")
            sys.stdout.flush()

        if method == "initialize":
            self.respond(
                message,
                {
                    "protocolVersion": 1,
                    "agentInfo": {"name": "fake-agent", "version": "0.0.1"},
                    "agentCapabilities": self.capabilities(),
                },
            )
        elif method == "session/new":
            session_id = f"sess-{len(self.sessions) + 1}"
            self.sessions[session_id] = str(params.get("cwd") or "")
            self.respond(
                message,
                {
                    "sessionId": session_id,
                    "configOptions": self.config_options(),
                    "modes": {"currentModeId": "normal", "availableModes": []},
                    "mcpServers": params.get("mcpServers") or [],
                },
            )
        elif method == "session/load":
            if not self.args.load_session:
                self.respond_error(message, -32601, "session/load is not supported")
                return
            session_id = str(params.get("sessionId"))
            self.sessions.setdefault(session_id, str(params.get("cwd") or ""))
            self.respond(
                message,
                {
                    "sessionId": session_id,
                    "configOptions": self.config_options(),
                    "modes": {"currentModeId": "normal"},
                },
            )
        elif method == "session/list":
            cwd = params.get("cwd")
            sessions = [
                {
                    "sessionId": session_id,
                    "cwd": session_cwd,
                    "updatedAt": f"2026-01-01T00:00:0{index}Z",
                }
                for index, (session_id, session_cwd) in enumerate(sorted(self.sessions.items()))
                if cwd is None or session_cwd == cwd
            ]
            self.respond(message, {"sessions": sessions})
        elif method == "session/close":
            self.sessions.pop(str(params.get("sessionId")), None)
            self.respond(message, {})
        elif method == "session/cancel":
            self.cancel_session(str(params.get("sessionId")))
            self.respond(message, {})
        elif method == "_reasonix.io/session/steer":
            if not self.args.steer:
                self.respond_error(message, -32601, "steering is not supported")
                return
            self.respond(message, {})
        elif method == "session/prompt":
            self.start_prompt(message, params)
        else:
            self.respond_error(message, -32601, f"unknown method {method}")

    def start_prompt(self, message: Mapping[str, Any], params: Mapping[str, Any]) -> None:
        session_id = str(params.get("sessionId"))
        blocks = params.get("prompt") or []
        text = "".join(str(block.get("text") or "") for block in blocks if isinstance(block, Mapping))
        self.log("prompt", sessionId=session_id, text=text)
        request_id = message.get("id") or 0
        # The turn stays open until the agent answers this request id, exactly like
        # a real ACP agent (which streams updates for the whole turn).
        self.pending_prompts[session_id] = int(request_id)

        if self.args.crash_on_prompt:
            self.notify_update(
                session_id, "agent_message_chunk", content={"type": "text", "text": "partial "}
            )
            sys.stdout.flush()
            os._exit(9)

        self.notify_update(
            session_id, "agent_thought_chunk", content={"type": "text", "text": "considering"}
        )
        self.notify_update(
            session_id, "agent_message_chunk", content={"type": "text", "text": f"echo: {text}"}
        )
        self.notify_update(
            session_id,
            "tool_call",
            toolCallId="call-1",
            title="Run tests",
            kind="execute",
            status="pending",
        )
        self.notify_update(session_id, "tool_call_update", toolCallId="call-1", status="completed")
        emit(
            {
                "jsonrpc": "2.0",
                "method": "fake/unknown_notification",
                "params": {"sessionId": session_id, "note": "ignored by the client"},
            }
        )

        if self.args.permission:
            self.permission_seq += 1
            self.waiting_for[self.permission_seq] = "permission"
            emit(
                {
                    "jsonrpc": "2.0",
                    "id": self.permission_seq,
                    "method": "session/request_permission",
                    "params": {
                        "sessionId": session_id,
                        "toolCall": {
                            "toolCallId": "call-1",
                            "title": "Run the test suite",
                            "kind": "execute",
                            "status": "pending",
                            "rawInput": {"command": "pytest -q"},
                        },
                        "options": [
                            {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
                            {"optionId": "allow_always", "name": "Always", "kind": "allow_always"},
                            {"optionId": "reject_once", "name": "Reject", "kind": "reject_once"},
                        ],
                    },
                }
            )
            return

        if self.args.unknown_request:
            self.permission_seq += 1
            self.waiting_for[self.permission_seq] = "unknown"
            emit(
                {
                    "jsonrpc": "2.0",
                    "id": self.permission_seq,
                    "method": "fs/read_text_file",
                    "params": {"sessionId": session_id, "path": "/etc/hosts"},
                }
            )
            return

        self.finish_prompt(session_id, "end_turn")

    def finish_prompt(self, session_id: str, stop_reason: str) -> None:
        request_id = self.pending_prompts.pop(session_id, None)
        if request_id is None:
            return
        self.log("prompt_finished", sessionId=session_id, stopReason=stop_reason)
        emit({"jsonrpc": "2.0", "id": request_id, "result": {"stopReason": stop_reason}})

    def cancel_session(self, session_id: str) -> None:
        self.cancelled.add(session_id)
        self.log("cancel", sessionId=session_id)
        if session_id in self.pending_prompts:
            self.finish_prompt(session_id, "cancelled")

    def handle_response(self, message: Mapping[str, Any]) -> None:
        response_id = int(message.get("id") or 0)
        kind = self.waiting_for.pop(response_id, None)
        if kind is None:
            self.log("unexpected_response", message=message)
            return
        self.log(f"{kind}_response", message=message)
        self.finish_prompt_for_waiting("end_turn")

    def finish_prompt_for_waiting(self, stop_reason: str) -> None:
        for session_id in list(self.pending_prompts):
            self.finish_prompt(session_id, stop_reason)
            return

    # ------------------------------------------------------------------ loop

    def run(self) -> int:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            if "id" in message and "method" in message:
                self.handle_request(message)
            elif "id" in message:
                self.handle_response(message)
            elif "method" in message:
                params = message.get("params") or {}
                self.log("notification", method=message.get("method"), params=params)
                if message.get("method") == "session/cancel":
                    self.cancel_session(str(params.get("sessionId")))
            if self.args.exit_after and self.handled >= self.args.exit_after:
                sys.stdout.flush()
                return 0
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fake ACP agent for tests")
    parser.add_argument("--steer", action="store_true", help="advertise session steering")
    parser.add_argument("--load-session", action="store_true", help="advertise session/load")
    parser.add_argument("--junk", action="store_true", help="emit non-JSON stdout lines")
    parser.add_argument("--permission", action="store_true", help="ask permission mid-turn")
    parser.add_argument("--unknown-request", action="store_true", help="send a request we must decline")
    parser.add_argument("--crash-on-prompt", action="store_true", help="die mid-turn")
    parser.add_argument("--exit-after", type=int, default=0, help="exit after N requests")
    parser.add_argument("--log", help="append received messages here as JSON lines")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return FakeAgent(args).run()


if __name__ == "__main__":
    sys.exit(main())
