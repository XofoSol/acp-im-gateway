"""Shared test helpers: doubles, a fake clock, waiting and the gateway harness.

Everything here is hermetic: :class:`FakeTelegram` never touches the network and
the agent side is ``fake_agent.py`` on stdio. No test ever sends a
``session/prompt`` to a real agent.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # make `import acp_im_gateway` work from a checkout
    sys.path.insert(0, str(REPO_ROOT))

FAKE_AGENT = REPO_ROOT / "tests" / "fake_agent.py"
PYTHON = sys.executable

from acp_im_gateway.config import Config  # noqa: E402
from acp_im_gateway.gateway import Gateway  # noqa: E402
from acp_im_gateway.telegram import chunk_message  # noqa: E402

LOG = logging.getLogger("tests.harness")
CHAT = 111
USER = 900


# --------------------------------------------------------------------------- clock


class FakeClock:
    """A monotonic clock that only moves when the test says so."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += max(float(seconds), 0.0)

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


# --------------------------------------------------------------------------- telegram


@dataclass
class RecordedMessage:
    """A Telegram message as it currently looks (edits mutate it in place)."""

    method: str
    chat_id: int
    message_id: int
    text: str
    reply_markup: Mapping[str, Any] | None = None
    parse_mode: str | None = None
    message_thread_id: int | None = None
    seq: int = 0


class FakeTelegram:
    """Records every Bot API call instead of touching the network.

    ``messages`` holds the *current state* of each message (like Telegram itself:
    an edit changes the message, it does not add one), ``sent_messages`` and
    ``edits_log`` keep the full call history.
    """

    def __init__(self) -> None:
        self.messages: list[RecordedMessage] = []
        self.sent_messages: list[RecordedMessage] = []
        self.edits_log: list[RecordedMessage] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.callback_answers: list[tuple[str, str | None]] = []
        self.chat_actions: list[tuple[int, str]] = []
        self.fail_send: Exception | None = None
        self.fail_edit: Exception | None = None
        self._next_id = 500
        self._seq = 0

    # -- helpers ---------------------------------------------------------------

    def _bump(self) -> int:
        self._seq += 1
        return self._seq

    def reset(self) -> None:
        for collection in (self.messages, self.sent_messages, self.edits_log, self.calls):
            collection.clear()

    def sent(self, chat_id: int | None = None) -> list[RecordedMessage]:
        return [
            message
            for message in self.sent_messages
            if chat_id is None or message.chat_id == chat_id
        ]

    def edits(self, chat_id: int | None = None) -> list[RecordedMessage]:
        return [
            message for message in self.edits_log if chat_id is None or message.chat_id == chat_id
        ]

    def texts(self, chat_id: int | None = None) -> list[str]:
        return [message.text for message in self.sent(chat_id)]

    def calls_named(self, method: str) -> list[dict[str, Any]]:
        return [payload for name, payload in self.calls if name == method]

    def last_sent_text(self, chat_id: int) -> str | None:
        sent = self.sent(chat_id)
        return sent[-1].text if sent else None

    def current_text(self, chat_id: int) -> str | None:
        """The newest visible text in the chat across sends and edits."""
        relevant = [message for message in self.messages if message.chat_id == chat_id]
        return max(relevant, key=lambda message: message.seq).text if relevant else None

    # -- Bot API surface -------------------------------------------------------

    def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: Mapping[str, Any] | None = None,
        chunk: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self.fail_send is not None:
            raise self.fail_send
        parse_mode = kwargs.get("parse_mode")
        thread_id = kwargs.get("message_thread_id")
        pieces = chunk_message(text, html=parse_mode == "HTML") if chunk else [text]
        out: list[dict[str, Any]] = []
        for index, piece in enumerate(pieces):
            self._next_id += 1
            markup = reply_markup if index == 0 else None
            self.calls.append(
                (
                    "sendMessage",
                    {
                        "chat_id": chat_id,
                        "text": piece,
                        "reply_markup": markup,
                        "parse_mode": parse_mode,
                        "message_thread_id": thread_id,
                    },
                )
            )
            record = RecordedMessage(
                method="sendMessage",
                chat_id=int(chat_id),
                message_id=self._next_id,
                text=piece,
                reply_markup=markup,
                parse_mode=parse_mode,
                message_thread_id=thread_id,
                seq=self._bump(),
            )
            self.messages.append(record)
            self.sent_messages.append(replace(record))  # snapshot: edits mutate state only
            out.append({"message_id": self._next_id, "chat": {"id": chat_id}})
        return out

    def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        reply_markup: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.fail_edit is not None:
            raise self.fail_edit
        chat_id = int(chat_id)
        message_id = int(message_id)
        parse_mode = kwargs.get("parse_mode")
        self.calls.append(
            (
                "editMessageText",
                {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
            )
        )
        record = RecordedMessage(
            method="editMessageText",
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            seq=self._bump(),
        )
        self.edits_log.append(record)
        for existing in self.messages:
            if existing.chat_id == chat_id and existing.message_id == message_id:
                existing.method = "editMessageText"
                existing.text = text
                existing.reply_markup = reply_markup
                existing.parse_mode = parse_mode
                existing.seq = record.seq
                break
        else:  # pragma: no cover - editing a message we never sent
            self.messages.append(record)
        return {"message_id": message_id, "chat": {"id": chat_id}}

    def send_chat_action(self, chat_id: int, action: str = "typing", **kwargs: Any) -> bool:
        self.calls.append(("sendChatAction", {"chat_id": chat_id, "action": action}))
        self.chat_actions.append((int(chat_id), action))
        return True

    def answer_callback_query(
        self, callback_query_id: str, *, text: str | None = None, **kwargs: Any
    ) -> bool:
        self.calls.append(
            ("answerCallbackQuery", {"callback_query_id": callback_query_id, "text": text})
        )
        self.callback_answers.append((str(callback_query_id), text))
        return True

    def set_my_commands(self, commands: Iterable[tuple[str, str]]) -> bool:
        self.calls.append(("setMyCommands", {"commands": list(commands)}))
        return True

    def delete_message(self, chat_id: int, message_id: int) -> bool:
        self.calls.append(("deleteMessage", {"chat_id": chat_id, "message_id": message_id}))
        return True

    def get_updates(self, offset: int | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(("getUpdates", {"offset": offset}))
        return []


# --------------------------------------------------------------------------- updates


def message_update(
    text: str,
    *,
    chat_id: int = 111,
    user_id: int = 900,
    chat_type: str = "private",
    update_id: int = 1,
    username: str = "tester",
    message_thread_id: int | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "message_id": update_id,
        "date": 0,
        "chat": {"id": chat_id, "type": chat_type},
        "from": {"id": user_id, "username": username, "is_bot": False},
        "text": text,
    }
    if message_thread_id is not None:
        message["message_thread_id"] = message_thread_id
        message["is_topic_message"] = True
    return {"update_id": update_id, "message": message}


def callback_update(
    data: str,
    *,
    chat_id: int = 111,
    user_id: int = 900,
    chat_type: str = "private",
    update_id: int = 2,
    message_id: int = 1,
    callback_id: str = "cb-1",
) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": callback_id,
            "from": {"id": user_id, "is_bot": False},
            "data": data,
            "message": {
                "message_id": message_id,
                "chat": {"id": chat_id, "type": chat_type},
            },
        },
    }


def approval_buttons(fake: FakeTelegram, chat_id: int) -> list[dict[str, Any]]:
    """Return the inline keyboard buttons of the newest approval prompt."""
    for message in reversed(fake.messages):
        if message.method != "sendMessage" or message.chat_id != chat_id:
            continue
        markup = message.reply_markup or {}
        if isinstance(markup, Mapping) and markup.get("inline_keyboard"):
            return [button for row in markup["inline_keyboard"] for button in row]
    return []


def wait_until(predicate: Callable[[], bool], *, timeout: float = 8.0, interval: float = 0.02) -> bool:
    """Poll ``predicate`` until true. Returns False on timeout (assert in the test)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def html_text(fake: FakeTelegram, chat_id: int, message_id: int | None = None) -> str:
    """The newest visible text of a chat (or of one message)."""
    relevant = [message for message in fake.messages if message.chat_id == chat_id]
    if message_id is not None:
        relevant = [message for message in relevant if message.message_id == message_id]
    if not relevant:
        return ""
    return max(relevant, key=lambda message: message.seq).text


def read_agent_log(path: Path) -> list[dict[str, Any]]:
    """Parse the fake agent's JSON-lines log."""
    if not Path(path).is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def events_named(events: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    return [event for event in events if event.get("event") == name]


# --------------------------------------------------------------------------- projects


def make_project(parent: Path, name: str, *, git: bool = True) -> Path:
    path = parent / name
    path.mkdir(parents=True, exist_ok=True)
    if git:
        (path / ".git").mkdir(exist_ok=True)
    return path


# --------------------------------------------------------------------------- harness


@dataclass
class Harness:
    """A gateway wired to the fake agent and the in-memory Telegram double."""

    gateway: Gateway
    telegram: FakeTelegram
    log_path: Path
    chat_id: int = CHAT

    # -- driving ---------------------------------------------------------------

    def send(self, text: str, **kwargs: Any) -> None:
        kwargs.setdefault("chat_id", self.chat_id)
        kwargs.setdefault("user_id", USER)
        self.gateway.handle_update(message_update(text, **kwargs))

    def callback(self, data: str, **kwargs: Any) -> None:
        kwargs.setdefault("chat_id", self.chat_id)
        kwargs.setdefault("user_id", USER)
        self.gateway.handle_update(callback_update(data, **kwargs))

    # -- inspecting ------------------------------------------------------------

    @property
    def runtime(self) -> Any:
        return self.gateway.router.runtime(self.chat_id)

    def replies(self) -> list[str]:
        return self.telegram.texts(self.chat_id)

    def last_reply(self) -> str:
        sent = self.telegram.sent(self.chat_id)
        return sent[-1].text if sent else ""

    def current_text(self) -> str:
        return self.telegram.current_text(self.chat_id) or ""

    def agent_events(self) -> list[dict[str, Any]]:
        return read_agent_log(self.log_path)

    def agent_requests(self, method: str) -> list[dict[str, Any]]:
        """JSON-RPC requests the agent received, by method."""
        return [
            event
            for event in self.agent_events()
            if event.get("event") == "request" and event.get("method") == method
        ]

    def prompts(self) -> list[dict[str, Any]]:
        """Prompts the agent received, with the text exactly as it arrived."""
        return events_named(self.agent_events(), "prompt")

    def any_text(self, needle: str) -> bool:
        return any(
            needle in message.text
            for message in self.telegram.messages
            if message.chat_id == self.chat_id
        )

    def all_text(self) -> str:
        """Every message this chat has ever shown (sends and edits, in order)."""
        return "\n".join(message.text for message in self.telegram.messages)

    def visible_text(self) -> str:
        """The chat as the owner sees it: every message in order, current state."""
        return "\n".join(
            message.text for message in self.telegram.messages if message.chat_id == self.chat_id
        )

    def permission_responses(self) -> list[dict[str, Any]]:
        return events_named(self.agent_events(), "permission_response")

    def permission_decisions(self) -> list[dict[str, Any]]:
        return events_named(self.agent_events(), "permission_decision")

    def buttons(self) -> list[dict[str, Any]]:
        return approval_buttons(self.telegram, self.chat_id)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        return wait_until(lambda: not self.runtime.busy, timeout=timeout)

    def wait_for_buttons(self, timeout: float = 10.0) -> bool:
        return wait_until(lambda: bool(self.buttons()), timeout=timeout)

    def state(self) -> dict[str, Any]:
        return json.loads(self.gateway.store.path.read_text(encoding="utf-8"))


@pytest.fixture()
def make_harness(tmp_path: Path, config: Config) -> Iterator[Callable[..., Harness]]:
    """Build gateways around the scriptable fake agent; stop them afterwards."""
    created: list[Gateway] = []

    def factory(
        *agent_flags: str,
        allowed_users: Sequence[int] = (USER,),
        allowed_chats: Sequence[int] = (),
        chat_id: int = CHAT,
        **overrides: Any,
    ) -> Harness:
        log_path = tmp_path / f"agent-{len(created)}.jsonl"
        cfg = replace(
            config,
            agent_cmd=(PYTHON, str(FAKE_AGENT), "--log", str(log_path), *agent_flags),
            allowed_user_ids=frozenset(allowed_users),
            allowed_chat_ids=frozenset(allowed_chats),
            **overrides,
        )
        telegram = FakeTelegram()
        gateway = Gateway(cfg, telegram=telegram, log=LOG)
        gateway.start()
        created.append(gateway)
        return Harness(gateway=gateway, telegram=telegram, log_path=log_path, chat_id=chat_id)

    yield factory
    for gateway in created:
        gateway.stop()
