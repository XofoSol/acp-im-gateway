"""Telegram Bot API client (stdlib ``urllib``) plus message rendering helpers.

Design notes that matter for correctness:

* **Long polling only** — ``getUpdates`` with an offset and a server-side timeout.
  No webhook, no public URL.
* **One message per turn** — :class:`MessageStream` edits a single message in place
  while the agent streams, coalescing updates and enforcing a minimum interval
  (default 1.2s) between edits. Never one message per token.
* **Chunking** — anything over 4096 characters is split into ordered parts, and
  code fences left open by a split are closed and reopened so every part renders.
* **Plain text** — no ``parse_mode``: a broken Markdown message that fails to send
  is worse than plain text.
* **Rate limits** — at most one new message per second per chat, and ``429``
  answers are honoured with their ``retry_after`` instead of crashing.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

DEFAULT_API_BASE = "https://api.telegram.org"
MAX_MESSAGE_LENGTH = 4096
MIN_CHUNK_LIMIT = 256  # below this, fence bookkeeping cannot be honoured safely

_logger = logging.getLogger("acp_im_gateway.telegram")


class TelegramError(Exception):
    """The Bot API refused a call (or the transport failed after retries)."""

    def __init__(self, message: str, *, status: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class TelegramTransportError(TelegramError):
    """Network-level failure (DNS, timeout, connection reset)."""


# --------------------------------------------------------------------------- chunking

_FENCE_LINE = re.compile(r"^\s*```(.*)$")

#: Tags :func:`chunk_message` keeps balanced when ``html=True``. Only tags the
#: gateway itself emits are tracked; agent text is escaped, so it can never
#: smuggle one in.
HTML_TAGS = ("pre", "code", "b", "i", "u", "s", "tg-spoiler", "blockquote")
_HTML_TAG = re.compile(
    r"</?(?P<tag>" + "|".join(re.escape(tag) for tag in HTML_TAGS) + r")(?:\s[^>]*)?>",
    re.IGNORECASE,
)
#: Room kept for the tags that close at the end of a part.
_HTML_TAG_RESERVE = 64


def scan_fences(text: str, state: str | None = None) -> str | None:
    """Return the language of the code fence left open by ``text`` (None if balanced).

    ``state`` is the fence language carried in from the previous part.
    """
    open_lang = state
    for line in text.split("\n"):
        match = _FENCE_LINE.match(line)
        if match is None:
            continue
        if open_lang is None:
            open_lang = match.group(1).strip()
        else:
            open_lang = None
    return open_lang


def _choose_cut(text: str, budget: int) -> int:
    """Pick a cut point <= ``budget``, preferring a line, then a word boundary."""
    window = text[:budget]
    newline = window.rfind("\n")
    if newline >= max(1, budget // 2):
        return newline + 1
    space = window.rfind(" ")
    if space >= max(1, budget // 2):
        return space + 1
    return budget


def chunk_message(text: str, limit: int = MAX_MESSAGE_LENGTH, *, html: bool = False) -> list[str]:
    """Split ``text`` into ordered parts of at most ``limit`` characters.

    With ``html=False`` (the default) a part that would leave a ``` code fence
    open gets a closing fence appended, and the next part is prefixed with the
    same fence, so each part is valid on its own; the original characters are
    always preserved in order.

    With ``html=True`` the same guarantee is given for the HTML tags the gateway
    emits (``<pre>``, ``<code>``, ``<tg-spoiler>``, …): an open tag is closed at
    the end of a part and reopened at the start of the next one, so Telegram's
    HTML parser never sees a broken message.
    """
    if html:
        return _chunk_html(text, limit)
    if limit < MIN_CHUNK_LIMIT:
        raise ValueError(f"limit must be >= {MIN_CHUNK_LIMIT}, got {limit}")
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    remaining = text
    open_lang: str | None = None
    while remaining:
        prefix = "" if open_lang is None else f"```{open_lang}\n"
        end_state = scan_fences(remaining, open_lang)
        suffix = "" if end_state is None else "\n```"
        if len(prefix) + len(remaining) + len(suffix) <= limit:
            parts.append(prefix + remaining + suffix)
            break

        budget = limit - len(prefix) - len("\n```")
        if budget <= 0:  # pragma: no cover - guarded by MIN_CHUNK_LIMIT
            raise ValueError(f"limit {limit} is too small to split safely")
        cut = _choose_cut(remaining, budget)
        body = remaining[:cut]
        remaining = remaining[cut:]
        open_lang = scan_fences(body, open_lang)
        parts.append(prefix + body + ("" if open_lang is None else "\n```"))
    return parts


def scan_html(text: str, state: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return the HTML tags left open by ``text`` (as a stack, innermost last)."""
    stack = list(state or ())
    for match in _HTML_TAG.finditer(text):
        tag = match.group("tag").lower()
        if match.group(0).startswith("</"):
            if stack and stack[-1] == tag:
                stack.pop()
            elif tag in stack:
                stack.remove(tag)
        else:
            stack.append(tag)
    return tuple(stack)


def _close_tags(stack: Iterable[str]) -> str:
    return "".join(f"</{tag}>" for tag in reversed(list(stack)))


def _open_tags(stack: Iterable[str]) -> str:
    return "".join(f"<{tag}>" for tag in stack)


def _avoid_tag_split(body: str) -> str:
    """Never cut inside a tag: a part ending in ``</pr`` is not a tag Telegram sees."""
    start = body.rfind("<")
    if start != -1 and ">" not in body[start:]:
        return body[:start]
    return body


def _chunk_html(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split while keeping the gateway's own HTML tags balanced in every part."""
    if limit < MIN_CHUNK_LIMIT:
        raise ValueError(f"limit must be >= {MIN_CHUNK_LIMIT}, got {limit}")
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    remaining = text
    state: tuple[str, ...] = ()
    while remaining:
        prefix = _open_tags(state)
        room = limit - len(prefix) - _HTML_TAG_RESERVE
        if room < 1:
            raise ValueError(f"limit {limit} is too small to split HTML safely")
        if len(prefix) + len(remaining) + len(_close_tags(state)) <= limit:
            parts.append(prefix + remaining + _close_tags(state))
            break

        body = _avoid_tag_split(remaining[:room])
        body = _avoid_tag_split(body[: _choose_cut(body, len(body))])
        end_state = scan_html(body, state)
        while body and len(prefix) + len(body) + len(_close_tags(end_state)) > limit:
            body = body.rsplit("\n", 1)[0] if "\n" in body else body[:-8]
            body = _avoid_tag_split(body)
            end_state = scan_html(body, state)
        if not body:  # pragma: no cover - a single line longer than `limit`
            body = _avoid_tag_split(remaining[: max(1, limit - len(prefix) - _HTML_TAG_RESERVE)])
            end_state = scan_html(body, state)
        parts.append(prefix + body + _close_tags(end_state))
        remaining = remaining[len(body) :]
        state = end_state
    return parts


def reassemble(parts: Sequence[str], *, strip_wrapper_fences: bool = True) -> str:
    """Join chunked parts back together (used by tests and for sanity checks).

    With ``strip_wrapper_fences`` the fences that :func:`chunk_message` injected
    around a split are removed, recovering the original text.
    """
    if not strip_wrapper_fences:
        return "".join(parts)
    out: list[str] = []
    for index, part in enumerate(parts):
        body = part
        if index > 0 and body.startswith("```"):
            newline = body.find("\n")
            if newline != -1:
                body = body[newline + 1 :]
        if index < len(parts) - 1 and body.endswith("\n```"):
            body = body[: -len("\n```")]
        out.append(body)
    return "".join(out)


# --------------------------------------------------------------------------- rendering


def inline_keyboard(rows: Iterable[Iterable[tuple[str, str]]]) -> dict[str, Any]:
    """Build an InlineKeyboardMarkup from ``[[(label, callback_data), ...], ...]``."""
    return {
        "inline_keyboard": [
            [{"text": label, "callback_data": data} for label, data in row] for row in rows
        ]
    }


def plain(text: str) -> str:
    """Sanitise outgoing text: too long is handled by chunking, so only trim NULs."""
    return text.replace("\x00", "")


# --------------------------------------------------------------------------- rate limits


class Throttle:
    """Minimum interval between operations sharing a key (per chat, by default)."""

    def __init__(
        self,
        min_interval: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = max(float(min_interval), 0.0)
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._slots: dict[Hashable, float] = {}

    def wait_for(self, key: Hashable = "global") -> float:
        """Sleep until this key's next slot, returning how long we waited."""
        if self.min_interval <= 0:
            return 0.0
        with self._lock:
            now = self._clock()
            previous = self._slots.get(key)
            start = now if previous is None else max(now, previous + self.min_interval)
            self._slots[key] = start
            wait = start - now
        if wait > 0:
            self._sleeper(wait)
        return wait


# --------------------------------------------------------------------------- client


class TelegramClient:
    """Thin Bot API client. ``dry_run=True`` prints instead of calling the network."""

    def __init__(
        self,
        token: str,
        *,
        api_base: str = DEFAULT_API_BASE,
        timeout: float = 40.0,
        dry_run: bool = False,
        send_interval: float = 1.0,
        max_retries: int = 2,
        log: logging.Logger | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        throttle: Throttle | None = None,
    ) -> None:
        if not token and not dry_run:
            raise TelegramError("TELEGRAM_BOT_TOKEN is required")
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.timeout = float(timeout)
        self.dry_run = dry_run
        self.max_retries = max(0, int(max_retries))
        self.log = log or _logger
        self._clock = clock
        self._sleeper = sleeper
        self._throttle = throttle or Throttle(send_interval, clock=clock, sleeper=sleeper)
        self._dry_counter = 1000
        self._dry_lock = threading.Lock()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    # ------------------------------------------------------------------ transport

    @property
    def base_url(self) -> str:
        return f"{self.api_base}/bot{self.token}"

    def _http_post(self, url: str, payload: Mapping[str, Any], timeout: float | None = None) -> Any:
        """Overridable seam: POST JSON and return the decoded Bot API response."""
        body = json.dumps(plain_json(payload)).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                raise TelegramError(
                    f"HTTP {exc.code} from Telegram: {raw[:200]}", status=exc.code
                ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TelegramTransportError(f"cannot reach Telegram: {exc}") from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - Bot API is always JSON
            raise TelegramTransportError(f"invalid JSON from Telegram: {raw[:200]}") from exc

    def call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
        respect_rate_limit: bool = False,
    ) -> Any:
        """Call a Bot API method, honouring 429/retry_after and transient errors."""
        payload = dict(params or {})
        if self.dry_run:
            with self._dry_lock:
                self.calls.append((method, payload))
                result = self._dry_result(method, payload)
            self.log.info("[dry-run] %s %s", method, _summarise(payload))
            return result

        if respect_rate_limit:
            chat_id = payload.get("chat_id")
            self._throttle.wait_for(chat_id if chat_id is not None else "global")

        url = f"{self.base_url}/{method}"
        attempt = 0
        while True:
            try:
                response = self._http_post(url, payload, timeout=timeout)
            except TelegramTransportError as exc:
                if attempt >= self.max_retries:
                    raise
                backoff = min(2.0**attempt, 8.0)
                self.log.warning("%s failed (%s); retrying in %.1fs", method, exc, backoff)
                self._sleeper(backoff)
                attempt += 1
                continue

            if isinstance(response, Mapping) and response.get("ok"):
                return response.get("result")

            code, description, retry_after = _read_error(response)
            if code == 429 and attempt < self.max_retries:
                delay = retry_after if retry_after is not None else 1.0
                self.log.warning("rate limited on %s; sleeping %.1fs", method, delay)
                self._sleeper(delay)
                attempt += 1
                continue
            if code is not None and code >= 500 and attempt < self.max_retries:
                backoff = min(2.0**attempt, 8.0)
                self.log.warning("Telegram %s returned %s; retrying in %.1fs", method, code, backoff)
                self._sleeper(backoff)
                attempt += 1
                continue
            raise TelegramError(
                f"{method} failed: {description or 'unknown error'}", status=code, retry_after=retry_after
            )

    def _dry_result(self, method: str, payload: Mapping[str, Any]) -> Any:
        if method == "getUpdates":
            # Long polling has no network here: mirror the server-side wait.
            timeout = float(payload.get("timeout") or 0)
            if timeout > 0:
                self._sleeper(timeout)
            return []
        if method in ("sendMessage", "editMessageText", "sendPhoto", "sendDocument"):
            self._dry_counter += 1
            return {
                "message_id": self._dry_counter,
                "chat": {"id": payload.get("chat_id")},
                "text": payload.get("text"),
                "dry_run": True,
            }
        if method == "getMe":
            return {"id": 0, "is_bot": True, "username": "dry-run-bot"}
        return {"dry_run": True}

    # ------------------------------------------------------------------ methods

    def get_me(self, *, timeout: float | None = None) -> Mapping[str, Any]:
        return self.call("getMe", timeout=timeout)

    def get_updates(
        self,
        offset: int | None = None,
        *,
        timeout: int = 30,
        allowed_updates: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"timeout": int(timeout)}
        if offset is not None:
            params["offset"] = int(offset)
        params["allowed_updates"] = list(allowed_updates or ("message", "callback_query"))
        result = self.call("getUpdates", params, timeout=float(timeout) + 15.0)
        return list(result or [])

    def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: Mapping[str, Any] | None = None,
        reply_to_message_id: int | None = None,
        message_thread_id: int | None = None,
        parse_mode: str | None = None,
        disable_notification: bool = False,
        chunk: bool = False,
    ) -> list[dict[str, Any]]:
        """Send text. ``chunk=True`` splits long text into ordered parts.

        ``parse_mode`` defaults to ``None`` (plain text): a broken Markdown or
        HTML message that fails to send is worse than plain text. The turn
        transcript opts into ``"HTML"`` with its own escaping.
        """
        pieces = chunk_message(plain(text), html=parse_mode == "HTML") if chunk else [plain(text)]
        sent: list[dict[str, Any]] = []
        for index, piece in enumerate(pieces):
            params: dict[str, Any] = {
                "chat_id": chat_id,
                "text": piece,
                "disable_web_page_preview": True,
            }
            if parse_mode is not None:
                params["parse_mode"] = parse_mode
            if disable_notification:
                params["disable_notification"] = True
            if message_thread_id is not None:
                params["message_thread_id"] = int(message_thread_id)
            if reply_markup is not None and index == 0:
                params["reply_markup"] = dict(reply_markup)
            if reply_to_message_id is not None and index == 0:
                params["reply_to_message_id"] = int(reply_to_message_id)
            result = self.call("sendMessage", params, respect_rate_limit=True)
            if isinstance(result, Mapping):
                sent.append(dict(result))
        return sent

    def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        reply_markup: Mapping[str, Any] | None = None,
        parse_mode: str | None = None,
    ) -> Mapping[str, Any] | None:
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": int(message_id),
            "text": plain(text),
            "disable_web_page_preview": True,
        }
        if parse_mode is not None:
            params["parse_mode"] = parse_mode
        if reply_markup is not None:
            params["reply_markup"] = dict(reply_markup)
        try:
            result = self.call("editMessageText", params)
        except TelegramError as exc:
            # "message is not modified" is a benign race with our own coalescing.
            if "not modified" in str(exc).lower():
                self.log.debug("edit skipped: %s", exc)
                return None
            raise
        return dict(result) if isinstance(result, Mapping) else None

    def delete_message(self, chat_id: int, message_id: int) -> bool:
        try:
            self.call("deleteMessage", {"chat_id": chat_id, "message_id": int(message_id)})
            return True
        except TelegramError as exc:
            self.log.debug("delete_message failed: %s", exc)
            return False

    def send_chat_action(self, chat_id: int, action: str = "typing") -> bool:
        try:
            self.call("sendChatAction", {"chat_id": chat_id, "action": action})
            return True
        except TelegramError as exc:
            self.log.debug("send_chat_action failed: %s", exc)
            return False

    def answer_callback_query(
        self, callback_query_id: str, *, text: str | None = None, show_alert: bool = False
    ) -> bool:
        params: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            params["text"] = text
            params["show_alert"] = bool(show_alert)
        try:
            self.call("answerCallbackQuery", params)
            return True
        except TelegramError as exc:
            self.log.debug("answer_callback_query failed: %s", exc)
            return False

    def set_my_commands(self, commands: Sequence[tuple[str, str]]) -> bool:
        payload = {"commands": [{"command": name, "description": desc} for name, desc in commands]}
        try:
            self.call("setMyCommands", payload)
            return True
        except TelegramError as exc:
            self.log.warning("setMyCommands failed: %s", exc)
            return False


def plain_json(value: Any) -> Any:
    """Recursively drop ``None`` values so payloads stay tidy."""
    if isinstance(value, Mapping):
        return {key: plain_json(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [plain_json(item) for item in value]
    return value


def _read_error(response: Any) -> tuple[int | None, str | None, float | None]:
    if not isinstance(response, Mapping):
        return None, None, None
    code = response.get("error_code")
    description = response.get("description")
    retry_after = None
    parameters = response.get("parameters")
    if isinstance(parameters, Mapping) and parameters.get("retry_after") is not None:
        try:
            retry_after = float(parameters["retry_after"])
        except (TypeError, ValueError):
            retry_after = None
    return (int(code) if isinstance(code, int) else None), description, retry_after


def _summarise(payload: Mapping[str, Any]) -> str:
    parts = []
    for key, value in payload.items():
        if isinstance(value, str) and len(value) > 80:
            value = value[:80] + "…"
        parts.append(f"{key}={value!r}")
    return " ".join(parts)


# --------------------------------------------------------------------------- streaming


@dataclass
class _SentMessage:
    chat_id: int
    message_id: int
    text: str


@dataclass
class MessageStream:
    """Keep one Telegram message (plus ordered overflow parts) in sync with a text.

    ``push()`` records the newest desired text; ``flush()`` does the actual
    Bot API work, and never more often than ``edit_interval`` seconds. The caller
    is expected to call :meth:`flush` periodically (the gateway does it on every
    idle poll of the prompt loop) and :meth:`close` when the turn ends.

    :meth:`seal` is what turns a growing transcript into an ordered log: it
    freezes every message written so far (they are never edited again) and makes
    the next ``flush`` start a new message, which the gateway does once a message
    would pass ``overflow_limit`` characters.
    """

    client: TelegramClient
    chat_id: int
    edit_interval: float = 1.2
    limit: int = MAX_MESSAGE_LENGTH
    overflow_limit: int = 3500
    reply_markup: Mapping[str, Any] | None = None
    parse_mode: str | None = None
    thread_id: int | None = None
    log: logging.Logger = field(default=_logger)
    clock: Callable[[], float] = time.monotonic

    _text: str = ""
    _sent_text: str | None = None
    _messages: list[_SentMessage] = field(default_factory=list)
    _sealed_text: str = ""
    _sealed_count: int = 0
    _last_flush: float = 0.0
    _closed: bool = False

    @property
    def text(self) -> str:
        return self._text

    @property
    def messages(self) -> list[_SentMessage]:
        return list(self._messages)

    @property
    def sealed_messages(self) -> list[_SentMessage]:
        """The messages already frozen: they will never be edited again."""
        return list(self._messages[: self._sealed_count])

    @property
    def sealed_text(self) -> str:
        """The text covered by the sealed messages (a prefix of ``text``)."""
        return self._sealed_text

    @property
    def message_id(self) -> int | None:
        return self._messages[0].message_id if self._messages else None

    @property
    def live_message_id(self) -> int | None:
        """The message still being edited, if any."""
        if len(self._messages) <= self._sealed_count:
            return None
        return self._messages[self._sealed_count].message_id

    @property
    def up_to_date(self) -> bool:
        """True when every message already shows the newest pushed text."""
        return self._sent_text == self._text

    def push(self, text: str) -> bool:
        """Record new desired text; flush immediately when the interval allows."""
        self._text = text
        return self.flush()

    def seal(self, covered: str | None = None) -> int:
        """Freeze every message written so far and start a new one.

        Returns the number of sealed messages, or 0 when nothing was frozen — in
        particular when the newest text is not on screen yet, because freezing
        there would leave a gap between the frozen message and the next one.
        ``covered`` is the text the sealed messages hold; it defaults to whatever
        was last pushed.
        """
        if self._closed or not self._messages or not self.up_to_date:
            return 0
        self._sealed_count = len(self._messages)
        self._sealed_text = self._text if covered is None else covered
        # Force the next flush to write the remainder, even inside the interval:
        # the frozen part must not be part of it.
        self._sent_text = None
        return self._sealed_count

    def flush(self, *, force: bool = False) -> bool:
        """Sync Telegram with the desired text. Returns True when something changed."""
        if self._closed:
            return False
        if self._sent_text == self._text and not force:
            return False
        if not force and self._messages:
            elapsed = self.clock() - self._last_flush
            if elapsed < self.edit_interval:
                return False

        tail = self._text[len(self._sealed_text) :]
        parts = chunk_message(tail, self.limit, html=self.parse_mode == "HTML") or [""]
        changed = False
        for index, part in enumerate(parts):
            position = self._sealed_count + index
            if position < len(self._messages):
                message = self._messages[position]
                if message.text == part:
                    continue
                self.client.edit_message_text(
                    message.chat_id,
                    message.message_id,
                    part,
                    parse_mode=self.parse_mode,
                )
                message.text = part
                changed = True
            else:
                markup = self.reply_markup if index == 0 else None
                sent = self.client.send_message(
                    self.chat_id,
                    part,
                    reply_markup=markup,
                    parse_mode=self.parse_mode,
                    message_thread_id=self.thread_id,
                )
                if not sent:  # pragma: no cover - dry-run/network failure path
                    break
                result = sent[0]
                self._messages.append(
                    _SentMessage(
                        chat_id=int(result.get("chat", {}).get("id", self.chat_id)),
                        message_id=int(result.get("message_id", 0)),
                        text=part,
                    )
                )
                changed = True
        self._sent_text = self._text
        self._last_flush = self.clock()
        return changed

    def close(self, final_text: str | None = None) -> bool:
        """Flush the final text unconditionally and stop accepting updates."""
        if final_text is not None:
            self._text = final_text
        changed = self.flush(force=True)
        self._closed = True
        return changed
