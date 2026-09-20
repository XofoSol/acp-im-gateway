"""Turn rendering: ``session/update`` notifications -> the chat transcript.

The chat must read like the CLI conversation. A turn is therefore rendered as an
*ordered transcript* of blocks, not one blob:

* the agent's text, as it arrives;
* every tool call, first class: ``$ <the exact command> [<status>]`` for shell
  calls, ``<icon> <title> [<status>]`` otherwise, with the real output fenced
  underneath (truncated to ``GATEWAY_TOOL_OUTPUT_LINES``);
* file writes and edits as ``✏️ <path>`` plus a short diff or excerpt;
* the agent's reasoning, available but quiet, inside a Telegram
  ``<tg-spoiler>`` (or omitted entirely when ``GATEWAY_SHOW_THINKING`` is off);
* lifecycle notices: start (⏳ project + model), finish (✅ with the outcome and
  a test summary when a test command ran), cancelled (⏹️), error (❌).

Everything the agent produces is HTML-escaped here, so agent output can never
break a Telegram message sent with ``parse_mode="HTML"``.
"""

from __future__ import annotations

import difflib
import html
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_TOOL_OUTPUT_LINES = 15
#: A transcript message past this many characters is frozen and a new one starts.
DEFAULT_OVERFLOW_CHARS = 3500
#: Longest reasoning excerpt kept in the spoiler.
MAX_THOUGHT_CHARS = 900

SPOILER_OPEN = "<tg-spoiler>"
SPOILER_CLOSE = "</tg-spoiler>"

TOOL_ICONS: dict[str, str] = {
    "read": "📖",
    "edit": "✏️",
    "write": "✏️",
    "execute": "▶️",
    "think": "💭",
    "fetch": "🌐",
    "search": "🔎",
}

#: Stop reason -> the mark that ends the turn.
STOP_MARKS: dict[str, str] = {
    "end_turn": "✅ done",
    "cancelled": "⏹️ cancelled",
    "canceled": "⏹️ cancelled",
    "max_tokens": "⚠️ stopped: token limit",
    "max_turn_requests": "⚠️ stopped: request limit",
    "refusal": "🚫 refused by the agent",
}

_PLACEHOLDER = "💭 working…"

#: Commands that run a test suite: their output feeds the finish summary.
TEST_COMMANDS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"^(?:\S*/)?(?:python[0-9.]*|py)\s+-m\s+pytest\b",
        r"^(?:\S*/)?pytest\b",
        r"^(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:test|t)\b",
        r"^(?:npx\s+)?(?:jest|vitest|mocha|ava)\b",
        r"^(?:\S*/)?go\s+test\b",
        r"^(?:\S*/)?cargo\s+test\b",
        r"^(?:\S*/)?make\s+(?:test|check|tests)\b",
        r"^(?:\S*/)?(?:tox|nox|rspec|phpunit|pytest-xdist)\b",
        r"^(?:\S*/)?dotnet\s+test\b",
        r"^(?:\S*/)?(?:mvn|gradle|\./gradlew)\b[^\n]*\btest\b",
        r"^(?:\S*/)?(?:django-admin|manage\.py)\s+test\b",
    )
)

_SUMMARY_FIELDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("passed", re.compile(r"(\d+)\s+passed")),
    ("failed", re.compile(r"(\d+)\s+failed")),
    ("errors", re.compile(r"(\d+)\s+errors?")),
    ("skipped", re.compile(r"(\d+)\s+skipped")),
)

_COMMAND_KEYS = ("command", "cmd", "script", "commandLine", "command_line", "shell")
_PATH_KEYS = ("file_path", "filePath", "path", "paths", "files", "target")

#: Command separators. A single ``&`` counts too: ``ls & rm -fr x`` is two
#: commands, and the second one must not hide behind the first one's whitelist.
_SPLIT_OPERATORS = re.compile(r"&&|\|\||;|\||\n|&")


# --------------------------------------------------------------------------- escaping


def escape(text: Any) -> str:
    """Escape ``text`` for Telegram's HTML parse mode.

    ``<``, ``>`` and ``&`` are the only characters Telegram requires escaped; a
    quote is left alone so the transcript stays readable.
    """
    return html.escape(str(text), quote=False)


# --------------------------------------------------------------------------- summaries


@dataclass(frozen=True)
class TestSummary:
    """Counts parsed out of a test runner's output."""

    __test__ = False  # not a test class: pytest must not try to collect it

    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    def label(self) -> str:
        """``"12 passed, 1 failed"`` — empty when nothing was recognised."""
        bits: list[str] = []
        if self.passed:
            bits.append(f"{self.passed} passed")
        if self.failed:
            bits.append(f"{self.failed} failed")
        if self.errors:
            bits.append(f"{self.errors} error" + ("s" if self.errors != 1 else ""))
        if self.skipped:
            bits.append(f"{self.skipped} skipped")
        return ", ".join(bits)


def is_test_command(command: str) -> bool:
    """True when ``command`` looks like it runs a test suite."""
    first = command.strip().split("\n", 1)[0].strip()
    return any(pattern.search(first) for pattern in TEST_COMMANDS)


def parse_test_summary(output: str) -> TestSummary | None:
    """Read ``N passed`` / ``N failed`` … out of ``output`` (last occurrence wins)."""
    counts: dict[str, int] = {}
    for name, pattern in _SUMMARY_FIELDS:
        matches = pattern.findall(output)
        if matches:
            counts[name] = max(int(value) for value in matches)
    if not counts:
        return None
    return TestSummary(
        passed=counts.get("passed", 0),
        failed=counts.get("failed", 0),
        errors=counts.get("errors", 0),
        skipped=counts.get("skipped", 0),
    )


def format_duration(seconds: float | None) -> str:
    """``0.4s`` / ``1m 12s`` / ``1h 03m`` — short and unambiguous."""
    if seconds is None:
        return ""
    total = max(0, int(round(seconds)))
    if total < 60:
        return f"{seconds:.1f}s" if seconds < 10 else f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total // 3600}h {(total % 3600) // 60:02d}m"


# --------------------------------------------------------------------------- notices


def start_notice(project: str | None, model: str | None) -> str:
    """``⏳ alpha · model fake-model`` — the first block of a turn."""
    if project and model:
        return f"⏳ {escape(project)} · model {escape(model)}"
    if project:
        return f"⏳ {escape(project)}"
    if model:
        return f"⏳ model {escape(model)}"
    return "⏳ working…"


def finish_notice(
    stop_reason: Any,
    *,
    elapsed: float | None = None,
    tool_calls: int = 0,
    test: TestSummary | None = None,
) -> str:
    """``✅ done · 1 tool call(s) · 12.4s · tests: 5 passed``."""
    mark = STOP_MARKS.get(str(stop_reason), f"⏹️ stopped ({stop_reason or 'unknown'})")
    bits: list[str] = []
    if tool_calls:
        bits.append(f"{tool_calls} tool call(s)")
    duration = format_duration(elapsed)
    if duration:
        bits.append(duration)
    if test is not None and test.label():
        bits.append(f"tests: {test.label()}")
    return mark if not bits else f"{mark} · " + " · ".join(bits)


def error_notice(reason: Any) -> str:
    """``❌ <the reason>`` — the turn did not finish."""
    text = str(reason).strip() or "the turn failed"
    return f"❌ {text}"


def heartbeat_notice(elapsed: float) -> str:
    """Shown only when a turn has been visibly silent for a while."""
    return f"⏳ still working · {format_duration(elapsed)} elapsed"


# --------------------------------------------------------------------------- content


def content_text(content: Any) -> str:
    """Extract text from an ACP content block (or a list of them)."""
    if isinstance(content, Mapping):
        if content.get("type") in (None, "text"):
            return str(content.get("text") or "")
        return ""
    if isinstance(content, (list, tuple)):
        return "".join(content_text(item) for item in content)
    if isinstance(content, str):
        return content
    return ""


def _short(value: Any, limit: int = 200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, default=str)
    if value is None:
        return ""
    return str(value)


def _raw_input_commands(raw_input: Any) -> list[str]:
    """Shell command strings advertised by a tool call."""
    if isinstance(raw_input, str):
        return [raw_input] if raw_input.strip() else []
    if not isinstance(raw_input, Mapping):
        return []
    commands: list[str] = []
    for key in _COMMAND_KEYS:
        value = raw_input.get(key)
        if isinstance(value, str) and value.strip():
            commands.append(value)
        elif isinstance(value, (list, tuple)):
            commands.extend(part for part in value if isinstance(part, str) and part.strip())
    return commands


def _raw_input_paths(raw_input: Any) -> list[str]:
    if not isinstance(raw_input, Mapping):
        return []
    paths: list[str] = []
    for key in _PATH_KEYS:
        value = raw_input.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value)
        elif isinstance(value, (list, tuple)):
            paths.extend(part for part in value if isinstance(part, str) and part.strip())
    return paths


def _tool_paths(tool_call: Mapping[str, Any]) -> list[str]:
    """File locations a tool call touches (``locations`` plus ``rawInput`` paths)."""
    paths: list[str] = []
    locations = tool_call.get("locations")
    if isinstance(locations, (list, tuple)):
        for location in locations:
            if isinstance(location, Mapping):
                path = location.get("path") or location.get("file")
                if isinstance(path, str) and path.strip():
                    paths.append(path)
            elif isinstance(location, str) and location.strip():
                paths.append(location)
    paths.extend(_raw_input_paths(tool_call.get("rawInput")))
    out: list[str] = []
    for path in paths:
        if path not in out:
            out.append(path)
    return out


# --------------------------------------------------------------------------- output


def _diff_excerpt(block: Mapping[str, Any], *, lines: int) -> str:
    """A short unified diff (or an excerpt) for one ``diff`` content block."""
    path = str(block.get("path") or "")
    old = block.get("oldText")
    new = block.get("newText")
    header = f"--- {path}\n+++ {path}\n" if path else ""
    if isinstance(old, str) and isinstance(new, str):
        diff = list(
            difflib.unified_diff(
                old.splitlines(), new.splitlines(), fromfile="", tofile="", n=1, lineterm=""
            )
        )
        # Drop difflib's own headers: we render our own, without timestamps.
        while diff and diff[0].startswith(("---", "+++")):
            diff.pop(0)
        return header + "\n".join(diff[: max(1, lines)])
    body = new if isinstance(new, str) else (old if isinstance(old, str) else "")
    return header + "\n".join(str(body).splitlines()[: max(1, lines)])


def tool_output(update: Mapping[str, Any], *, lines: int = DEFAULT_TOOL_OUTPUT_LINES) -> tuple[str, str]:
    """Return ``(stdout, diff)`` carried by a tool-call update.

    ACP sends tool results as content blocks (``{"type":"content", ...}`` for
    text, ``{"type":"diff", ...}`` for file edits) and some agents add a
    ``rawOutput`` object with ``stdout``/``stderr``. Both shapes are read.
    """
    texts: list[str] = []
    diffs: list[str] = []
    content = update.get("content")
    if isinstance(content, (list, tuple)):
        for block in content:
            if not isinstance(block, Mapping):
                continue
            kind = str(block.get("type") or "")
            if kind == "diff":
                excerpt = _diff_excerpt(block, lines=max(2, min(lines, 8)))
                if excerpt.strip():
                    diffs.append(excerpt)
            elif kind == "text":
                texts.append(str(block.get("text") or ""))
            elif kind in ("", "content"):
                texts.append(content_text(block.get("content", block)))
    elif isinstance(content, Mapping):
        texts.append(content_text(content))
    elif isinstance(content, str):
        texts.append(content)

    raw = update.get("rawOutput")
    if isinstance(raw, Mapping):
        for key in ("stdout", "stderr", "output", "result"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)
    elif isinstance(raw, str):
        texts.append(raw)

    # Only trailing whitespace goes: the leading column of `git status` output
    # (or of an indented diff) is part of the real output.
    body = "\n".join(part for part in texts if part and part.strip()).rstrip()
    return body, "\n".join(part for part in diffs if part.strip()).rstrip()


def merge_output(previous: str, new: str) -> str:
    """Accumulate streamed tool output without duplicating a repeated payload."""
    previous = previous or ""
    new = new or ""
    if not new:
        return previous
    if not previous:
        return new
    if new in previous:
        return previous
    if previous in new:
        return new
    return f"{previous}\n{new}"


def truncate_lines(text: str, limit: int) -> tuple[str, int]:
    """Keep the **last** ``limit`` lines; return ``(body, omitted)``.

    The tail of a command's output is where the verdict lives (a failing test,
    the error, the summary), so the tail is what survives.
    """
    lines = text.splitlines()
    if limit <= 0 or len(lines) <= limit:
        return text, 0
    omitted = len(lines) - limit
    return "\n".join(lines[-limit:]), omitted


# --------------------------------------------------------------------------- transcript


@dataclass
class TurnView:
    """Reduce a turn's ``session/update`` stream into ordered chat blocks.

    ``blocks()`` is the ordered log: the start notice, then text runs, tool
    calls (with their output) and reasoning in the order they happened, then
    whatever is still streaming, then the finish notice. Only the *last* block
    can still change, which is what lets the gateway freeze messages safely.
    """

    project: str | None = None
    model: str | None = None
    show_thinking: bool = True
    tool_output_lines: int = DEFAULT_TOOL_OUTPUT_LINES
    placeholder: str = _PLACEHOLDER

    thought_chunks: int = 0
    tool_calls: int = 0
    test_run: TestSummary | None = None

    _start: str | None = None
    _finish: str | None = None
    _segments: list[str] = field(default_factory=list)
    _buffer: list[str] = field(default_factory=list)
    _thought: list[str] = field(default_factory=list)
    _tool_segments: dict[str, int] = field(default_factory=dict)
    _tool_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    _seen_tools: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------ notices

    def start(self, project: str | None = None, model: str | None = None) -> str:
        """Freeze the start notice (⏳ with the project and the model)."""
        if project is not None:
            self.project = project
        if model is not None:
            self.model = model
        self._start = start_notice(self.project, self.model)
        return self._start

    def finish(self, stop_reason: Any, *, elapsed: float | None = None) -> str:
        """Freeze the finish notice, carrying the test summary when one exists."""
        self._finish = finish_notice(
            stop_reason,
            elapsed=elapsed,
            tool_calls=self.tool_calls,
            test=self.test_run,
        )
        return self._finish

    def fail(self, reason: Any) -> str:
        """Freeze the error notice (❌ with the reason)."""
        self._finish = error_notice(reason)
        return self._finish

    @property
    def finished(self) -> str | None:
        return self._finish

    @property
    def edits(self) -> int:
        """How many distinct file writes/edits the turn made."""
        return sum(
            1
            for state in self._tool_state.values()
            if str(state.get("kind") or "").strip().lower() in ("edit", "write")
        )

    def has_content(self) -> bool:
        """True once anything but the start notice has been rendered."""
        return bool(self._segments or self._buffer or self._thought or self._finish)

    # ------------------------------------------------------------------ reducing

    def apply(self, notification: Mapping[str, Any]) -> None:
        """Fold one ``session/update`` notification into the transcript."""
        params = notification.get("params") or {}
        update = params.get("update") or {}
        if not isinstance(update, Mapping):
            return
        kind = str(update.get("sessionUpdate") or "")

        if kind == "agent_message_chunk":
            self._flush_thought()
            self._buffer.append(content_text(update.get("content")))
        elif kind == "agent_thought_chunk":
            self._apply_thought(update)
        elif kind == "tool_call":
            self._apply_tool_call(update)
        elif kind == "tool_call_update":
            self._apply_tool_call(update, is_update=True)
        elif kind == "plan":
            self._apply_plan(update)
        # user_message_chunk / available_commands_update / current_mode_update /
        # config_option_update and anything unknown are intentionally ignored.

    def _apply_thought(self, update: Mapping[str, Any]) -> None:
        self.thought_chunks += 1
        text = content_text(update.get("content")).strip()
        if not self.show_thinking or not text:
            # Nothing to hide: the generic "thinking…" placeholder covers this.
            return
        self._flush_text()
        self._thought.append(text)

    def _flush_text(self) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer.clear()
        if text.strip():
            self._segments.append(escape(text))

    def _flush_thought(self) -> None:
        if not self._thought:
            return
        body = "".join(self._thought)
        self._thought.clear()
        body = body.strip()
        if not body:
            return
        if len(body) > MAX_THOUGHT_CHARS:
            body = body[:MAX_THOUGHT_CHARS].rstrip() + "…"
        self._segments.append(f"💭 {SPOILER_OPEN}{escape(body)}{SPOILER_CLOSE}")

    # ------------------------------------------------------------------ tool calls

    def _apply_tool_call(self, update: Mapping[str, Any], *, is_update: bool = False) -> None:
        tool_id = str(update.get("toolCallId") or update.get("id") or f"tool-{self.tool_calls}")
        state = self._tool_state.get(tool_id)
        if state is None and is_update and tool_id in self._seen_tools:
            return
        if state is None:
            state = {"id": tool_id}
            self._tool_state[tool_id] = state
            if tool_id not in self._seen_tools:
                self.tool_calls += 1
                self._seen_tools.add(tool_id)
            self._flush_thought()
            self._flush_text()
            self._tool_segments[tool_id] = len(self._segments)
            self._segments.append("")

        for key in ("title", "kind", "status"):
            value = update.get(key)
            if value not in (None, ""):
                state[key] = value
        if isinstance(update.get("rawInput"), (Mapping, str)):
            state["rawInput"] = update["rawInput"]
        if isinstance(update.get("locations"), (list, tuple)):
            state["locations"] = list(update["locations"])

        body, diff = tool_output(update, lines=self.tool_output_lines)
        if body:
            state["output"] = merge_output(str(state.get("output") or ""), body)
        if diff:
            state["diff"] = merge_output(str(state.get("diff") or ""), diff)

        self._segments[self._tool_segments[tool_id]] = self._render_tool(state)
        self._note_test_run(state)

    def _note_test_run(self, state: Mapping[str, Any]) -> None:
        """Record the verdict of the last test command, for the finish notice."""
        command = _shell_command_of(state)
        if not command or not is_test_command(command):
            return
        output = str(state.get("output") or "")
        summary = parse_test_summary(output)
        if summary is not None:
            self.test_run = summary

    def _render_tool(self, state: Mapping[str, Any]) -> str:
        kind = str(state.get("kind") or "").strip().lower()
        status = str(state.get("status") or "").strip() or "pending"
        tool_id = str(state.get("id") or "tool")
        title = str(state.get("title") or "").strip() or tool_id
        command = _shell_command_of(state)
        paths = _paths_of(state)
        path = paths[0] if paths else ""
        icon = TOOL_ICONS.get(kind, "🔧")

        if kind == "execute" and command:
            head = f"$ {escape(command)} [{status}]"
        elif kind in ("edit", "write") and path:
            head = f"✏️ {escape(path)} [{status}]"
        elif kind == "read" and path:
            head = f"{icon} {escape(path)} [{status}]"
        else:
            head = f"{icon} {escape(title)} [{status}]"

        lines = [head]
        diff = str(state.get("diff") or "")
        if diff:
            lines.extend(self._pre_block(diff))
        output = str(state.get("output") or "")
        if output:
            lines.extend(self._pre_block(output))
        return "\n".join(lines)

    def _pre_block(self, text: str) -> list[str]:
        body, omitted = truncate_lines(text, self.tool_output_lines)
        parts: list[str] = []
        if omitted:
            parts.append(f"… (+{omitted} more lines)")
        parts.append(f"<pre>{escape(body)}</pre>")
        return parts

    # ------------------------------------------------------------------ plans

    def _apply_plan(self, update: Mapping[str, Any]) -> None:
        entries = update.get("entries") or update.get("plan") or []
        lines: list[str] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            status = str(entry.get("status") or entry.get("priority") or "")
            mark = "✅" if status in ("completed", "done") else "▫️"
            text = str(entry.get("content") or entry.get("text") or "").strip()
            if text:
                lines.append(f"{mark} {escape(text)}")
        if not lines:
            return
        self._flush_thought()
        self._flush_text()
        self._segments.append("📋 plan:\n" + "\n".join(lines))

    # ------------------------------------------------------------------ rendering

    def blocks(self) -> list[str]:
        """The ordered transcript. Only the last block can still change."""
        parts: list[str] = []
        if self._start:
            parts.append(self._start)
        parts.extend(self._segments)
        pending = "".join(self._thought) or "".join(self._buffer)
        if pending.strip():
            if self._thought:
                body = pending.strip()
                if len(body) > MAX_THOUGHT_CHARS:
                    body = body[:MAX_THOUGHT_CHARS].rstrip() + "…"
                parts.append(f"💭 {SPOILER_OPEN}{escape(body)}{SPOILER_CLOSE}")
            else:
                parts.append(escape(pending))
        if self._finish:
            parts.append(self._finish)
        return parts

    def _parts(self) -> list[str]:
        return [part for part in self.blocks() if part.strip()]

    def render(self) -> str:
        body = "\n\n".join(self._parts())
        if body:
            return body
        if self.thought_chunks:
            return f"💭 thinking… ({self.thought_chunks} chunk(s) of reasoning withheld)"
        return self.placeholder

    def freeze_point(self) -> int:
        """How much of :meth:`render` can never change again.

        A message may only be frozen up to this offset, otherwise a tool call that
        is still running (or an answer still being written) would be cut in half
        and its verdict would never make it to the chat.

        * every *whole* block before the last one is final;
        * for the agent's own message buffer, whole lines are final too: ACP text
          chunks are append-only deltas, so a finished line never changes.
        """
        parts = self._parts()
        if not parts:
            return 0
        offset = sum(len(part) + 2 for part in parts[:-1])
        if self._finish is not None or self._thought:
            return offset
        cut = parts[-1].rfind("\n")
        return offset + cut if cut > 0 else offset


def _shell_command_of(state: Mapping[str, Any]) -> str:
    commands = _raw_input_commands(state.get("rawInput"))
    if commands:
        return commands[0]
    value = state.get("command")
    if isinstance(value, str) and value.strip():
        return value
    return ""


def _paths_of(state: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    locations = state.get("locations")
    if isinstance(locations, (list, tuple)):
        for location in locations:
            if isinstance(location, Mapping):
                path = location.get("path") or location.get("file")
                if isinstance(path, str) and path.strip():
                    paths.append(path)
    paths.extend(_raw_input_paths(state.get("rawInput")))
    out: list[str] = []
    for path in paths:
        if path not in out:
            out.append(path)
    return out


def shell_commands(text: str) -> list[str]:
    """Split a shell line into its simple commands (``a && b | c`` -> ``a``, ``b``, ``c``)."""
    return [part.strip() for part in _SPLIT_OPERATORS.split(text or "") if part.strip()]


def _split_tokens(command: str) -> list[str]:
    import shlex

    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def _invocation(parts: Sequence[str]) -> list[str] | None:
    """The ``rm``/``curl`` argv inside ``parts``, wherever a wrapper put it.

    ``sudo -u root rm -fr /`` and ``env -i curl -d@f URL`` must still be seen; a
    wrapper's own option parsing is not what matters here, only that the real
    command does not slip past the gate.
    """
    for index, token in enumerate(parts):
        if os.path.basename(token) in _GATED_COMMANDS:
            return list(parts[index:])
    return None


#: curl flags that send a body or write something, in every spelling curl accepts.
_CURL_DANGEROUS = (
    "-d",
    "-F",
    "-T",
    "-o",
    "-O",
    "--data",
    "--form",
    "--json",
    "--upload-file",
    "--output",
    "--remote-name",
)
_CURL_METHODS = ("POST", "PUT", "PATCH", "DELETE")
#: Commands whose *own* argv is what the always_ask detectors must inspect, even
#: when a wrapper (``sudo -u root``, ``env -i``) put them further along.
_GATED_COMMANDS = ("rm", "curl")


def curl_creates_data(commands: Iterable[str]) -> bool:
    """True for a ``curl`` that creates data (``-d``, ``-F``, ``-X POST`` …).

    Covers the cramped spellings too: ``-d@file``, ``--data-binary=@f``,
    ``-XPOST``, ``--request=POST``, ``-T file``, ``-o out``.
    """
    for command in commands:
        if "curl" in command and "method-override" in command.lower():
            return True  # -H 'X-HTTP-Method-Override: POST' and friends
        for simple in shell_commands(command):
            parts = _invocation(_split_tokens(simple))
            if not parts or os.path.basename(parts[0]) != "curl":
                continue
            rest = parts[1:]
            for index, token in enumerate(rest):
                if token.startswith(_CURL_DANGEROUS):
                    return True
                if token in ("-X", "--request"):
                    if index + 1 < len(rest) and rest[index + 1].upper() in _CURL_METHODS:
                        return True
                    continue
                if token.startswith("--request="):
                    if token.split("=", 1)[1].upper() in _CURL_METHODS:
                        return True
                elif token.startswith("-X") and token[2:].upper() in _CURL_METHODS:
                    return True
    return False


def rm_forced_recursive(commands: Iterable[str]) -> bool:
    """True for ``rm -rf``/``rm -fr``/``rm -r -f``/``rm --recursive --force``."""
    for command in commands:
        for simple in shell_commands(command):
            parts = _invocation(_split_tokens(simple))
            if not parts or os.path.basename(parts[0]) != "rm":
                continue
            recursive = forced = False
            for flag in parts[1:]:
                if flag.startswith("--"):
                    recursive = recursive or flag == "--recursive"
                    forced = forced or flag == "--force"
                    continue
                if flag.startswith("-"):
                    recursive = recursive or "r" in flag[1:] or "R" in flag[1:]
                    forced = forced or "f" in flag[1:]
            if recursive and forced:
                return True
    return False


def keyword_match(pattern: str, haystack: str) -> bool:
    """Case-insensitive match.

    A single word-like pattern (``aws``, ``stripe``, ``fal_client``) must not
    land inside a longer word; anything else (``git push``, ``fal.ai``,
    ``rm -rf``) is a plain substring test. Both are *widening*: a false positive
    only costs a tap.
    """
    needle = pattern.strip().lower()
    if not needle:
        return False
    haystack = (haystack or "").lower()
    if re.fullmatch(r"[a-z0-9_]+", needle):
        return (
            re.search(rf"(?<![a-z0-9_]){re.escape(needle)}(?![a-z0-9_])", haystack) is not None
        )
    return needle in haystack
