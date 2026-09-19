# acp-im-gateway

Drive an **ACP-speaking coding agent** (Reasonix today — any ACP client agent tomorrow)
from a chat app, per project, with approvals answered from your phone.

Telegram first. Slack is roadmap.

```text
 Telegram  <->  gateway process (Python, on your host)
                 |- telegram adapter : long polling (getUpdates) + sendMessage/editMessageText
                 |- router           : chat_id -> project root -> ACP session (persisted JSON state)
                 |- ACP client       : JSON-RPC 2.0 over stdio  <->  `reasonix acp`
                 |- approval bridge  : session/request_permission -> inline buttons -> response
```

Two claims are worth stating up front, because they are the whole point:

* **No LLM in the middle.** Your text goes into the agent verbatim over ACP. The
  gateway is plumbing: it never rewrites, summarises or interprets prompts.
* **The runtime is the Python standard library only.** Python 3.11+, no third-party
  runtime dependencies. `pytest` is a development dependency and nothing else.

---

## Table of contents

- [What it is](#what-it-is)
- [How it talks to the agent](#how-it-talks-to-the-agent)
- [Install](#install)
- [Quick start](#quick-start)
- [Telegram setup](#telegram-setup)
- [Commands](#commands)
- [Configuration](#configuration)
- [Approvals](#approvals)
- [Access control and containment](#access-control-and-containment)
- [Security warning](#security-warning)
- [Honest limits](#honest-limits)
- [Running it as a service](#running-it-as-a-service)
- [Tests and verification](#tests-and-verification)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)
- [License](#license)

---

## What it is

A small gateway that turns a chat app into the remote control for a coding agent
that already lives on your machine:

* **Per project.** A chat is *bound* to a project directory. `/bind my-app` and
  every message in that chat runs in `my-app`. Different chats can drive different
  projects at the same time.
* **Streaming in one message.** While a turn runs, the gateway edits a single
  Telegram message in place — never one message per token. Edits are coalesced and
  rate limited (default: at most one edit every 1.2 s).
* **Approvals from your phone.** When the agent wants to run a gated tool call, the
  gateway turns `session/request_permission` into one message with inline buttons
  (Allow / Deny plus whatever options the agent advertised) and answers the ACP
  request with the option you tapped.
* **Sessions that survive a restart.** Bindings and session ids are persisted to a
  JSON state file; on start the gateway re-attaches (`session/load`) or resumes the
  most recent session for that `cwd` (`session/list`) before creating a new one.
* **Deny by default.** Unknown senders reach nothing. A directory can only be bound
  inside an allowed root, enforced with `os.path.realpath`, so `..` traversal and
  symlink escapes are rejected.

## How it talks to the agent

The agent command is configurable (`REASONIX_ACP_CMD`, default `reasonix acp`) and
speaks **ACP**: newline-delimited JSON-RPC 2.0 over stdin/stdout, one JSON object per
line. The gateway uses this surface:

| Direction | Method | Purpose |
| --- | --- | --- |
| client -> agent | `initialize` | handshake; reads `protocolVersion`, `agentInfo`, `agentCapabilities` |
| client -> agent | `session/new {cwd, mcpServers}` | new session; returns `sessionId` + `configOptions` (model, effort, work_mode, tool_approval) |
| client -> agent | `session/prompt {sessionId, prompt:[{type:"text", text}]}` | one turn; the request stays open while the agent streams |
| agent -> client | `session/update` | streamed chunks, tool calls, plans (rendered into the chat) |
| client -> agent | `session/cancel {sessionId}` | `/stop` |
| client -> agent | `session/list {cwd}` / `session/load` / `session/close` | resume/close sessions |
| agent -> client | `session/request_permission` | approvals; answered with the exact `optionId` the agent advertised |
| client -> agent | `_reasonix.io/session/steer` | vendor mid-turn guidance, **only** when the agent advertises it via `_meta` |

Nothing else is required, and the gateway tolerates the rest of the protocol:
unknown notifications are ignored, inbound requests other than permission are
declined cleanly (never left hanging), and stdout lines that are not JSON are
logged and skipped. The agent process is respawned with exponential backoff if it
crashes, and in-flight turns fail cleanly instead of hanging.

The default approval posture is `tool_approval = "ask"` (`GATEWAY_APPROVAL_POSTURE`):
when a session reports a different posture the gateway asks the agent to change it,
and if the agent does not expose or accept that option the agent's own setting stays
in charge and the log says so. `/status` shows the model, mode and posture in force.

## Install

Requires **Python 3.11+** and an ACP-speaking agent on `PATH` (Reasonix is the
reference: `reasonix acp`).

```bash
git clone https://github.com/XofoSol/acp-im-gateway
cd acp-im-gateway

# Option A: isolated virtualenv
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m acp_im_gateway --help

# Option B: pipx (console script on PATH)
pipx install .

# Development (adds pytest)
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

There are no runtime dependencies to resolve — installation is just unpacking the
package and (optionally) the `acp-im-gateway` console script.

## Quick start

```bash
cp .env.example .env          # then edit it (the file is git-ignored)
$EDITOR .env                  # TELEGRAM_BOT_TOKEN, ALLOWED_USER_IDS, PROJECTS_ROOT

python -m acp_im_gateway config    # shows the effective config, token redacted
python -m acp_im_gateway projects  # what it can bind (both discovery sources)
python -m acp_im_gateway run       # long polling; no webhook, no public URL
```

In the chat:

```text
/projects        ->  alpha — /home/you/Projects/alpha
/bind alpha      ->  ✅ Bound to alpha: /home/you/Projects/alpha
review the failing test in tests/test_api.py and fix it        <- sent verbatim to the agent
```

If a sender is not on the allowlist, the gateway replies that they are not
authorised, writes a short-lived one-time code **to its log**, and waits:

```text
WARNING acp_im_gateway.gateway: PAIRING: user 12345 (someone) in private chat 12345 is not
allowed. Code K7M2PQRS (valid 900s). Approve with: python -m acp_im_gateway pairing approve K7M2PQRS
```

```bash
python -m acp_im_gateway pairing list
python -m acp_im_gateway pairing approve K7M2PQRS
```

`pairing` / `bindings` / `projects` only touch the state file and the filesystem, so
they are safe to run while the gateway is up: a running gateway picks allowlist and
pairing changes up on its next poll, adopts bindings for chats it has never seen,
and tells the newly approved user. `--dry-run` prints what would be sent to Telegram
instead of calling the network (useful to check configuration before going live).

## Telegram setup

1. **Create the bot.** Talk to [@BotFather](https://t.me/BotFather) → `/newbot` → copy
   the token into `TELEGRAM_BOT_TOKEN`. Treat that token as a password: anyone who
   has it can read every update the bot receives.
2. **Find your user id.** Send any message to [@userinfobot](https://t.me/userinfobot)
   (or read the `from.id` in your updates) and put it in `ALLOWED_USER_IDS`. You can
   also leave it empty and pair yourself through the CLI flow above — the gateway
   denies everything by default, including you.
3. **Direct messages** need nothing else. **Groups** need two things:
   * `ALLOWED_CHAT_IDS` must list the group id (an empty list means *direct messages
     only*), and
   * the sender must still be allowlisted.
   In groups, disable privacy mode for the bot with `@BotFather → /setprivacy → your
   bot → Disable`, otherwise the bot only sees commands addressed to it.
4. **No webhook, ever.** The gateway only calls `getUpdates` (long polling, 30 s
   timeout) plus `sendMessage` / `editMessageText` / `answerCallbackQuery`. Nothing
   listens on a port, so there is no public URL to expose.

## Commands

| Command | What it does |
| --- | --- |
| `/projects` | Lists discovered projects; marks the bound one with ✅ and non-bindable ones (outside the allowed roots) |
| `/bind <name> [<path>]` | Binds this chat to a project. `<path>` is optional when the name matches exactly one discovery |
| `/new` | Starts a fresh session in this chat's project |
| `/stop` | Cancels the running turn and drops queued messages |
| `/status` | Bound project, session id, model, mode, approval posture, agent state, queue |
| `/unbind` | Forgets this chat's binding |
| `/help`, `/start` | The command list |

Anything else is a prompt: it goes to the agent verbatim. If the chat is not bound
yet, the gateway explains how to bind — it never guesses a project. A second message
during a turn is **steered** into the running turn when the agent advertises session
steering, otherwise it is **queued** (`GATEWAY_BUSY_MODE`, default `steer`).

CLI: `run`, `pairing list|approve|reject`, `bindings list|add|remove`,
`projects`, `config`. Run `python -m acp_im_gateway --help`.

## Configuration

Precedence: **built-in defaults < TOML file < environment / `.env` < CLI flags.**

`.env` is read from `./.env` (or `GATEWAY_ENV_FILE`), and real environment variables
always win over the file. An optional TOML file is read from
`~/.config/acp-im-gateway/config.toml` (or `GATEWAY_CONFIG`).

| Variable | Default | Meaning |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | *(empty)* | Bot token from @BotFather. Required to `run` |
| `TELEGRAM_API_BASE` | `https://api.telegram.org` | Bot API base URL (override for a local Bot API server) |
| `ALLOWED_USER_IDS` | *(empty)* | Comma-separated Telegram user ids that may drive the gateway. Empty = nobody (deny by default) |
| `ALLOWED_CHAT_IDS` | *(empty)* | Comma-separated chat ids allowed in **groups**. Empty = direct messages only |
| `PROJECTS_ROOT` | `~/Projects` | Root scanned for project candidates (directories containing `.git`) |
| `ALLOWED_ROOTS` | *(empty = `PROJECTS_ROOT`)* | Comma-separated roots a chat may bind a directory inside |
| `REASONIX_PROJECTS_DIR` | `~/.reasonix/projects` | The agent's own project index, used as discovery hints. Absent = fine |
| `GATEWAY_STATE_FILE` | `~/.local/state/acp-im-gateway/state.json` | Bindings, polling offset, allowlist, pairing codes (written 0600) |
| `GATEWAY_CONFIG` | `~/.config/acp-im-gateway/config.toml` | Optional TOML config file |
| `GATEWAY_ENV_FILE` | `./.env` | Optional dotenv file (read from the real environment only) |
| `REASONIX_ACP_CMD` | `reasonix acp` | Agent command; any ACP-speaking agent works |
| `GATEWAY_TURN_TIMEOUT` | `0` | Seconds before a turn is abandoned. `0` = no cap (the agent decides) |
| `GATEWAY_RESTART_BACKOFF_MAX` | `30` | Upper bound (seconds) of the agent respawn backoff |
| `GATEWAY_EDIT_INTERVAL` | `1.2` | Minimum seconds between edits of the streaming message |
| `GATEWAY_SEND_INTERVAL` | `1.0` | Minimum seconds between new outgoing messages per chat |
| `GATEWAY_POLL_TIMEOUT` | `30` | `getUpdates` long-poll timeout in seconds |
| `GATEWAY_BUSY_MODE` | `steer` | What to do with a message that arrives mid-turn: `steer` or `queue` |
| `GATEWAY_PAIRING_TTL` | `900` | How long a pairing code stays valid, seconds |
| `GATEWAY_APPROVAL_TIMEOUT` | `300` | How long an unanswered approval stays pending, seconds |
| `GATEWAY_APPROVAL_POSTURE` | `ask` | `tool_approval` posture requested from the agent: `ask`, `auto`, `yolo`, or empty to leave the agent's own setting |
| `GATEWAY_DISCOVERY_DEPTH` | `1` | How deep to scan `PROJECTS_ROOT` for `.git` |
| `GATEWAY_LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `GATEWAY_DRY_RUN` | `0` | Print what would be sent to Telegram instead of calling the API |

Equivalent TOML shape (all keys optional):

```toml
[telegram]
bot_token = ""
api_base = "https://api.telegram.org"
allowed_user_ids = [111111111]
allowed_chat_ids = [-1001234567890]

[paths]
projects_root = "~/Projects"
allowed_roots = ["~/Projects", "~/work"]
state_file = "~/.local/state/acp-im-gateway/state.json"
agent_index = "~/.reasonix/projects"

[agent]
command = "reasonix acp"
turn_timeout = 0
restart_backoff_max = 30

[runtime]
edit_interval = 1.2
send_interval = 1.0
poll_timeout = 30
busy_mode = "steer"
pairing_ttl = 900
approval_timeout = 300
approval_posture = "ask"
discovery_depth = 1
log_level = "INFO"
dry_run = false
```

Discovery uses two generic sources and **no hand-written list**:

1. directories under `PROJECTS_ROOT` that contain `.git` (a directory or a `.git`
   *file*, so worktrees count), up to `GATEWAY_DISCOVERY_DEPTH` levels;
2. the agent's own on-disk project index (`~/.reasonix/projects/<name>`, where the
   directory name is an encoded path whose dashes decode back to `/`). The encoding
   is lossy for directory names that contain real dashes, so index entries are
   *hints*: one that does not exist on disk is ignored, and the source directory may
   be missing entirely.

## Approvals

When the agent asks for permission, the gateway sends one message:

```text
▶️ Approval needed — Run the test suite
kind: execute
command: pytest -q
tool call: call-1
Tap a button to answer the agent.

[ Allow once ] [ Always ]
[ Reject ]
```

* Buttons come from the options the agent advertised, allow-options first. If the
  agent advertises no reject option, a `Deny` button answers `{"outcome":"cancelled"}`.
* Each tap is answered with the exact `optionId` the agent offered, so the agent's
  own permission rules stay in charge.
* Callbacks from users who are not allowlisted are refused (`Not authorised.`), and
  a callback from another chat than the one that was asked is refused too.
* Nothing is left hanging: an unanswered approval expires (`GATEWAY_APPROVAL_TIMEOUT`)
  and is answered as `cancelled`, and when a turn ends its still-open approvals are
  cancelled and their buttons removed.
* More than a handful of simultaneous approval prompts in one chat are declined
  rather than flooding the chat.

## Access control and containment

Both are **code gates**, not conventions:

* **Allowlist.** `ALLOWED_USER_IDS` is the allowlist and everything else is denied by
  default. An unknown sender gets a short-lived, single-use pairing code that is
  printed to the gateway log and *never* sent back to them; the operator approves it
  with `python -m acp_im_gateway pairing approve <code>`. Codes expire
  (`GATEWAY_PAIRING_TTL`) and a refused code cannot be resurrected by a stale copy of
  the state file.
* **Groups.** A group chat needs its id in `ALLOWED_CHAT_IDS` *and* an allowlisted
  sender. Empty `ALLOWED_CHAT_IDS` means direct messages only.
* **Containment.** A chat may only bind a directory inside `ALLOWED_ROOTS`
  (default: `PROJECTS_ROOT`). The path is expanded, resolved with
  `os.path.realpath` (so `..` and symlinks collapse to their real location) and
  compared component-wise against the roots. `/bind ../secrets`, a symlink pointing
  outside the root and a sibling like `/srv/app-2` when the root is `/srv/app` are all
  refused. Refusals are logged.

## Security warning

**A chat bot wired to a coding agent is remote command execution on your host.**
Read this before you run it:

* Anyone who can send messages as you can drive an agent that reads and writes files
  inside the bound project root, and runs commands under the approval posture the
  agent was configured with. Treat the chat account as a shell on your machine.
* The bot token is a password. Keep it in `.env` / the systemd `EnvironmentFile`
  (`chmod 600`), never in git, never in a screenshot, never in a chat.
* Keep `ALLOWED_USER_IDS` tight, and prefer `tool_approval = "ask"` (the default) so
  gated tool calls still need a tap. `yolo`-style postures exist in the agent; the
  gateway will happily relay whatever the agent is configured to do.
* Prefer a separate, unprivileged user for the gateway, and scope `ALLOWED_ROOTS` to
  the projects you actually want reachable.
* The gateway itself opens no listening port and never rewrites your prompt, but it
  *does* store your bindings and (transiently) pairing codes on disk — see
  `GATEWAY_STATE_FILE`.
* Group chats are dangerous by nature: every allowed participant can drive the agent,
  and the agent's output is visible to the whole group.

## Honest limits

What this v1 deliberately does **not** do, so you are not surprised:

* **No attachments in or out.** ACP's `promptCapabilities` on the reference agent
  report `image: false, audio: false`; the gateway sends text prompts only and ignores
  non-text Telegram messages with a short explanation.
* **Approvals are one-shot and chat-wide.** One tap answers one request; there is no
  per-user approval routing, no "remember this forever" beyond whatever the agent
  itself offers as an `allow_always` option, and no way to answer an approval from a
  different chat.
* **Streaming is coarse.** Edits are coalesced to at most one per
  `GATEWAY_EDIT_INTERVAL` (1.2 s), so a fast turn looks like a burst of updates, and
  a long silent stretch shows no movement until the next edit. There is no typing
  indicator per token either.
* **Messages longer than 4096 characters are chunked** into ordered parts, and code
  fences left open by a split are closed and reopened in the next part. Telegram is
  still the rendering authority: the gateway sends plain text (no `parse_mode`),
  because a broken Markdown message that fails to send is worse than plain text.
* **One in-flight turn per chat.** A second message steers or queues (configurable);
  it never runs in parallel. `/stop` drops the queue.
* **Telegram rate limits are respected, not transcended**: at most one new message per
  second per chat (`GATEWAY_SEND_INTERVAL`), and a `429` answer is honoured with its
  `retry_after` instead of crashing. Very chatty turns therefore arrive a little late.
* **Resuming is best-effort.** The gateway re-attaches a bound session with
  `session/load` and otherwise resumes the most recent session for that `cwd`; if the
  agent does not support `loadSession`, a new session is created. A mid-turn agent
  crash loses that turn (you are told, and the agent is respawned).
* **No history backfill.** Previous ACP sessions are visible to the agent, not in the
  chat; the gateway does not replay old turns.
* **No conversation branching, no message editing, no reactions.** Edited Telegram
  messages are ignored.
* **Groups are basic**: no forum topics (Telegram requires ~100+ members to enable
  them), no per-thread routing, no multi-chat fan-out for one project.
* **State file is the source of truth.** Delete it (gateway stopped) to start over.
  A running gateway owns its in-memory bindings; a CLI `bindings remove` for a chat it
  is actively using will be overridden on its next save.
* **One bot, one host, one state file.** Multi-tenant hosting of several bots is out
  of scope.

## Running it as a service

`systemd/acp-im-gateway.service` is a template with no secrets in it; the token comes
from an `EnvironmentFile` you own. The recommended install is a **user** unit, so the
gateway runs as you and can reach your projects and the agent's credentials:

```bash
mkdir -p ~/.config/systemd/user ~/.config/acp-im-gateway
cp systemd/acp-im-gateway.service ~/.config/systemd/user/
printf 'TELEGRAM_BOT_TOKEN=...\nALLOWED_USER_IDS=...\nPROJECTS_ROOT=/home/you/Projects\n' \
  > ~/.config/acp-im-gateway/gateway.env
chmod 600 ~/.config/acp-im-gateway/gateway.env
systemctl --user enable --now acp-im-gateway
journalctl --user -u acp-im-gateway -f      # pairing codes appear here
```

```ini
[Unit]
Description=acp-im-gateway — drive an ACP coding agent from Telegram
Documentation=https://github.com/XofoSol/acp-im-gateway#readme
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# Secrets never live in the unit file: point this at a 0600 file that exports
# TELEGRAM_BOT_TOKEN, ALLOWED_USER_IDS, PROJECTS_ROOT, ALLOWED_ROOTS, ...
EnvironmentFile=%h/.config/acp-im-gateway/gateway.env
# Console script installed by `pipx install .` / `pip install --user .`.
# With a virtualenv use: ExecStart=/path/to/.venv/bin/python -m acp_im_gateway run
ExecStart=%h/.local/bin/acp-im-gateway run
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

For a system-wide install, uncomment and set `User=` / `WorkingDirectory=` to your own
account (a service running as `root` would hand the agent root over the chat).

## Tests and verification

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

The suite (178 tests) uses only the standard library plus pytest: an in-memory
Telegram double, a fake clock, and a scriptable fake ACP agent on stdio. It covers

* the containment gate: inside the root, outside it, shared-prefix siblings, `..`
  traversal, symlink escapes, a symlinked root, files, missing paths, no roots;
* chunking at 4096 (including balanced code fences across parts and reassembly),
  edit coalescing and the per-chat rate limit, and `429`/`retry_after` handling;
* binding persistence round-trips, corrupt state files, state-file permissions,
  discovery from both sources, and per-chat queueing;
* the allowlist, group gating, pairing approval/rejection and **pairing expiry**;
* the ACP client against a fake agent: verbatim prompts, streamed updates, unknown
  notifications, non-JSON stdout, declined inbound requests, deferred permission
  requests, timeouts, crashes + respawn, steering only when advertised;
* gateway behaviour end to end: denial + pairing codes, `/bind` refusals, one edited
  message per turn, mid-turn steering vs queueing, `/stop`, `/status`, and approvals
  reaching the agent;
* the CLI: `--help` listing subcommands, config errors, pairing, bindings, dry-run.

One test is marked `integration`. It spawns the **real** agent (`reasonix acp` when it
is on `PATH`) and runs `initialize` -> `session/new {cwd=tmpdir}` -> `session/close`.
It **never** sends `session/prompt`, because that spends your money — the test asserts
that no prompt was ever written to the agent. It skips cleanly when the binary is
missing:

```bash
.venv/bin/python -m pytest -q -m integration
```

## Project layout

```text
acp_im_gateway/
  __main__.py     CLI entry: run | pairing list|approve|reject | bindings | projects | config
  config.py       env + optional TOML config loading, precedence, validation
  containment.py  the containment gate (realpath + component-wise root check)
  access.py       allowlist, group gate, pairing codes with expiry
  acp.py          ACP client: stdio JSON-RPC, streaming, respawn, inbound requests
  telegram.py     Bot API client (urllib), chunking, throttling, in-place message stream
  router.py       bindings, project discovery, state persistence, per-chat serialisation
  approvals.py    permission request -> inline buttons -> ACP response
  gateway.py      the polling/turn loop that ties it all together
tests/            unit tests, a fake ACP agent, and one integration test
systemd/          service template (no secrets)
.env.example      every variable, placeholder values
```

## Roadmap

* Slack adapter (same router, different transport).
* Telegram forum topics once a group has enough members to enable them.
* Opt-in image/audio attachments once agents advertise those prompt capabilities.
* A small web UI for bindings and approvals.

Out of scope for v1 by design: web UI, multi-tenant hosting of several bots, and any
kind of prompt rewriting.

## License

MIT — see [LICENSE](LICENSE).
