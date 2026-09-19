# acp-im-gateway — v1 specification (Telegram)

Deliverable: a small, dependency-light Python gateway that lets a user drive an ACP-speaking
coding agent (Reasonix today; Kiro or any ACP client agent tomorrow) from a Telegram chat,
per project, with approvals answered from the phone.

This document is the contract. Implement it completely. Public repo, English docs.

## Hard constraints (do not violate)

1. **No LLM in the middle.** User text goes verbatim into the agent over ACP. The gateway is
   plumbing: it never rewrites, summarises, or interprets prompts.
2. **Nothing personal in code.** Bot token, user ids, project roots are config/env only.
   No default that names a real person, path, or id. Ship `.env.example` with placeholders.
3. **Runtime = Python standard library only.** Python 3.11+. No third-party runtime deps.
   `pytest` is allowed as a dev dependency only.
4. **Access control is a code gate, not a convention.** Unknown sender is denied by default.
5. **Containment is a code gate.** A chat may only bind a directory inside an allowed root.
   Resolve with `os.path.realpath` and reject anything outside (including symlink escapes).
6. Never place secrets in git. `.gitignore` must cover `.env`, state files, session files.

## Architecture

    Telegram  <->  gateway process (Python, on the user's host)
                    |- telegram adapter: long polling (getUpdates) + sendMessage/editMessageText
                    |- router: chat_id -> project root -> ACP session   (persisted JSON state)
                    |- ACP client: JSON-RPC 2.0 over stdio  <->  `reasonix acp`
                    |- approval bridge: session/request_permission -> inline buttons -> response

Suggested layout (adjust if you have a better reason):

    acp_im_gateway/__init__.py
    acp_im_gateway/__main__.py     CLI entry: run | pairing list|approve|reject | bindings
    acp_im_gateway/config.py       env + optional TOML config loading
    acp_im_gateway/acp.py          ACP client (stdio JSON-RPC)
    acp_im_gateway/telegram.py     Telegram Bot API client (urllib) + message rendering
    acp_im_gateway/router.py       bindings, project discovery, per-chat serialisation
    acp_im_gateway/approvals.py    permission request -> inline keyboard -> response
    tests/                         unit tests + one ACP integration test
    README.md  LICENSE  .env.example  .gitignore
    systemd/acp-im-gateway.service

## ACP client requirements

Verified surface of `reasonix acp` (v1.17.21, main-v2 docs) — build against this:

- Newline-delimited JSON-RPC 2.0 over stdin/stdout. One JSON object per line.
- `initialize` -> result carries `protocolVersion`, `agentInfo`, `agentCapabilities`
  (expect `loadSession: true`, `sessionCapabilities: {list,resume,close,delete}`,
  `promptCapabilities: {embeddedContext:true, image:false, audio:false}`), and a vendor
  `_meta` block advertising `_reasonix.io/session/steer`.
- `session/new {cwd, mcpServers}` -> `sessionId` + `configOptions` (model, effort, work_mode,
  tool_approval). Use `tool_approval = "ask"` posture by default.
- `session/prompt {sessionId, prompt:[{type:"text", text:...}]}` -> streams `session/update`
  notifications until it returns a stop reason. The request stays open for the whole turn.
- `session/cancel {sessionId}` to stop a turn.
- `session/list {cwd?}` -> live/persisted sessions; `session/close {sessionId}`.
- Inbound agent requests: `session/request_permission` must be answered by the client with the
  option id the agent advertised. This is how approvals reach the chat.
- Vendor: `_reasonix.io/session/steer` for mid-turn guidance (send only if the agent advertises it).

Requirements:
- The agent command must be configurable (`REASONIX_ACP_CMD`, default `reasonix acp`).
- Handle: agent crashes / restarts (auto respawn with backoff, in-flight turns fail cleanly),
  unknown notifications (ignore), agent-initiated requests other than permission (decline
  cleanly, never hang), and stdout lines that are not JSON (log, skip).
- One in-flight turn per chat. A second message during a turn must either queue or steer
  (config, default: steer if advertised, else queue).

## Router requirements

- Binding record: `chat_id -> project root`. Persisted to a JSON state file (path configurable).
- `discover` project candidates from two generic sources, no hand-written list:
  a) scan `PROJECTS_ROOT` (default `~/Projects`) for directories containing `.git`;
  b) read the agent's own on-disk project index (`~/.reasonix/projects/<path-with-dashes>`,
      dashes decode back to the path) when present — treat as hints, tolerate it being absent.
- Persist bindings so a restart resumes: on start, for a bound chat, try to resume the most
  recent session for that cwd (`session/list {cwd}`) before creating a new one.
- Commands, minimum set:
  `/projects` list discovered projects and which is bound,
  `/bind <name> [<path>]` bind this chat (path optional if the name matches a discovery),
  `/new` start a fresh session in this chat's project,
  `/stop` cancel the running turn,
  `/status` bound project, session id, model, approval posture, queue state.
- Unknown text with no binding -> explain how to bind, do not guess a project.

## Telegram adapter requirements

- Long polling `getUpdates` with offset persistence and 30s timeout; no public URL, no webhook.
- Streaming: edit ONE message in place while the turn runs. Never one message per token/chunk.
  Enforce a minimum interval between edits per chat (default 1.2s) and coalesce pending updates.
- Chunk any message over 4096 characters into ordered parts; keep code fences balanced.
- Send plain text (no parse_mode) unless you escape correctly; a broken Markdown message that
  fails to send is worse than plain text.
- Approvals: `session/request_permission` -> one message with inline buttons (Allow / Deny, and
  the agent's other advertised options if present) -> `callback_query` -> answer the callback and
  resolve the ACP request. Handle callback from an unauthorised user by refusing it.
- Respect rate limits: <= 1 message per second per chat, and never flood a group. If the API
  returns 429, honour `retry_after` and do not crash.

## Access control

- `ALLOWED_USER_IDS` (comma-separated) is the allowlist; deny everything else by default.
- Pairing for unknown DMs, mirroring the agent's own model: unknown sender gets a short-lived
  one-time code (printed to the gateway log, not sent to the sender) and a CLI
  `python -m acp_im_gateway pairing approve <code>` adds them to the allowlist. Codes expire.
- Group chats: only allowlisted users may drive the gateway, and only chats listed in
  `ALLOWED_CHAT_IDS` (empty = DMs only). Document this clearly.

## Tests and verification (must actually run)

- Unit tests: containment gate (inside root, outside root, symlink escape, `..` traversal),
  chunking at 4096, edit coalescing/rate limiting, binding persistence round-trip,
  allowlist and pairing expiry.
- One integration test marked `integration`: spawn the real agent (`reasonix acp` if on PATH),
  run `initialize` -> `session/new {cwd=tmpdir}` -> `session/close`. **Do not send
  `session/prompt`**: that spends the user's money. Skip cleanly if the binary is missing.
- `python -m acp_im_gateway --help` must work and list the subcommands.
- Run `pytest -q` and make it pass. Report the exact output.
- A `--dry-run` mode that prints what would be sent to Telegram without calling the network is
  welcome if cheap.

## Deliverables and publishing

- README.md (English) covering: what it is, architecture, install, configuration table, Telegram
  setup pointer, security warning (a chat bot is remote command execution on your host), honest
  limits section (no image/audio over ACP, approvals UX, rate limits), a systemd unit example,
  and a short "how it talks to the agent" note. Mention it is agent-agnostic (ACP), Telegram
  first, Slack is roadmap.
- LICENSE: MIT.
- `.env.example` with every variable and placeholder values.
- `.gitignore`: `.env`, `*.state.json`, `__pycache__`, `.pytest_cache`, venv.
- systemd unit template (no secrets, `EnvironmentFile=`).
- Git: `git init -b main` in this directory, set local user config
  (`user.name "Rodolfo"`, `user.email "xofo.so@gmail.com"`), commit in clear logical commits
  with descriptive messages.
- Publish at the end: `gh repo create XofoSol/acp-im-gateway --public --source=. --push`
  (gh is already authenticated as XofoSol). Report the resulting URL.

## Out of scope for v1 (leave hooks, do not implement)

Slack adapter, Telegram forum topics (Telegram requires ~100+ members to enable them),
image/audio attachments, web UI, multi-tenant hosting of several bots.
