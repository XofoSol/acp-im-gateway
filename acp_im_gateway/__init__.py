"""acp-im-gateway — drive an ACP-speaking coding agent from a chat app.

Telegram first; Slack is roadmap. The gateway is pure plumbing: user text goes
verbatim into the agent over ACP, and permission requests come back as chat
buttons. Runtime is the Python standard library only.
"""

from __future__ import annotations

__version__ = "1.1.0"
__all__ = ["__version__"]
