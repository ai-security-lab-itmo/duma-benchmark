"""Shared message-signature helpers.

These power both the orchestrator loop-guard and the crash classifier so the two
cannot drift apart. ``message_signature`` mirrors the historical
``Orchestrator._message_signature``; ``tool_signature`` is a prose-insensitive variant
used to catch loops where an agent repeats the same tool call while varying its
surrounding text.

The signatures are used as *equality keys* on fatal paths (loop guard -> reward=0,
user give-up -> early stop), so they hash the FULL content/arguments rather than a
prefix: two legitimately-different long payloads that share a long prefix must not
collapse to the same signature and falsely trip a terminating decision.
"""

import hashlib
import json
from typing import Optional

from duma.data_model.message import AssistantMessage, UserMessage

ParticipantMessage = AssistantMessage | UserMessage


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_text(text: Optional[str], max_len: int = 240) -> str:
    """Collapse whitespace + lowercase. Truncated form, suitable for log previews."""
    if text is None:
        return ""
    normalized = " ".join(text.split()).strip().lower()
    if len(normalized) > max_len:
        return normalized[:max_len]
    return normalized


def content_key(text: Optional[str]) -> str:
    """Stable equality key for a message's text: hash of the FULL normalized content."""
    normalized = " ".join((text or "").split()).strip().lower()
    return _hash(normalized)


def _tool_sig_parts(message: ParticipantMessage) -> list[str]:
    parts = []
    for tool_call in message.tool_calls or []:
        try:
            args = json.dumps(tool_call.arguments, sort_keys=True, ensure_ascii=True)
        except Exception:
            args = str(tool_call.arguments)
        # Hash the full args so distinguishing fields past a prefix are not lost.
        parts.append(f"{tool_call.name}:{_hash(args)}")
    return parts


def message_signature(message: ParticipantMessage) -> str:
    """Role + full-content hash + tool-call name/args hash. Content-sensitive."""
    tool_sig = "|".join(_tool_sig_parts(message))
    return f"{message.role}|{content_key(message.content)}|{tool_sig}"


def tool_signature(message: ParticipantMessage) -> Optional[str]:
    """Role + tool-call name/args only, ignoring prose.

    Returns ``None`` for messages without tool calls so callers can restrict a
    tool-loop check to turns that actually act on the environment.
    """
    parts = _tool_sig_parts(message)
    if not parts:
        return None
    return f"{message.role}|{'|'.join(parts)}"
