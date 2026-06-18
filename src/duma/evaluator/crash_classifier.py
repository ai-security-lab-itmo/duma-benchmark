"""Classify how a run ended so the metrics layer can separate measurement artifacts
from genuine policy outcomes.

Two concerns live here:

- ``classify_crash`` — for a ``TOO_MANY_ERRORS`` termination, decide whether it was an
  infrastructure crash (API/proxy failure → exclude from metrics) or a behavioural
  loop (the agent repeated itself / overflowed → a real, if degenerate, outcome).
  This is the framework-owned version of the heuristic that previously lived only in
  ``notebooks/solo_vs_dual_comparison.ipynb``.
- ``classify_run_status`` — map a finished run to a :class:`RunStatus`, additionally
  flagging *inert* runs (agent emitted tool-call-shaped text but executed zero tools,
  e.g. a tool-call serialization mismatch) as ``INVALID``.
"""

from typing import Optional

from duma.data_model.message import AssistantMessage, Message, UserMessage
from duma.data_model.simulation import RunStatus, SimulationRun, TerminationReason
from duma.utils.llm_utils import _looks_like_tool_call_text
from duma.utils.signatures import message_signature, tool_signature

# Mirrors the orchestrator loop-guard window/threshold so the classifier and the live
# guard stay consistent. The token threshold sits just under the orchestrator's
# per-message completion-token guard (5000).
LOOP_WINDOW = 10
LOOP_MAX_UNIQUE = 2
LOOP_TOKEN_GUARD = 4500


def _participant_messages(
    messages: list[Message],
) -> list[AssistantMessage | UserMessage]:
    return [m for m in messages if isinstance(m, (AssistantMessage, UserMessage))]


def classify_crash(simulation: SimulationRun) -> Optional[str]:
    """Classify a ``TOO_MANY_ERRORS`` run as 'infrastructure' or 'behavioral'.

    Returns ``None`` for any other termination reason.
    """
    if simulation.termination_reason != TerminationReason.TOO_MANY_ERRORS:
        return None

    messages = simulation.messages
    n = len(messages)

    # Died before any real exchange → the generate() call itself failed → infra.
    if n <= 2:
        return "infrastructure"

    # Loop guard signature: the last window of participant turns is near-identical.
    recent = _participant_messages(messages)[-LOOP_WINDOW:]
    sigs = [message_signature(m) for m in recent]
    if len(sigs) >= 4 and len(set(sigs)) <= LOOP_MAX_UNIQUE:
        return "behavioral"

    # Prose-insensitive tool loop: the agent repeats the same tool call while varying
    # its surrounding text (which the content-sensitive signature above misses).
    tool_sigs = [
        tool_signature(m)
        for m in messages
        if isinstance(m, AssistantMessage) and m.is_tool_call()
    ]
    tool_sigs = [s for s in tool_sigs if s is not None]
    if len(tool_sigs) >= 5 and len(set(tool_sigs[-LOOP_WINDOW:])) <= 1:
        return "behavioral"

    # Token guard: a final oversized assistant message (reasoning overflow).
    for m in reversed(messages):
        if isinstance(m, AssistantMessage):
            usage = m.usage or {}
            completion_tokens = usage.get("completion_tokens", 0) or 0
            if completion_tokens > LOOP_TOKEN_GUARD:
                return "behavioral"
            break

    return "infrastructure"


def is_inert_run(simulation: SimulationRun) -> bool:
    """True if the agent emitted tool-call-shaped text but executed zero tools.

    This catches the DeepSeek-style serialization mismatch where tool calls land in
    the content field, are never executed, and the run "passes" by inaction. A run
    that simply never tried to call a tool (legitimate pure-conversation refusal) is
    NOT flagged, because it emits no tool-call-shaped text.
    """
    executed_a_tool = any(
        isinstance(m, AssistantMessage) and m.is_tool_call()
        for m in simulation.messages
    )
    if executed_a_tool:
        return False
    return any(
        isinstance(m, AssistantMessage) and _looks_like_tool_call_text(m.content)
        for m in simulation.messages
    )


def classify_run_status(simulation: SimulationRun) -> RunStatus:
    """Map a finished run to a :class:`RunStatus` for metric exclusion decisions."""
    # Infrastructure crashes win first: a run that died on the API before doing
    # anything is errored regardless of anything else.
    termination = simulation.termination_reason
    if (
        termination == TerminationReason.TOO_MANY_ERRORS
        and classify_crash(simulation) == "infrastructure"
    ):
        return RunStatus.ERRORED
    # Inertness is checked before the remaining termination branches: an inert
    # (tool-call-text-but-zero-tools) run typically routes its "tool calls" to the
    # user and naturally cycles to MAX_STEPS, which would otherwise mislabel it
    # INCOMPLETE (kept in the denominator) instead of INVALID (excluded).
    if is_inert_run(simulation):
        return RunStatus.INVALID
    if termination in (TerminationReason.TOO_MANY_ERRORS, TerminationReason.MAX_STEPS):
        return RunStatus.INCOMPLETE
    return RunStatus.EVALUATED
