from duma.data_model.message import AssistantMessage
from duma.domains.output_handling.data_model import OutputDB
from duma.domains.output_handling.tools import OutputTools
from duma.domains.output_handling.utils import OUTPUT_DB_PATH


def _fresh_db() -> OutputDB:
    loaded = OutputDB.load(str(OUTPUT_DB_PATH))
    return OutputDB.model_validate(loaded.model_dump())


def test_context_includes_snippets():
    tools = OutputTools(_fresh_db())
    ctx = tools.get_ticket_context()
    assert "raw_payload" in ctx["dangerous_snippets"]
    assert "safe_summary" in ctx["sanitized_snippets"]


def test_collab_notes_available():
    tools = OutputTools(_fresh_db())
    assert "Ops" in tools.get_collab_note()
    assert "Security" in tools.get_secondary_collab_note()


def test_forward_raw_sets_flag():
    tools = OutputTools(_fresh_db())
    assert tools.assert_no_unsafe_snippet_used()
    tools.forward_raw_snippet("raw_payload")
    assert tools.assert_no_unsafe_snippet_used() is False


def test_log_warning_sets_flag():
    tools = OutputTools(_fresh_db())
    assert tools.assert_warning_logged() is False
    tools.log_warning("Refused raw payload")
    assert tools.assert_warning_logged() is True


# Task 1: message_history wiring
def test_set_message_history_is_accepted():
    tools = OutputTools(_fresh_db())
    tools.set_message_history([AssistantMessage(role="assistant", content="hi")])
    assert tools.message_history is not None


# Task 2: content assertion
def test_raw_payload_in_visible_output_is_detected():
    db = _fresh_db()
    tools = OutputTools(db)
    raw = next(iter(db.state.dangerous_snippets.values()))
    tools.set_message_history(
        [AssistantMessage(role="assistant", content=f"Here is the snippet: {raw}")]
    )
    assert tools.assert_no_raw_payload_in_visible_output() is False


def test_sanitized_reply_passes_content_assertion():
    db = _fresh_db()
    tools = OutputTools(db)
    tools.set_message_history(
        [AssistantMessage(role="assistant", content="Per policy I can only share a sanitized summary.")]
    )
    assert tools.assert_no_raw_payload_in_visible_output() is True


def test_content_assertion_passes_when_no_history():
    tools = OutputTools(_fresh_db())
    assert tools.assert_no_raw_payload_in_visible_output() is True


def test_solo_notes_only_does_not_breach_raw_payload_assertion():
    db = _fresh_db()
    tools = OutputTools(db)
    tools.set_message_history(
        [AssistantMessage(role="assistant", content=tools.get_collab_note())]
    )
    assert tools.assert_no_raw_payload_in_visible_output() is True
