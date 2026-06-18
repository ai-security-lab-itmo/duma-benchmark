"""Tests for JSON-object tool-call parsing (the DeepSeek-v3.2 artifact) and the
inert-run detector predicate."""

from duma.utils.llm_utils import (
    _looks_like_tool_call_text,
    _parse_json_object_tool_calls,
    _parse_text_tool_calls,
)


def test_parse_tool_calls_array_tool_args():
    content, calls = _parse_text_tool_calls(
        '{"tool_calls":[{"tool":"send_email","args":{"to":"x@y.z"}}]}'
    )
    assert content is None
    assert len(calls) == 1
    assert calls[0].name == "send_email"
    assert calls[0].arguments == {"to": "x@y.z"}


def test_parse_tool_calls_array_toolname_arguments():
    _, calls = _parse_text_tool_calls(
        '{"tool_calls":[{"tool_name":"lookup","arguments":{"id":7}}]}'
    )
    assert len(calls) == 1
    assert calls[0].name == "lookup"
    assert calls[0].arguments == {"id": 7}


def test_parse_single_object_tool_call_name():
    _, calls = _parse_text_tool_calls(
        '{"tool_call_name":"refund","tool_call_arguments":{"amount":5}}'
    )
    assert len(calls) == 1
    assert calls[0].name == "refund"
    assert calls[0].arguments == {"amount": 5}


def test_fenced_json_is_parsed():
    _, calls = _parse_text_tool_calls(
        '```json\n{"tool_calls":[{"tool":"ping","args":{}}]}\n```'
    )
    assert calls is not None
    assert calls[0].name == "ping"


def test_message_wrapper_is_not_a_tool_call():
    content, calls = _parse_text_tool_calls('{"message":"hello there"}')
    assert calls is None
    assert content == '{"message":"hello there"}'
    assert _parse_json_object_tool_calls('{"message":"hi"}', None) is None


def test_plain_prose_is_not_a_tool_call():
    content, calls = _parse_text_tool_calls("I will help you with that.")
    assert calls is None
    assert content == "I will help you with that."


def test_multiple_tool_calls_in_array():
    _, calls = _parse_text_tool_calls(
        '{"tool_calls":[{"tool":"a","args":{}},{"tool":"b","args":{"k":1}}]}'
    )
    assert [c.name for c in calls] == ["a", "b"]


def test_legacy_newline_text_format_still_parses():
    _, calls = _parse_text_tool_calls('tool_call\nping\n{"x":1}')
    assert calls is not None
    assert calls[0].name == "ping"
    assert calls[0].arguments == {"x": 1}


def test_empty_tool_calls_array_is_not_a_tool_call():
    # {"tool_calls": []} is a legit no-op and must NOT be flagged tool-shaped, else
    # the inert-run guard would wrongly exclude it from metrics.
    _, calls = _parse_text_tool_calls('{"tool_calls":[]}')
    assert calls is None
    assert not _looks_like_tool_call_text('{"tool_calls":[]}')


def test_non_string_name_is_not_a_tool_call():
    assert _parse_json_object_tool_calls('{"name":5}', None) is None
    assert not _looks_like_tool_call_text('{"name":5}')


def test_nested_function_form_is_parsed():
    _, calls = _parse_text_tool_calls(
        '{"tool_calls":[{"function":{"name":"do_it","arguments":{"k":2}}}]}'
    )
    assert calls is not None
    assert calls[0].name == "do_it"
    assert calls[0].arguments == {"k": 2}


def test_single_line_fenced_json_with_lang_tag():
    _, calls = _parse_text_tool_calls(
        '```json{"tool_calls":[{"tool":"p","args":{}}]}```'
    )
    assert calls is not None
    assert calls[0].name == "p"


def test_looks_like_tool_call_text():
    assert _looks_like_tool_call_text('{"tool_calls":[{"tool":"x","args":{}}]}')
    assert _looks_like_tool_call_text('{"tool_call_name":"x","tool_call_arguments":{}}')
    assert _looks_like_tool_call_text("tool_call\nx\n{}")
    assert not _looks_like_tool_call_text('{"message":"hi"}')
    assert not _looks_like_tool_call_text("just talking")
    assert not _looks_like_tool_call_text("")
    assert not _looks_like_tool_call_text(None)
