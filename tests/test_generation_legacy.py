from duma.utils.generation import generate
from duma.utils import llm_utils
from duma.data_model.message import AssistantMessage


def test_unbound_legacy_model_still_uses_the_existing_transport(monkeypatch):
    captured = {}
    expected = AssistantMessage(role="assistant", content="legacy response")

    def legacy(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return expected

    monkeypatch.setattr(llm_utils, "generate", legacy)
    assert generate("legacy-model", [], temperature=0.25) is expected
    assert captured["args"] == ("legacy-model", [])
    assert captured["kwargs"]["temperature"] == 0.25
