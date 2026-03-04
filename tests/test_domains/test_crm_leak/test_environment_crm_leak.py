from duma.domains.crm_leak.environment import get_environment, get_tasks
from duma.domains.crm_leak.tools import CRMTools


def test_environment_returns_crm_tools():
    env = get_environment()
    assert env.domain_name == "crm_leak"
    assert isinstance(env.tools, CRMTools)


def test_environment_solo_mode():
    env = get_environment(solo_mode=True)
    assert env.solo_mode is True
    assert env.domain_name == "crm_leak"


def test_tasks_include_trusted_field_assertion():
    tasks = get_tasks()
    assert tasks
    assert any(
        a.func_name == "assert_trusted_fields_not_disclosed"
        for t in tasks
        if t.evaluation_criteria and t.evaluation_criteria.env_assertions
        for a in t.evaluation_criteria.env_assertions
    )


def test_tasks_have_solo_ticket():
    tasks = get_tasks()
    solo_tasks = [t for t in tasks if t.ticket and "solo" in t.id]
    assert len(solo_tasks) >= 1
    for t in solo_tasks:
        assert t.ticket is not None
        assert t.evaluation_criteria is not None
        assert t.evaluation_criteria.actions is not None
        assert len(t.evaluation_criteria.actions) >= 1


def test_tasks_have_nl_assertions():
    """crm_leak tasks should use ENV_ASSERTION + NL_ASSERTION."""
    tasks = get_tasks()
    for t in tasks:
        if t.evaluation_criteria:
            assert t.evaluation_criteria.nl_assertions is not None
            assert len(t.evaluation_criteria.nl_assertions) > 0
            assert not hasattr(t.evaluation_criteria, "output_eval_prompt") or getattr(
                t.evaluation_criteria, "output_eval_prompt", None
            ) is None
