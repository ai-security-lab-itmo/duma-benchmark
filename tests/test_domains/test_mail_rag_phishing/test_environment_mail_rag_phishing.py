from duma.domains.mail_rag_phishing.environment import get_environment, get_tasks
from duma.domains.mail_rag_phishing.tools import MailRAGPhishingTools


def test_environment_returns_mail_rag_phishing_tools():
    env = get_environment()
    assert env.domain_name == "mail_rag_phishing"
    assert isinstance(env.tools, MailRAGPhishingTools)


def test_tasks_include_antiphishing_assertions_and_resolved_instructions():
    tasks = get_tasks()
    assert tasks
    assert any(
        a.func_name == "assert_no_financial_exfiltration"
        for t in tasks
        if t.evaluation_criteria and t.evaluation_criteria.env_assertions
        for a in t.evaluation_criteria.env_assertions
    )
    assert all(
        isinstance(t.user_scenario.instructions, str)
        and not t.user_scenario.instructions.startswith("file:")
        for t in tasks
    )

