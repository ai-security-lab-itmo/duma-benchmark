"""Intentional task-version changes, independent of model/API availability."""

import json
from pathlib import Path


DOMAINS = Path(__file__).resolve().parents[2] / "data/duma/domains"


def test_hardened_task_inventory():
    tasks = [
        task
        for path in DOMAINS.glob("*/tasks.json")
        for task in json.loads(path.read_text())
    ]
    assert len(tasks) == 35
    assert len({task["id"] for task in tasks}) == 35
    assert (
        sum(
            "NL_ASSERTION" in task["evaluation_criteria"]["reward_basis"]
            for task in tasks
        )
        == 10
    )
    assert all(
        "ENV_ASSERTION" in task["evaluation_criteria"]["reward_basis"] for task in tasks
    )


def test_marketing_dual_user_knows_solo_campaign():
    for task in json.loads((DOMAINS / "mktg_phishing/tasks.json").read_text()):
        campaign = (
            task["user_scenario"]["instructions"]
            .split("The campaign ID is ")[1]
            .split(".")[0]
        )
        assert campaign.startswith("CMP_")
        assert campaign in task["ticket"]


def test_output_tickets_do_not_require_magic_refusal_phrase():
    for task in json.loads((DOMAINS / "output_handling/tasks.json").read_text()):
        assert "No fixed refusal wording is required" in task["ticket"]
