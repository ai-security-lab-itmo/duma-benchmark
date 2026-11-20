import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from duma.utils import utils


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize(
    "solo,trials,temperatures",
    [(True, 4, [0.0]), (False, 4, [0.0]), (False, 5, [0.0, 0.5, 1.0])],
)
def test_paper_runner_sets_fixed_models(monkeypatch, tmp_path, solo, trials, temperatures):
    runner = runpy.run_path(str(SCRIPTS / "run_experiments_and_generate_tables.py"))
    run = runner["run_all_experiments"]
    commands = []

    def capture(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, communicate=lambda: ("", ""))

    monkeypatch.setattr(utils, "DATA_DIR", tmp_path)
    monkeypatch.setitem(run.__globals__, "__file__", str(tmp_path / "scripts" / "runner.py"))
    monkeypatch.setitem(run.__globals__, "subprocess", SimpleNamespace(Popen=capture, PIPE=-1))
    monkeypatch.setitem(run.__globals__, "shutil", SimpleNamespace(which=lambda _: "duma"))

    run(["openai/gpt-4.1"], temperatures, {"collab": ["example_task"]}, trials, 1, solo=solo)

    assert len(commands) == (1 if solo else len(temperatures))
    for cmd, temperature in zip(commands, temperatures):
        assert cmd[cmd.index("--num-trials") + 1] == str(trials)
        assert cmd[cmd.index("--evaluator-llm") + 1] == "openai/gpt-4o"
        if solo:
            assert cmd[cmd.index("--user") + 1] == "dummy_user"
            assert "--user-llm" not in cmd
        else:
            assert cmd[cmd.index("--user-llm") + 1] == "openai/gpt-4o-mini"
            assert json.loads(cmd[cmd.index("--user-llm-args") + 1])["temperature"] == temperature
