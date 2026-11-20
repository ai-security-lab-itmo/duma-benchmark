"""Single-episode execution without experiment, display or provider dependencies."""

from collections.abc import Callable

from duma.agent.llm_agent import LLMAgent, LLMGTAgent, LLMSoloAgent
from duma.evaluator.evaluator import EvaluationType, evaluate_simulation
from duma.orchestrator.orchestrator import Orchestrator
from duma.registry import registry
from duma.user.user_simulator import DummyUser
from duma.data_model.tasks import Task
from duma.data_model.simulation import SimulationRun


def get_tasks(
    task_set_name: str, task_ids: list[str] | None = None, num_tasks: int | None = None
) -> list[Task]:
    tasks = registry.get_tasks_loader(task_set_name)()
    if task_ids is not None:
        tasks = [task for task in tasks if task.id in task_ids]
        missing = set(task_ids) - {task.id for task in tasks}
        if missing or len(tasks) != len(task_ids):
            raise ValueError(f"Not all tasks were found for {task_set_name}: {missing}")
    return tasks if num_tasks is None else tasks[:num_tasks]


def run_task(
    domain: str,
    task: Task,
    agent: str,
    user: str,
    llm_agent: str | None = None,
    llm_args_agent: dict | None = None,
    llm_user: str | None = None,
    llm_args_user: dict | None = None,
    max_steps: int = 100,
    max_errors: int = 10,
    evaluation_type: EvaluationType = EvaluationType.ALL_WITH_NL_ASSERTIONS,
    seed: int | None = None,
    judge_model: str | None = None,
    judge_args: dict | None = None,
    on_simulation_complete: Callable[[SimulationRun], None] | None = None,
) -> SimulationRun:
    if max_steps <= 0 or max_errors <= 0:
        raise ValueError("Max steps and max errors must be greater than zero")
    constructor = registry.get_env_constructor(domain)
    environment = constructor()
    agent_class = registry.get_agent_constructor(agent)
    solo = False
    options = dict(
        domain_policy=environment.get_policy(), llm=llm_agent, llm_args=llm_args_agent
    )
    if issubclass(agent_class, LLMAgent):
        agent_instance = agent_class(tools=environment.get_tools(), **options)
    elif issubclass(agent_class, LLMGTAgent):
        agent_instance = agent_class(
            tools=environment.get_tools(), task=task, **options
        )
    elif issubclass(agent_class, LLMSoloAgent):
        solo = True
        environment = constructor(solo_mode=True)
        options["domain_policy"] = environment.get_policy()
        user_tools = environment.get_user_tools() if environment.user_tools else []
        agent_instance = agent_class(
            tools=environment.get_tools() + user_tools, task=task, **options
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_class}")
    try:
        user_tools = environment.get_user_tools()
    except Exception:
        user_tools = None
    user_class = registry.get_user_constructor(user)
    if issubclass(user_class, DummyUser):
        assert isinstance(agent_instance, LLMSoloAgent), (
            "Dummy user requires solo agent"
        )
    user_instance = user_class(
        tools=user_tools,
        instructions=str(task.user_scenario),
        llm=llm_user,
        llm_args=llm_args_user,
    )
    simulation = Orchestrator(
        domain=domain,
        agent=agent_instance,
        user=user_instance,
        environment=environment,
        task=task,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
        solo_mode=solo,
    ).run()
    if on_simulation_complete is not None:
        on_simulation_complete(simulation)
    judge_options = {}
    if judge_model is not None:
        judge_options["judge_model"] = judge_model
    if judge_args is not None:
        judge_options["judge_args"] = judge_args
    simulation.reward_info = evaluate_simulation(
        domain=domain,
        task=task,
        simulation=simulation,
        evaluation_type=evaluation_type,
        solo_mode=solo,
        **judge_options,
    )
    return simulation
