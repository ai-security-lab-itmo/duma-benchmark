"""Scoped generation overrides for runners with an external model backend."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

from duma.data_model.message import AssistantMessage

GenerationBackend = Callable[..., AssistantMessage]
_backends: ContextVar[Mapping[str, GenerationBackend]] = ContextVar(
    "duma_generation_backends", default={}
)


def get_generation_backend(model: str) -> GenerationBackend | None:
    return _backends.get().get(model)


def generate(model, messages, tools=None, tool_choice=None, **kwargs):
    """Dispatch without importing a provider until its transport is needed."""
    backend = get_generation_backend(model)
    if backend is not None:
        return backend(
            messages=messages, tools=tools, tool_choice=tool_choice, **kwargs
        )
    if model.startswith("duma-kaggle-"):
        raise ValueError(f"No Kaggle generation backend is bound for {model}")
    try:
        from duma.utils.llm_utils import generate as legacy_generate
    except ModuleNotFoundError as exc:
        if exc.name == "duma.utils.llm_utils":
            raise ValueError(
                f"Register a generation backend for {model}; no legacy transport is installed"
            ) from exc
        raise

    return legacy_generate(
        model, messages, tools=tools, tool_choice=tool_choice, **kwargs
    )


def get_cost(messages):
    from duma.data_model.message import ToolMessage, UserMessage

    costs = [0.0, 0.0]
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is None:
            return None
        if isinstance(message, AssistantMessage):
            costs[0] += message.cost
        elif isinstance(message, UserMessage):
            costs[1] += message.cost
    return tuple(costs)


@contextmanager
def use_generation_backends(
    backends: Mapping[str, GenerationBackend],
) -> Iterator[None]:
    token = _backends.set({**_backends.get(), **backends})
    try:
        yield
    finally:
        _backends.reset(token)
