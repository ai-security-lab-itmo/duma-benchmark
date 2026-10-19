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


@contextmanager
def use_generation_backends(
    backends: Mapping[str, GenerationBackend],
) -> Iterator[None]:
    token = _backends.set({**_backends.get(), **backends})
    try:
        yield
    finally:
        _backends.reset(token)
