"""Extensible Step Registry allowing dynamic registration and instantiation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mlkit.core.errors import RegistryError
from mlkit.core.step import BaseStep

_REGISTRY: dict[str, type[BaseStep]] = {}


def register(name: str) -> Callable[[type[Any]], type[Any]]:
    """
    Decorator to register a step into the global step registry.

    Example:
        @register("preprocessing.imputer")
        class CustomImputer(BaseStep):
            ...
    """

    def decorator(cls: type[Any]) -> type[Any]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_step(name: str, **kwargs: Any) -> BaseStep:
    """Instantiate a registered step by its key."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise RegistryError(f"Step '{name}' not found. Available steps: [{available}]")
    cls = _REGISTRY[name]
    return cls(**kwargs)


def list_steps() -> list[str]:
    """List all registered step names."""
    return sorted(_REGISTRY.keys())
