"""API freeze (v1.0): the public surface must only change on purpose.

The snapshot ``tests/api_snapshot.json`` lists every public name and its signature. A change
that removes or alters one is a breaking change (major version, see docs/ROADMAP.md); an
addition is a minor-version feature. Either way, regenerate the snapshot deliberately::

    MLKIT_UPDATE_API_SNAPSHOT=1 uv run pytest tests/test_public_api.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path

SNAPSHOT = Path(__file__).with_name("api_snapshot.json")

#: packages whose ``__all__`` is public API
PACKAGES = [
    "mlkit",
    "mlkit.core",
    "mlkit.qc_eda",
    "mlkit.preprocessing",
    "mlkit.features",
    "mlkit.selection",
    "mlkit.modelling",
    "mlkit.monitoring",
    "mlkit.viz",
]
#: modules whose public top-level functions / classes are public API
MODULES = [
    "mlkit.runner",
    "mlkit.explain",
    "mlkit.timeseries",
    "mlkit.modelling.presets",
    "mlkit.benchmarks",
    "mlkit.serving",
]


def _signature(obj: object) -> str:
    target = obj.__init__ if inspect.isclass(obj) else obj  # type: ignore[misc]
    try:
        sig = str(inspect.signature(target))
    except (TypeError, ValueError):
        return "<no signature>"
    return sig.replace("(self, ", "(").replace("(self)", "()")


def _describe(obj: object) -> str:
    if inspect.isclass(obj):
        return f"class{_signature(obj)}"
    if callable(obj):
        return f"def{_signature(obj)}"
    return f"value:{type(obj).__name__}"


def current_api() -> dict[str, dict[str, str]]:
    api: dict[str, dict[str, str]] = {}
    for name in PACKAGES:
        mod = importlib.import_module(name)
        api[name] = {n: _describe(getattr(mod, n)) for n in sorted(mod.__all__)}
    for name in MODULES:
        mod = importlib.import_module(name)
        api[name] = {
            n: _describe(obj)
            for n, obj in sorted(vars(mod).items())
            if not n.startswith("_")
            and (inspect.isfunction(obj) or inspect.isclass(obj))
            and getattr(obj, "__module__", None) == name
        }
    return api


def test_public_api_matches_snapshot():
    api = current_api()
    if os.environ.get("MLKIT_UPDATE_API_SNAPSHOT") or not SNAPSHOT.exists():
        SNAPSHOT.write_text(json.dumps(api, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    frozen = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    removed, changed, added = [], [], []
    for module, names in frozen.items():
        now = api.get(module, {})
        for n, desc in names.items():
            if n not in now:
                removed.append(f"{module}.{n}")
            elif now[n] != desc:
                changed.append(f"{module}.{n}: {desc} -> {now[n]}")
    for module, names in api.items():
        added += [f"{module}.{n}" for n in names if n not in frozen.get(module, {})]
    assert not (removed or changed or added), (
        "Public API differs from tests/api_snapshot.json "
        "(breaking changes need a major version; additions a minor one). "
        f"removed={removed} changed={changed} added={added}. "
        "If intended: MLKIT_UPDATE_API_SNAPSHOT=1 uv run pytest tests/test_public_api.py"
    )


def test_version_is_semver():
    import mlkit

    parts = mlkit.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)
