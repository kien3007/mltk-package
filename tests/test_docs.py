"""Executable documentation: every code block in docs/HUONG_DAN_SU_DUNG.md must work.

- ```python blocks run in order (each in a fresh namespace)
- ```bash blocks: every line starting with ``mlkit`` runs through the CLI entry point
- ```yaml blocks parse into TaskConfig; the full template is trained, ``steps:`` blocks are built
Blocks preceded by ``<!-- skip-test -->`` (servers, long benchmarks) are skipped.
Run with ``pytest -m docs`` (part of the CI benchmark job).
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "HUONG_DAN_SU_DUNG.md"
BLOCK = re.compile(r"(<!-- skip-test -->\s*)?```(\w+)\n(.*?)```", re.S)


def _blocks() -> list[tuple[str, str, int]]:
    text = GUIDE.read_text(encoding="utf-8")
    out = []
    for m in BLOCK.finditer(text):
        if m.group(1):
            continue
        line = text[: m.start()].count("\n") + 1
        out.append((m.group(2), m.group(3), line))
    return out


def _run_cli(command: str) -> int:
    from mlkit.cli import main

    redirect = None
    if ">" in command:
        command, redirect = (part.strip() for part in command.split(">", 1))
    argv = shlex.split(command.split(" #")[0])[1:]  # drop "mlkit" and trailing comments
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            main(argv)
            code = 0
        except SystemExit as exc:
            code = int(exc.code or 0)
    if redirect:
        Path(redirect).parent.mkdir(parents=True, exist_ok=True)
        Path(redirect).write_text(buf.getvalue(), encoding="utf-8")
    return code


@pytest.mark.docs
def test_guide_code_blocks(monkeypatch):
    monkeypatch.chdir(ROOT)
    blocks = _blocks()
    assert len(blocks) > 10
    for lang, code, line in blocks:
        where = f"{GUIDE.name}:{line}"
        if lang == "python":
            exec(compile(code, where, "exec"), {"__name__": "__guide__"})  # noqa: S102
            plt.close("all")
        elif lang == "bash":
            for cmd in (c.strip() for c in code.splitlines()):
                if cmd.startswith("mlkit "):
                    assert _run_cli(cmd) == 0, f"{where}: {cmd}"
        elif lang == "yaml":
            from mlkit.core import TaskConfig
            from mlkit.runner import build_pipeline_from_config, read_table, train

            data = yaml.safe_load(code) or {}
            cfg = TaskConfig(**({"target": "class"} | data))
            if cfg.steps:
                assert build_pipeline_from_config(cfg).steps, where
            elif cfg.data_path:  # the full template: it must actually train
                res = train(cfg, read_table(cfg.data_path))
                assert res.metrics["holdout"]["roc_auc"] > 0.8, where
