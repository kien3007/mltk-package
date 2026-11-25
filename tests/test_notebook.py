"""The full-functionality notebook must execute end to end (``pytest -m docs``)."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "mlkit_full_test.ipynb"


@pytest.mark.docs
def test_full_test_notebook_executes():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    client.execute()  # any failing check raises CellExecutionError
    summary = "".join(o.get("text", "") for o in nb.cells[-1].outputs if o.output_type == "stream")
    assert "kiểm tra đạt" in summary
