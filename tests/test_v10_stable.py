"""v1.0: signed bundles (ADR-012), removed config keys and aliases."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mlkit.core import TaskConfig
from mlkit.core.bundle import load_bundle, verify_bundle
from mlkit.core.errors import ValidationError


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(10)
    X = pd.DataFrame({"a": rng.normal(size=600), "b": rng.normal(size=600)})
    return X.assign(y=((X["a"] + rng.normal(scale=0.8, size=600)) > 0).astype(int))


def _train(df, out, monkeypatch, key=None):
    from mlkit.runner import train

    if key is None:
        monkeypatch.delenv("MLKIT_BUNDLE_KEY", raising=False)
    else:
        monkeypatch.setenv("MLKIT_BUNDLE_KEY", key)
    train(
        TaskConfig(target="y", cv_folds=3, modelling={"candidates": ["linear"]}), df, output_dir=out
    )
    monkeypatch.delenv("MLKIT_BUNDLE_KEY", raising=False)


class TestSignedBundles:
    def test_signed_roundtrip_and_wrong_key(self, frame, tmp_path, monkeypatch):
        _train(frame, tmp_path / "b", monkeypatch, key="s3cret")
        manifest = json.loads((tmp_path / "b" / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["signature"]["algorithm"] == "hmac-sha256"
        pipe, cfg, _ = load_bundle(tmp_path / "b", signing_key="s3cret")
        assert pipe.predict(frame.drop(columns="y").head(3)).shape == (3,)
        with pytest.raises(ValidationError, match="signature"):
            load_bundle(tmp_path / "b", signing_key="other")
        # without a key the bundle still loads (checksums only)
        load_bundle(tmp_path / "b")

    def test_unsigned_bundle_rejected_when_key_configured(self, frame, tmp_path, monkeypatch):
        _train(frame, tmp_path / "u", monkeypatch)
        assert "signature" not in json.loads((tmp_path / "u" / "manifest.json").read_text())
        monkeypatch.setenv("MLKIT_BUNDLE_KEY", "s3cret")
        with pytest.raises(ValidationError, match="not signed"):
            load_bundle(tmp_path / "u")

    def test_forged_manifest_detected(self, frame, tmp_path, monkeypatch):
        """Rewriting a file *and* its checksum still fails: the signature covers the manifest."""
        import hashlib

        _train(frame, tmp_path / "f", monkeypatch, key="s3cret")
        cfg_file = tmp_path / "f" / "config.yaml"
        cfg_file.write_text(cfg_file.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
        mpath = tmp_path / "f" / "manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        manifest["files"]["config.yaml"] = hashlib.sha256(cfg_file.read_bytes()).hexdigest()
        mpath.write_text(json.dumps(manifest), encoding="utf-8")
        verify_bundle(tmp_path / "f")  # checksums alone are fooled
        with pytest.raises(ValidationError, match="signature"):
            verify_bundle(tmp_path / "f", signing_key="s3cret")


def test_removed_config_keys_warn_and_are_ignored():
    with pytest.warns(UserWarning, match="removed in mlkit 1.0"):
        cfg = TaskConfig(target="y", preprocessing={"x": 1}, selection={"k": 2})
    assert not hasattr(cfg, "preprocessing") or getattr(cfg, "preprocessing", None) is None
    assert "preprocessing" not in cfg.model_dump()


def test_removed_aliases():
    from mlkit.modelling import ModelArena
    from mlkit.monitoring import DriftMonitor

    assert not hasattr(ModelArena(), "custom_models")
    assert "baseline_distributions_" not in dir(DriftMonitor)
