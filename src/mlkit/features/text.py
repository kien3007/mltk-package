"""Features from free-text columns: statistics plus a semantic representation.

``semantic="tfidf"`` (default) adds latent-semantic components (TF-IDF of words and character
n-grams reduced by truncated SVD) with scikit-learn only. ``semantic="embedding"`` uses a
sentence-transformers model (``pip install sentence-transformers``) reduced by PCA.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlkit.core.config import TaskConfig
from mlkit.core.registry import register
from mlkit.core.step import BaseStep
from mlkit.qc_eda.profile import ProfileOptions, infer_column


@register("features.text")
class TextFeatures(BaseStep):
    """Replace free-text columns with simple statistics.

    Columns are taken from ``columns`` or detected at fit time as semantic type ``text``
    (long strings / many words, mostly unique). Each becomes:
    ``<col>_n_chars``, ``<col>_n_words``, ``<col>_digit_ratio``, ``<col>_upper_ratio``,
    ``<col>_punct_ratio`` and ``<col>_is_missing``; the raw text is dropped (``drop_original``).

    With ``semantic`` set, ``<col>_sem0..k`` carry the meaning of the text (topics, sentiment):
    ``"tfidf"`` = TF-IDF + truncated SVD (fitted on the training rows only), ``"embedding"`` =
    sentence-transformers ``embedding_model`` + PCA; ``None`` = statistics only.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        drop_original: bool = True,
        profile_options: ProfileOptions | None = None,
        semantic: str | None = "tfidf",
        n_components: int = 32,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_features: int = 20_000,
    ) -> None:
        super().__init__()
        self.columns = columns
        self.drop_original = drop_original
        self.profile_options = profile_options
        self.semantic = semantic
        self.n_components = n_components
        self.embedding_model = embedding_model
        self.max_features = max_features
        self.text_columns_: list[Any] = []
        self.semantic_models_: dict[str, Any] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> TextFeatures:
        if self.columns is not None:
            self.text_columns_ = [c for c in self.columns if c in X.columns]
        else:
            candidates = [
                c for c in X.columns if X[c].dtype == object or pd.api.types.is_string_dtype(X[c])
            ]
            self.text_columns_ = [
                c
                for c in candidates
                if infer_column(X[c], str(c), self.profile_options).semantic_type == "text"
            ]
        if self.semantic not in (None, "tfidf", "embedding"):
            raise ValueError("semantic must be None, 'tfidf' or 'embedding'")
        self.semantic_models_ = {}
        if self.semantic:
            for col in self.text_columns_:
                model = self._fit_semantic(X[col].astype("string").fillna("").tolist())
                if model is not None:
                    self.semantic_models_[str(col)] = model
        self.report_.add(
            "text_features",
            "ok",
            f"Detected {len(self.text_columns_)} free-text columns: {self.text_columns_}"
            + (f"; semantic={self.semantic}" if self.semantic_models_ else ""),
        )
        return self

    def _fit_semantic(self, docs: list[str]) -> Any:
        """Fit the per-column semantic model; ``None`` when the text is too poor to model."""
        if self.semantic == "embedding":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                self.report_.add(
                    "text_features",
                    "warn",
                    "semantic='embedding' needs sentence-transformers (pip install sentence-transformers); "
                    "falling back to TF-IDF.",
                )
            else:
                from sklearn.decomposition import PCA

                encoder = SentenceTransformer(self.embedding_model)
                emb = np.asarray(encoder.encode(docs, show_progress_bar=False))
                k = max(1, min(self.n_components, emb.shape[1], len(docs) - 1))
                return ("embedding", encoder, PCA(n_components=k, random_state=0).fit(emb))
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion
        from sklearn.pipeline import Pipeline as SkPipeline

        vec = FeatureUnion(
            [
                (
                    "words",
                    TfidfVectorizer(max_features=self.max_features, sublinear_tf=True, min_df=2),
                ),
                (
                    "chars",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=self.max_features,
                        sublinear_tf=True,
                        min_df=2,
                    ),
                ),
            ]
        )
        try:
            m = vec.fit_transform(docs)
        except ValueError:  # empty vocabulary (no word appears twice)
            return None
        k = min(self.n_components, m.shape[1] - 1, len(docs) - 1)
        if k < 1:
            return None
        svd = TruncatedSVD(n_components=k, random_state=0)
        return ("tfidf", SkPipeline([("tfidf", vec), ("svd", svd)]).fit(docs), None)

    def _semantic(self, col: Any, docs: list[str]) -> np.ndarray | None:
        model = self.semantic_models_.get(str(col))
        if model is None:
            return None
        kind, first, second = model
        if kind == "embedding":
            return np.asarray(
                second.transform(np.asarray(first.encode(docs, show_progress_bar=False)))
            )
        return np.asarray(first.transform(docs))

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.text_columns_:
            if col not in out.columns:
                continue
            raw = out[col]
            s = raw.astype("string").fillna("")
            n_chars = s.str.len().astype(float)
            denom = n_chars.replace(0, np.nan)
            out[f"{col}_n_chars"] = n_chars
            out[f"{col}_n_words"] = s.str.count(r"\S+").astype(float)
            out[f"{col}_digit_ratio"] = (s.str.count(r"\d") / denom).fillna(0.0).astype(float)
            out[f"{col}_upper_ratio"] = (s.str.count(r"[A-Z]") / denom).fillna(0.0).astype(float)
            out[f"{col}_punct_ratio"] = (s.str.count(r"[^\w\s]") / denom).fillna(0.0).astype(float)
            out[f"{col}_is_missing"] = raw.isna().astype(int)
            sem = self._semantic(col, s.tolist())
            if sem is not None:
                out = pd.concat(
                    [
                        out,
                        pd.DataFrame(
                            sem,
                            index=out.index,
                            columns=[f"{col}_sem{k}" for k in range(sem.shape[1])],
                        ),
                    ],
                    axis=1,
                )
            if self.drop_original:
                out = out.drop(columns=[col])
        return out
