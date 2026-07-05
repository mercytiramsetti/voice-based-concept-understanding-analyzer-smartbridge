"""Tests for semantic_eval.py.

The Sentence-BERT model itself is mocked out so these tests run in
milliseconds and don't require the ~90MB model download / network access —
only the pure similarity math is under test here.
"""

import numpy as np
import pytest

import semantic_eval
from semantic_eval import _cosine_similarity, _normalize_score, semantic_similarity


def test_cosine_similarity_identical_vectors_is_one():
    vec = np.array([1.0, 2.0, 3.0])
    assert _cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_does_not_divide_by_zero():
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert _cosine_similarity(a, b) == 0.0


def test_normalize_score_clips_to_zero_one_range():
    assert _normalize_score(-0.5) == 0.0
    assert _normalize_score(1.5) == 1.0
    assert _normalize_score(0.42) == pytest.approx(0.42)


def test_semantic_similarity_returns_zero_for_empty_input():
    assert semantic_similarity("", "some reference text") == 0.0
    assert semantic_similarity("some student text", "") == 0.0


def test_semantic_similarity_uses_model_embeddings(monkeypatch):
    class FakeModel:
        def encode(self, texts):
            # return identical embeddings regardless of input ->
            # forces a perfect-similarity result so we can assert on it
            return np.array([[1.0, 0.0], [1.0, 0.0]])

    monkeypatch.setattr(semantic_eval, "_get_model", lambda: FakeModel())

    result = semantic_similarity("machine learning explanation", "reference text")
    assert result == pytest.approx(1.0)


def test_semantic_similarity_low_overlap_scores_low(monkeypatch):
    class FakeModel:
        def encode(self, texts):
            return np.array([[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(semantic_eval, "_get_model", lambda: FakeModel())

    result = semantic_similarity("unrelated text", "reference text")
    assert result == pytest.approx(0.0)
