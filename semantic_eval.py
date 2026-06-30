"""
semantic_eval.py — Sentence-BERT semantic similarity evaluation.
Compares student explanation against a reference concept.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# load model once to avoid reloading on every call
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _cosine_similarity(vec_a, vec_b):
    # dot product divided by product of magnitudes
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm == 0:
        return 0.0
    return dot / norm


def _normalize_score(raw_score):
    # cosine similarity is in [-1, 1]; clip to [0, 1] for clean display
    return float(np.clip(raw_score, 0.0, 1.0))


def semantic_similarity(student_text, reference_text):
    """
    Generate Sentence-BERT embeddings for both texts and return
    a normalized cosine similarity score in [0, 1].
    """
    if not student_text or not reference_text:
        return 0.0

    model = _get_model()
    embeddings = model.encode([student_text, reference_text])
    raw_score = _cosine_similarity(embeddings[0], embeddings[1])
    return _normalize_score(raw_score)
