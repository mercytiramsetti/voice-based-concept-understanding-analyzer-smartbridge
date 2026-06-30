"""
scoring_engine.py — Combines semantic similarity, filler word ratio,
and audio features into a final comprehension score and classification.
"""


def evaluate_understanding(similarity, filler_ratio, audio):
    """
    Score breakdown:
      - Similarity:   50 (>0.7) | 30 (>0.4) | 10 (else)
      - Filler ratio: 20 (<0.05) | 10 (else)
      - Pause ratio:  15 (<0.25) | 5 (else)
      - RMS energy:   15 (>0.01) | 5 (else)
    Max score: 100
    Returns (score, level, color).
    """
    score = 0
    score += 50 if similarity > 0.7 else 30 if similarity > 0.4 else 10
    score += 20 if filler_ratio < 0.05 else 10
    score += 15 if audio["pause_ratio"] < 0.25 else 5
    score += 15 if audio["rms_energy"] > 0.01 else 5

    if score >= 80:
        return score, "Strong Understanding", "#2ecc71"
    elif score >= 50:
        return score, "Moderate Understanding", "#f39c12"
    return score, "Poor Understanding", "#e74c3c"
