from scoring_engine import evaluate_understanding


def _audio(pause_ratio, rms_energy):
    return {"pause_ratio": pause_ratio, "rms_energy": rms_energy}


def test_all_strong_signals_yield_max_score_and_strong_level():
    score, level, color = evaluate_understanding(
        similarity=0.9, filler_ratio=0.0, audio=_audio(0.1, 0.05)
    )
    assert score == 100
    assert level == "Strong Understanding"
    assert color == "#2ecc71"


def test_mid_range_signals_yield_moderate_level():
    # 30 (similarity 0.4-0.7) + 10 (filler>=0.05) + 5 (pause>=0.25) + 15 (energy>0.01) = 60
    score, level, color = evaluate_understanding(
        similarity=0.5, filler_ratio=0.2, audio=_audio(0.3, 0.05)
    )
    assert score == 60
    assert level == "Moderate Understanding"
    assert color == "#f39c12"


def test_all_weak_signals_yield_poor_level():
    score, level, color = evaluate_understanding(
        similarity=0.1, filler_ratio=0.5, audio=_audio(0.9, 0.001)
    )
    assert score == 10 + 10 + 5 + 5
    assert level == "Poor Understanding"
    assert color == "#e74c3c"


def test_score_boundaries_are_exclusive_on_the_high_side():
    """Values exactly at a threshold should NOT get the higher bucket —
    evaluate_understanding uses strict '>' / '<' comparisons."""
    score, _, _ = evaluate_understanding(
        similarity=0.7, filler_ratio=0.05, audio=_audio(0.25, 0.01)
    )
    # similarity==0.7 -> 30 (not 50); filler==0.05 -> 10 (not 20);
    # pause==0.25 -> 5 (not 15); energy==0.01 -> 5 (not 15)
    assert score == 30 + 10 + 5 + 5


def test_classification_boundary_at_80_is_strong():
    # 30 (similarity 0.4-0.7) + 20 (filler<0.05) + 15 (pause<0.25) + 15 (energy>0.01) = 80
    score, level, _ = evaluate_understanding(
        similarity=0.5, filler_ratio=0.0, audio=_audio(0.1, 0.02)
    )
    assert score == 80
    assert level == "Strong Understanding"


def test_classification_boundary_at_50_is_moderate_not_poor():
    score, level, _ = evaluate_understanding(
        similarity=0.5, filler_ratio=0.2, audio=_audio(0.9, 0.001)
    )
    assert score == 30 + 10 + 5 + 5
    assert score >= 50
    assert level == "Moderate Understanding"
