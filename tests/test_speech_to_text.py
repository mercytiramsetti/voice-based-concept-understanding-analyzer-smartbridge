"""Tests for speech_to_text.py.

Whisper's model is mocked out so these tests are fast and don't require
the ~150MB model download / network access — the audio normalization and
temp-file plumbing are what's actually under test here.
"""

import numpy as np
import pytest

import speech_to_text
from speech_to_text import _normalize_audio, speech_to_text as transcribe


def test_normalize_audio_scales_to_unit_range():
    audio = np.array([0.0, 2.0, -4.0, 1.0])
    normalized = _normalize_audio(audio)
    assert np.max(np.abs(normalized)) == pytest.approx(1.0)


def test_normalize_audio_handles_silent_input_without_dividing_by_zero():
    silence = np.zeros(10)
    normalized = _normalize_audio(silence)
    assert np.all(normalized == 0.0)


def test_speech_to_text_returns_stripped_transcript(monkeypatch, sine_wav_path):
    class FakeModel:
        def transcribe(self, path):
            return {"text": "  machine learning is a subset of AI  "}

    monkeypatch.setattr(speech_to_text, "_get_model", lambda: FakeModel())

    result = transcribe(sine_wav_path)
    assert result == "machine learning is a subset of AI"


def test_speech_to_text_cleans_up_normalized_temp_file(monkeypatch, sine_wav_path, tmp_path):
    captured_paths = []

    class FakeModel:
        def transcribe(self, path):
            captured_paths.append(path)
            return {"text": "ok"}

    monkeypatch.setattr(speech_to_text, "_get_model", lambda: FakeModel())

    transcribe(sine_wav_path)

    assert len(captured_paths) == 1
    # the normalized temp wav handed to whisper must be cleaned up afterward
    import os
    assert not os.path.exists(captured_paths[0])
